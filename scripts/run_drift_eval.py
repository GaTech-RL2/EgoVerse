"""GT-replay vs model-rollout drift eval (HPT vs H-Net on the same episode).

For each picked val episode:
  1. GT replay: env.set_state(state[0]), step with raw GT actions from zarr.
  2. HPT rollout: same init, step with HPT.inference_step.
  3. H-Net rollout: same init, step with H-Net.inference_step.
Compute per-step state delta (agent / object position / object angle) and
coverage curves for each. Save:
  - 3-panel side-by-side mp4 (GT | HPT | H-Net) per episode
  - drift_curves.png per episode (4 subplots: agent drift / obj drift /
    angle drift / coverage, all three series overlaid)
  - metrics.json per episode + a summary.json across episodes

Run from EgoVerse7 root on a GPU node:
    PYTHONPATH=. .venv/bin/python scripts/run_drift_eval.py \
        --hpt-ckpt <path>/checkpoints/epoch=N-step=M.ckpt \
        --hpt-config-path <path>/.hydra/config.yaml \
        --hnet-ckpt <path>/checkpoints/epoch=N-step=M.ckpt \
        --hnet-config-path <path>/.hydra/config.yaml \
        --episodes-dir /storage/project/r-dxu345-0/paphiwetsa3/datasets/circle/basic \
        --n-episodes 4 --max-steps 600 \
        --out-dir drift_eval_out/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.io as tvio
import zarr
from hydra.utils import instantiate
from omegaconf import OmegaConf


# --------------------------------------------------------------------- #
# Checkpoint loader (copied from smoke_sim_eval, generic for both algos).
# --------------------------------------------------------------------- #


def load_algo_from_ckpt(ckpt_path: str, config_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(config_path)

    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    hparams = ckpt.get("hyper_parameters") or ckpt.get("hparams") or {}
    norm_state = hparams.get("norm_stats_state")
    if norm_state is None:
        raise SystemExit(f"hyper_parameters has no norm_stats_state in {ckpt_path}")
    norm_stats = MultiDataset.from_state(norm_state)
    algo = instantiate(cfg.model.robomimic_model, norm_stats=norm_stats)

    state_dict = ckpt["state_dict"]
    new_sd = {}
    for k, v in state_dict.items():
        for prefix in ("nets.", "model.nets."):
            if k.startswith(prefix):
                new_sd[k[len(prefix):]] = v
                break
        else:
            new_sd[k] = v
    missing, unexpected = algo.nets.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"  [{Path(ckpt_path).name}] missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"  [{Path(ckpt_path).name}] unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
    algo.nets = algo.nets.to(device)
    algo.device = device
    algo.nets.eval()
    return algo, cfg


# --------------------------------------------------------------------- #
# Env helpers.
# --------------------------------------------------------------------- #


def make_env(env_kwargs: dict):
    from Tsimulation.pushshapes import PushShapesEnv

    kwargs = dict(env_kwargs)
    kwargs.setdefault("render_mode", "rgb_array")
    return PushShapesEnv(**kwargs)


def reset_env_to_state(env, state_5: np.ndarray, seed: int, goal_pose=None):
    """state_5 = (agent_x, agent_y, obj_x, obj_y, obj_angle)."""
    env.reset(seed=seed)
    s = np.asarray(state_5, dtype=np.float32).reshape(-1)
    gp = None
    if goal_pose is not None:
        g = np.asarray(goal_pose, dtype=np.float32).reshape(-1)
        gp = (float(g[0]), float(g[1]), float(g[2]))
    env.set_state(
        agent_pos=(float(s[0]), float(s[1])),
        object_pose=(float(s[2]), float(s[3]), float(s[4])),
        goal_pose=gp,
    )


def env_obs_to_zarr_dict(obs_env: dict, device: torch.device) -> dict:
    state_5 = np.concatenate(
        [obs_env["agent_pos"], obs_env["object_pose"]], axis=0
    ).astype(np.float32)
    image_chw = np.transpose(obs_env["image"], (2, 0, 1)).astype(np.float32) / 255.0
    return {
        "state_agent_obj": torch.from_numpy(state_5).unsqueeze(0).to(device),
        "front_img_1": torch.from_numpy(image_chw).unsqueeze(0).to(device),
    }


def env_state_5(env) -> np.ndarray:
    obs = env._get_obs()
    return np.concatenate([obs["agent_pos"], obs["object_pose"]], axis=0).astype(
        np.float32
    )


# --------------------------------------------------------------------- #
# Rollouts.
# --------------------------------------------------------------------- #


def gt_replay(env_kwargs, init_state_5, gt_actions, seed, max_steps, goal_pose=None):
    env = make_env(env_kwargs)
    reset_env_to_state(env, init_state_5, seed=seed, goal_pose=goal_pose)
    states = [env_state_5(env)]
    coverages: List[float] = []
    frames: List[np.ndarray] = []
    actions_history: List[np.ndarray] = []
    T = min(len(gt_actions), max_steps)
    for t in range(T):
        a = np.asarray(gt_actions[t], dtype=np.float32).reshape(-1)[:2]
        actions_history.append(a)
        _, _, terminated, _, info = env.step(a)
        states.append(env_state_5(env))
        coverages.append(float(info.get("coverage", 0.0)))
        f = env.render()
        if f is not None:
            f = draw_action_overlay(f, a, actions_history)
            frames.append(np.ascontiguousarray(f))
        if terminated:
            break
    return np.stack(states), np.array(coverages, dtype=np.float32), frames, np.asarray(actions_history, dtype=np.float32)


def model_rollout(env_kwargs, init_state_5, algo, emb_id, seed, max_steps, device, goal_pose=None):
    env = make_env(env_kwargs)
    reset_env_to_state(env, init_state_5, seed=seed, goal_pose=goal_pose)
    states = [env_state_5(env)]
    coverages: List[float] = []
    frames: List[np.ndarray] = []
    actions_history: List[np.ndarray] = []
    for t in range(max_steps):
        obs_env = env._get_obs()
        obs_zarr = env_obs_to_zarr_dict(obs_env, device)
        a = algo.inference_step(obs_zarr, t, emb_id, T_max=max_steps)
        a = np.asarray(a, dtype=np.float32).reshape(-1)[:2]
        actions_history.append(a)
        _, _, terminated, _, info = env.step(a)
        states.append(env_state_5(env))
        coverages.append(float(info.get("coverage", 0.0)))
        f = env.render()
        if f is not None:
            f = draw_action_overlay(f, a, actions_history)
            frames.append(np.ascontiguousarray(f))
        if terminated:
            break
    return np.stack(states), np.array(coverages, dtype=np.float32), frames, np.asarray(actions_history, dtype=np.float32)


def draw_action_overlay(frame, a_now, history, world=512):
    """Cyan polyline trail + yellow marker (matches eval_sim style)."""
    frame = np.ascontiguousarray(frame)
    H, W, _ = frame.shape
    pts = np.array(history, dtype=np.float32) / world * np.array([W, H], dtype=np.float32)
    pts = pts.astype(np.int32)
    if len(pts) >= 2:
        cv2.polylines(frame, [pts], False, (255, 255, 0), 1, cv2.LINE_AA)  # BGR cyan
    cv2.circle(frame, tuple(pts[-1].tolist()), 3, (0, 0, 0), -1)
    cv2.circle(frame, tuple(pts[-1].tolist()), 2, (0, 255, 255), -1)  # BGR yellow
    return frame


# --------------------------------------------------------------------- #
# Drift + viz.
# --------------------------------------------------------------------- #


def compute_drift(states_model: np.ndarray, states_gt: np.ndarray) -> Dict[str, np.ndarray]:
    T = min(len(states_model), len(states_gt))
    delta = states_model[:T] - states_gt[:T]
    angle_delta = np.arctan2(np.sin(delta[:, 4]), np.cos(delta[:, 4]))
    return {
        "agent_drift": np.linalg.norm(delta[:, :2], axis=-1),
        "obj_drift": np.linalg.norm(delta[:, 2:4], axis=-1),
        "angle_drift": np.abs(angle_delta),
    }


def make_panel_video(frame_lists: List[List[np.ndarray]], labels: List[str]) -> np.ndarray:
    max_T = max(len(fl) for fl in frame_lists)
    H, W, _ = frame_lists[0][0].shape
    out = []
    for t in range(max_T):
        panels = []
        for i, fl in enumerate(frame_lists):
            f = fl[min(t, len(fl) - 1)].copy()
            label = f"{labels[i]} t={min(t,len(fl)-1)}"
            cv2.putText(f, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
            cv2.putText(f, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            panels.append(f)
        out.append(np.concatenate(panels, axis=1))
    return np.stack(out, axis=0)


def plot_drift_curves(drifts: Dict[str, Dict[str, np.ndarray]],
                      covs: Dict[str, np.ndarray],
                      out_path: Path,
                      title: str = ""):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    for name, d in drifts.items():
        axes[0].plot(d["agent_drift"], label=name)
        axes[1].plot(d["obj_drift"], label=name)
        axes[2].plot(d["angle_drift"], label=name)
    for name, c in covs.items():
        axes[3].plot(c, label=name)
    axes[0].set_ylabel("agent xy drift")
    axes[1].set_ylabel("object xy drift")
    axes[2].set_ylabel("object angle drift (rad)")
    axes[3].set_ylabel("coverage")
    axes[3].set_xlabel("env step")
    for ax in axes:
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------- #
# Episode loading.
# --------------------------------------------------------------------- #


def load_episode_zarr(zarr_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (state, actions, goal_pose[0], T_real).

    Zarrs are written with a chunk-alignment zero pad on numeric arrays
    (e.g. 750 real frames padded to 800). The real demo length is recorded
    as ``z.attrs["total_frames"]`` by the writer; using that directly
    bypasses the padding and matches what the training dataset uses.
    """
    z = zarr.open(str(zarr_path), mode="r")
    T_real = int(z.attrs.get("total_frames", -1))
    if T_real <= 0:
        T_real = int(np.asarray(z["actions"]).shape[0])
    state = np.asarray(z["observations.state"][:T_real], dtype=np.float32)
    actions = np.asarray(z["actions"][:T_real], dtype=np.float32)
    goal0 = np.asarray(z["goal_pose"][0], dtype=np.float32)
    return state, actions, goal0, T_real


# --------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hpt-ckpt", required=True)
    p.add_argument("--hpt-config-path", required=True)
    p.add_argument("--hnet-ckpt", required=True)
    p.add_argument("--hnet-config-path", required=True)
    p.add_argument(
        "--episodes-dir",
        required=True,
        help="Folder of episode_*.zarr — picks first N alphabetically.",
    )
    p.add_argument("--n-episodes", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--emb-id", type=int, default=15)  # pushshapes_sim
    p.add_argument("--out-dir", default="drift_eval_out")
    p.add_argument(
        "--env-kwargs",
        default='{"object_shape": "T", "pusher_shape": "circle", "obstacle_level": 0, "image_size": 96}',
        help="JSON for PushShapesEnv kwargs.",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_kwargs = json.loads(args.env_kwargs)

    print(f"[load] HPT  ckpt: {args.hpt_ckpt}")
    hpt, _ = load_algo_from_ckpt(args.hpt_ckpt, args.hpt_config_path, device)
    print(f"[load] HNet ckpt: {args.hnet_ckpt}")
    hnet, _ = load_algo_from_ckpt(args.hnet_ckpt, args.hnet_config_path, device)

    eps_dir = Path(args.episodes_dir)
    zarrs = sorted(eps_dir.glob("episode_*.zarr"))[: args.n_episodes]
    if not zarrs:
        raise SystemExit(f"no episodes found under {eps_dir}")
    print(f"[episodes] picked {len(zarrs)} from {eps_dir}:")
    for z in zarrs:
        print(f"    {z.name}")

    summary = []
    for ep_idx, ep_path in enumerate(zarrs):
        print(f"\n[ep {ep_idx}] {ep_path.name}")
        state_zarr, actions_zarr, goal_zarr, T_real = load_episode_zarr(ep_path)
        init_state = state_zarr[0]
        T_gt = min(len(actions_zarr), args.max_steps)
        print(f"  T_real={T_real}  T_gt={T_gt}  goal={goal_zarr.tolist()}  init_state={init_state.tolist()}")

        seed = args.seed + ep_idx
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            print("  [gt replay]")
            gt_states, gt_cov, gt_frames, gt_actions_arr = gt_replay(
                env_kwargs, init_state, actions_zarr, seed=seed, max_steps=args.max_steps,
                goal_pose=goal_zarr,
            )
            print(f"    T={len(gt_states)-1}  final_cov={gt_cov[-1]:.3f}")

            print("  [hpt rollout]")
            hpt_states, hpt_cov, hpt_frames, hpt_actions_arr = model_rollout(
                env_kwargs, init_state, hpt, args.emb_id, seed=seed,
                max_steps=args.max_steps, device=device, goal_pose=goal_zarr,
            )
            print(f"    T={len(hpt_states)-1}  final_cov={hpt_cov[-1]:.3f}")

            print("  [hnet rollout]")
            hnet_states, hnet_cov, hnet_frames, hnet_actions_arr = model_rollout(
                env_kwargs, init_state, hnet, args.emb_id, seed=seed,
                max_steps=args.max_steps, device=device, goal_pose=goal_zarr,
            )
            print(f"    T={len(hnet_states)-1}  final_cov={hnet_cov[-1]:.3f}")

        drifts = {
            "HPT-vs-GT": compute_drift(hpt_states, gt_states),
            "HNet-vs-GT": compute_drift(hnet_states, gt_states),
        }
        covs = {"GT": gt_cov, "HPT": hpt_cov, "HNet": hnet_cov}

        ep_dir = out_dir / f"ep{ep_idx:02d}_{ep_path.stem}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        if all(len(fl) > 0 for fl in [gt_frames, hpt_frames, hnet_frames]):
            ims = make_panel_video(
                [gt_frames, hpt_frames, hnet_frames], ["GT", "HPT", "HNet"]
            )
            video_path = ep_dir / "panel_gt_hpt_hnet.mp4"
            tvio.write_video(
                str(video_path), torch.from_numpy(ims), fps=30, video_codec="h264"
            )
            print(f"  wrote {video_path}  shape={ims.shape}")

        plot_path = ep_dir / "drift_curves.png"
        plot_drift_curves(drifts, covs, plot_path, title=ep_path.stem)
        print(f"  wrote {plot_path}")

        ep_metrics = {
            "episode": ep_path.name,
            "T_gt": int(T_gt),
            "T_hpt": int(len(hpt_states) - 1),
            "T_hnet": int(len(hnet_states) - 1),
            "final_coverage_gt": float(gt_cov[-1]) if len(gt_cov) else 0.0,
            "final_coverage_hpt": float(hpt_cov[-1]) if len(hpt_cov) else 0.0,
            "final_coverage_hnet": float(hnet_cov[-1]) if len(hnet_cov) else 0.0,
            "final_agent_drift_hpt": float(drifts["HPT-vs-GT"]["agent_drift"][-1]),
            "final_agent_drift_hnet": float(drifts["HNet-vs-GT"]["agent_drift"][-1]),
            "final_obj_drift_hpt": float(drifts["HPT-vs-GT"]["obj_drift"][-1]),
            "final_obj_drift_hnet": float(drifts["HNet-vs-GT"]["obj_drift"][-1]),
            "mean_agent_drift_hpt": float(np.mean(drifts["HPT-vs-GT"]["agent_drift"])),
            "mean_agent_drift_hnet": float(np.mean(drifts["HNet-vs-GT"]["agent_drift"])),
            "mean_obj_drift_hpt": float(np.mean(drifts["HPT-vs-GT"]["obj_drift"])),
            "mean_obj_drift_hnet": float(np.mean(drifts["HNet-vs-GT"]["obj_drift"])),
            "first10_actions_gt":   gt_actions_arr[:10].tolist(),
            "first10_actions_hpt":  hpt_actions_arr[:10].tolist(),
            "first10_actions_hnet": hnet_actions_arr[:10].tolist(),
        }
        (ep_dir / "metrics.json").write_text(json.dumps(ep_metrics, indent=2))
        summary.append(ep_metrics)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
