"""Run a trained HPT/HNet checkpoint in PushShapesEnv with optional live GUI.

Recreated minimal version. Loads the algo via the same path
``scripts/smoke_sim_eval.py`` uses, then drives PushShapesEnv in closed
loop using ``algo.sim_init_state`` + ``algo.sim_predict_step``. Renders
each step; with ``--render human`` opens a live cv2 window.

Initial state: read frame 0 of a chosen ``episode_*.zarr`` from the
dataset (replay). Obstacle layouts are not stored in zarr — for
``obstacle_level > 0`` episodes pass ``--allow-fresh-reset`` to accept a
fresh random layout instead.

Usage::

    python -m Tsimulation.examples.run_checkpoint_policy \\
        --dataset data/pushshapes_demos/circle \\
        --checkpoint logs/.../checkpoints/last.ckpt \\
        --episode 0 --render human --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import zarr
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.eval.eval_sim import _format_pushshapes_obs
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from Tsimulation.collect.zarr_writer import GOAL_KEY, STATE_KEY
from Tsimulation.pushshapes.env import PushShapesEnv
from Tsimulation.pushshapes.render import surface_to_rgb_array


def _load_algo_from_ckpt(ckpt_path: Path, config_path: Path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters") or ckpt.get("hparams") or {}
    if "config_tree" in hparams:
        cfg = OmegaConf.create(hparams["config_tree"])
    else:
        cfg = OmegaConf.load(str(config_path))
    norm_state = hparams.get("norm_stats_state")
    if norm_state is None:
        raise SystemExit("ckpt has no norm_stats_state in hyper_parameters")
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
        print(f"[load] missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"[load] unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    return algo


def _episode_obs_level(p: Path) -> int | None:
    for part in p.stem.split("_"):
        if part.startswith("obs") and part[3:].isdigit():
            return int(part[3:])
    return None


def _pick_episode(dataset_dir: Path, episode: int | None, seed: int, allow_fresh_reset: bool) -> Path:
    paths = sorted(p for p in dataset_dir.iterdir()
                   if p.is_dir() and p.name.startswith("episode_") and p.name.endswith(".zarr"))
    if not paths:
        raise FileNotFoundError(f"no episode_*.zarr under {dataset_dir}")
    if episode is not None:
        match = [p for p in paths if p.name.endswith(f"_{episode:06d}.zarr")]
        if not match:
            raise FileNotFoundError(f"no episode {episode} under {dataset_dir}")
        return match[0]
    rng = np.random.default_rng(seed)
    candidates = paths if allow_fresh_reset else [p for p in paths if _episode_obs_level(p) == 0]
    if not candidates:
        raise FileNotFoundError(
            "no obs0 episodes found; pass --allow-fresh-reset to use any episode "
            "with a fresh random obstacle layout"
        )
    return candidates[int(rng.integers(len(candidates)))]


def _render_overlay(frame: np.ndarray, *, step: int, coverage: float,
                    predicted_actions: np.ndarray | None,
                    active_idx: int) -> np.ndarray:
    img = frame.copy()
    cv2.putText(img, f"step={step} cov={coverage:.3f}", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if predicted_actions is not None and len(predicted_actions) > 1:
        pts = predicted_actions.astype(np.int32)
        for i in range(len(pts) - 1):
            color = (0, 200, 255) if i >= active_idx else (80, 80, 80)
            cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), color, 1, cv2.LINE_AA)
    return img


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="directory containing episode_*.zarr stores")
    parser.add_argument("--checkpoint", required=True,
                        help="path to a Lightning .ckpt")
    parser.add_argument("--config-path", default=None,
                        help="path to .hydra/config.yaml; auto-discovered "
                             "from the ckpt's parent run dir if omitted")
    parser.add_argument("--episode", type=int, default=None,
                        help="episode index to seed from; random obs0 by default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--render", default="human", choices=["human", "rgb_array", "none"])
    parser.add_argument("--save", default=None, help="optional mp4 output path")
    parser.add_argument("--live-predictions", action="store_true",
                        help="overlay the predicted action chunk in the live window")
    parser.add_argument("--query-frequency", type=int, default=8,
                        help="env steps per predicted chunk; 1 = per-step replanning")
    parser.add_argument("--allow-fresh-reset", action="store_true",
                        help="allow obstacle episodes to run on a fresh random layout")
    parser.add_argument("--coverage-threshold", type=float, default=0.7)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).resolve()
    ckpt_path = Path(args.checkpoint).resolve()
    if args.config_path is None:
        guess = ckpt_path.parent.parent / ".hydra" / "config.yaml"
        if not guess.exists():
            raise SystemExit(
                f"could not auto-find .hydra/config.yaml under {ckpt_path.parent.parent};"
                " pass --config-path"
            )
        config_path = guess
    else:
        config_path = Path(args.config_path).resolve()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    print(f"[init] device={device} ckpt={ckpt_path} cfg={config_path}")

    algo = _load_algo_from_ckpt(ckpt_path, config_path)
    algo.nets = algo.nets.to(device)
    algo.device = device
    algo.nets.eval()

    episode_path = _pick_episode(dataset_dir, args.episode, args.seed, args.allow_fresh_reset)
    obs_level = _episode_obs_level(episode_path) or 0
    print(f"[episode] {episode_path.name}  obs_level={obs_level}")

    ep = zarr.open(str(episode_path), mode="r")
    env_args = json.loads(ep.attrs["env_args"]) if "env_args" in ep.attrs else {}
    state0 = np.asarray(ep[STATE_KEY][0], dtype=np.float32).reshape(-1)
    goal0 = np.asarray(ep[GOAL_KEY][0], dtype=np.float32).reshape(-1)
    agent_pos = (float(state0[0]), float(state0[1]))
    object_pose = (float(state0[2]), float(state0[3]), float(state0[4]))
    goal_pose = (float(goal0[0]), float(goal0[1]), float(goal0[2]))

    render_mode = None if args.render == "none" else args.render
    env = PushShapesEnv(**{**env_args, "render_mode": render_mode})

    if obs_level == 0 or args.allow_fresh_reset:
        env.reset(seed=args.seed)
        if obs_level == 0:
            env.set_state(agent_pos=agent_pos, object_pose=object_pose, goal_pose=goal_pose)
    else:
        raise SystemExit(
            f"episode {episode_path.name} has obstacle_level={obs_level};"
            " obstacle layouts aren't stored in zarr — pass --allow-fresh-reset to use a"
            " random layout with the same task config"
        )

    emb_id = get_embodiment_id("pushshapes_sim")
    ac_key = algo.ac_keys["pushshapes_sim"]
    rollout_state = algo.sim_init_state(batch_size=1, T_max=args.steps, device=device, emb_id=emb_id)
    T = rollout_state.get("T_max", args.steps)
    chunk_size = int(algo.nets["policy"].action_horizon)
    qf = max(1, int(args.query_frequency))

    video_frames: list[np.ndarray] = []
    cached_chunk: np.ndarray | None = None
    chunk_offset = 0
    last_coverage = 0.0

    for step in range(T):
        obs_env = env._get_obs()
        obs_raw = _format_pushshapes_obs(obs_env, device)
        obs_norm = algo.norm_stats.normalize(obs_raw, emb_id)

        # Re-plan when chunk empty or exhausted (every qf steps).
        if cached_chunk is None or chunk_offset >= qf or chunk_offset >= chunk_size:
            # Force a new chunk by clearing state's chunk and calling t=0
            # path inside sim_predict_step; easiest: re-init.
            rollout_state["action_chunk"] = None
            rollout_state["chunk_idx"] = 0
            # Drive sim_predict_step for the whole chunk to extract it.
            chunk_actions = []
            for ci in range(chunk_size):
                a_norm = algo.sim_predict_step(rollout_state, obs_norm, ci, emb_id)
                a_world = (
                    algo.norm_stats.unnormalize({ac_key: a_norm.squeeze(0)}, emb_id)[ac_key]
                    .detach().cpu().numpy().reshape(-1)
                )
                chunk_actions.append(a_world[:2])
            cached_chunk = np.stack(chunk_actions, axis=0)
            chunk_offset = 0

        action = cached_chunk[min(chunk_offset, len(cached_chunk) - 1)].astype(np.float32)
        chunk_offset += 1

        _, _, terminated, truncated, info = env.step(action)
        last_coverage = float(info.get("coverage", 0.0))

        frame_rgb = np.ascontiguousarray(surface_to_rgb_array(env.world_surface()))
        overlay = (cached_chunk if args.live_predictions else None)
        canvas = _render_overlay(frame_rgb, step=step + 1, coverage=last_coverage,
                                 predicted_actions=overlay, active_idx=chunk_offset - 1)

        if render_mode == "human":
            cv2.imshow("PushShapes Rollout", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("stopped by user")
                break

        if args.save is not None:
            video_frames.append(canvas)

        print(f"step={step:03d} action=({action[0]:7.2f},{action[1]:7.2f}) "
              f"coverage={last_coverage:.3f}")

        if terminated or truncated:
            break
        if last_coverage >= args.coverage_threshold:
            print(f"[success] coverage {last_coverage:.3f} >= {args.coverage_threshold}")
            break

    if args.save is not None and video_frames:
        from egomimic.utils.video_utils import save_preview_mp4
        save_preview_mp4(np.stack(video_frames, axis=0), Path(args.save), fps=30)
        print(f"saved {len(video_frames)} frames -> {args.save}")

    if render_mode == "human":
        cv2.destroyAllWindows()
    print(f"final_coverage={last_coverage:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
