from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from egomimic.eval import eval_sim


@torch.no_grad()
def _rollout_one_chunk32(
    self,
    sample: Dict[str, torch.Tensor],
    seq_len: int,
    emb_id: int,
    ep_idx: int,
) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
    """HPT chunk execution rollout.

    Predict one action_horizon chunk, execute it sequentially, and only replan
    after the chunk is consumed. This intentionally disables the current
    per-frame replan + temporal ensemble behavior in eval_sim.SimRolloutEval.
    """
    algo = self.model
    device = self.trainer.lightning_module.device
    env = self._get_env()

    obs_env = self._init_env(env, sample, ep_seed_offset=ep_idx, emb_id=emb_id)
    T_eff = self.max_steps if self.max_steps is not None else int(seq_len)

    if not hasattr(algo, "sim_init_state") or not hasattr(algo, "sim_predict_step"):
        raise RuntimeError(
            f"Algo {type(algo).__name__} does not implement "
            "sim_init_state / sim_predict_step."
        )

    state = algo.sim_init_state(batch_size=1, T_max=T_eff, device=device, emb_id=emb_id)
    T = state.get("T_max", T_eff)
    ac_key = getattr(algo, "resolved_ac_keys", algo.ac_keys)[emb_id]
    obs_formatter = eval_sim._OBS_FORMATTERS[self.embodiment_name]

    frames: List[np.ndarray] = []
    actions_taken: List[np.ndarray] = []
    last_coverage = 0.0
    chunk_starts: list[int] = []

    for t in range(T):
        obs_raw = obs_formatter(obs_env, device)
        obs_norm = algo.norm_stats.normalize(obs_raw, emb_id)

        was_new_chunk = state.get("action_chunk") is None or int(state.get("chunk_idx", 0)) >= 32
        a_t_norm = algo.sim_predict_step(state, obs_norm, t, emb_id).squeeze(0)
        if was_new_chunk:
            chunk_starts.append(t)

        a_t_world = (
            algo.norm_stats.unnormalize({ac_key: a_t_norm}, emb_id)[ac_key]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        action_xy = np.array([float(a_t_world[0]), float(a_t_world[1])], dtype=np.float64)

        obs_env, _, terminated, _, info = env.step(action_xy)
        last_coverage = max(last_coverage, float(info.get("coverage", 0.0)))
        actions_taken.append(action_xy)

        frame = env.render()
        if frame is not None:
            frames.append(np.ascontiguousarray(frame))

        if terminated or last_coverage >= self.coverage_threshold:
            break

    if actions_taken:
        a = np.asarray(actions_taken)
        first = [[round(float(x), 1) for x in p] for p in actions_taken[:8]]
        print(
            f"[CHUNK32_DBG] ep={ep_idx} steps={len(actions_taken)} "
            f"chunks={chunk_starts[:12]} "
            f"act_x[{a[:, 0].min():.1f},{a[:, 0].max():.1f}] "
            f"act_y[{a[:, 1].min():.1f},{a[:, 1].max():.1f}] "
            f"first8={first} final_cov={last_coverage:.3f}",
            flush=True,
        )

    rollout_video_dir = os.environ.get("ROLLOUT_VIDEO_DIR")
    if rollout_video_dir and frames:
        try:
            import torchvision.io as tvio

            os.makedirs(rollout_video_dir, exist_ok=True)
            path = os.path.join(
                rollout_video_dir, f"chunk32_roll_ep{ep_idx}_cov{last_coverage:.3f}.mp4"
            )
            vid = torch.from_numpy(np.stack(frames, axis=0)).to(torch.uint8)
            tvio.write_video(path, vid, fps=int(self.video_fps), video_codec="h264")
            print(
                f"[CHUNK32_VIDEO] ep={ep_idx} wrote {path} "
                f"({len(frames)} frames) cov={last_coverage:.3f}",
                flush=True,
            )
        except Exception as exc:
            print(f"[CHUNK32_VIDEO] ep={ep_idx} FAILED: {exc}", flush=True)

    return last_coverage, frames, actions_taken


eval_sim.SimRolloutEval._rollout_one = _rollout_one_chunk32

import hydra
from omegaconf import DictConfig, OmegaConf

from egomimic.trainHydra import train
from egomimic.pl_utils.utils import extras


@hydra.main(
    version_base="1.3",
    config_path="hydra_configs",
    config_name="train_zarr_cartesian.yaml",
)
def main(cfg: DictConfig):
    extras(cfg)
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
