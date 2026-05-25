"""
Generic closed-loop sim evaluator.

Wraps ``Tsimulation.pushshapes.PushShapesEnv``. For each validation
episode, resets the env, then steps one frame at a time by delegating
inference to the algo's ``sim_init_state`` + ``sim_predict_step``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


def _format_pushshapes_obs(obs_env: dict, device: torch.device) -> dict:
    """Format PushShapesEnv obs -> model batch (B=1)."""
    state_5 = np.concatenate(
        [obs_env["agent_pos"], obs_env["object_pose"]], axis=0
    ).astype(np.float32)
    image_chw = np.transpose(obs_env["image"], (2, 0, 1)).astype(np.float32) / 255.0
    return {
        "state_agent_obj": torch.from_numpy(state_5).unsqueeze(0).to(device),
        "front_img_1": torch.from_numpy(image_chw).unsqueeze(0).to(device),
    }


_OBS_FORMATTERS = {
    "pushshapes_sim": _format_pushshapes_obs,
    "pushshapes_sim_stick": _format_pushshapes_obs,
}

_REPLAY_RESET_SEED = 0


def _state_to_init(state: np.ndarray) -> tuple:
    """Split (5,) state into (agent_pos, object_pose) for set_state."""
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    if state.shape[0] < 5:
        raise ValueError(f"expected state of len >= 5, got {state.shape}")
    return (float(state[0]), float(state[1])), (float(state[2]), float(state[3]), float(state[4]))


class SimRolloutEval(EvalVideo):
    """Closed-loop sim rollout eval. Algo-agnostic."""

    def __init__(
        self,
        env_kwargs: dict | None = None,
        embodiment_name: str = "pushshapes_sim",
        init_mode: str = "replay",
        init_seeds: list[int] | None = None,
        max_steps: int | None = None,
        coverage_threshold: float = 0.7,
        video_fps: int = 30,
        limit_val_batches: int = 4,
        max_videos: int | None = None,
        viz_func: dict | None = None,
        transform_lists: dict | None = None,
    ):
        super().__init__(
            limit_val_batches=limit_val_batches,
            viz_func=viz_func,
            transform_lists=transform_lists,
            max_videos=max_videos,
        )
        self.env_kwargs = dict(env_kwargs or {})
        self.embodiment_name = str(embodiment_name)
        if self.embodiment_name not in _OBS_FORMATTERS:
            raise ValueError(
                f"No obs formatter for embodiment {self.embodiment_name!r}. "
                f"Add an entry to _OBS_FORMATTERS in eval_sim.py."
            )
        self.init_mode = str(init_mode)
        if self.init_mode not in {"replay", "random", "seeds"}:
            raise ValueError(f"init_mode must be replay/random/seeds, got {self.init_mode!r}")
        self.init_seeds = list(init_seeds or [])
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.coverage_threshold = float(coverage_threshold)
        self.video_fps = int(video_fps)
        self._target_emb_id = get_embodiment_id(self.embodiment_name)
        self._env = None
        self._init_counter = 0

    def _get_env(self):
        if self._env is None:
            kwargs = dict(self.env_kwargs)
            kwargs.setdefault("render_mode", "rgb_array")
            from Tsimulation.pushshapes import PushShapesEnv
            self._env = PushShapesEnv(**kwargs)
        return self._env

    def _init_env(self, env, sample: dict, ep_seed_offset: int, emb_id: int) -> dict:
        """Reset + optionally set_state. Returns the initial observation dict."""
        if self.init_mode == "replay":
            state_seq = sample.get("state_agent_obj")
            if state_seq is None:
                raise KeyError("init_mode='replay' requires 'state_agent_obj' in batch")

            unnorm = self.model.norm_stats.unnormalize(
                {"state_agent_obj": state_seq}, emb_id
            )
            frame0 = unnorm["state_agent_obj"][0].detach().cpu().numpy()
            agent_pos, object_pose = _state_to_init(frame0)

            goal_pose = None
            goal_seq = sample.get("goal_pose")
            if goal_seq is not None:
                goal_pose = tuple(
                    float(x) for x in goal_seq[0].detach().cpu().numpy().reshape(-1)[:3]
                )

            env.reset(seed=_REPLAY_RESET_SEED)
            env.set_state(
                agent_pos=agent_pos,
                object_pose=object_pose,
                goal_pose=goal_pose,
            )
        elif self.init_mode == "random":
            env.reset(seed=ep_seed_offset)
        elif self.init_mode == "seeds":
            if not self.init_seeds:
                raise ValueError("init_mode='seeds' requires init_seeds")
            seed = self.init_seeds[self._init_counter % len(self.init_seeds)]
            self._init_counter += 1
            env.reset(seed=int(seed))

        return env._get_obs()

    @torch.no_grad()
    def _rollout_one(
        self,
        sample: Dict[str, torch.Tensor],
        seq_len: int,
        emb_id: int,
        ep_idx: int,
    ) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
        """One closed-loop sim rollout. Returns (final_coverage, frames, actions)."""
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
        obs_formatter = _OBS_FORMATTERS[self.embodiment_name]

        frames: List[np.ndarray] = []
        actions_taken: List[np.ndarray] = []
        last_coverage = 0.0

        for t in range(T):
            obs_raw = obs_formatter(obs_env, device)
            obs_norm = algo.norm_stats.normalize(obs_raw, emb_id)

            a_t_norm = algo.sim_predict_step(state, obs_norm, t, emb_id)
            a_t_world = (
                algo.norm_stats.unnormalize({ac_key: a_t_norm.squeeze(0)}, emb_id)[ac_key]
                .detach().cpu().numpy().reshape(-1)
            )
            action_xy = np.array(
                [float(a_t_world[0]), float(a_t_world[1])], dtype=np.float64
            )

            obs_env, _, terminated, _, info = env.step(action_xy)
            last_coverage = float(info.get("coverage", 0.0))
            actions_taken.append(action_xy)

            frame = env.render()
            if frame is not None:
                frames.append(np.ascontiguousarray(frame))

            if terminated:
                break

        return last_coverage, frames, actions_taken

    def compute_metrics_and_viz(
        self, batch: Dict[int, Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, np.ndarray]]:
        metrics: Dict[str, torch.Tensor] = {}
        images_dict: Dict[int, np.ndarray] = {}
        device = self.trainer.lightning_module.device

        for emb_id, _batch in batch.items():
            if emb_id != self._target_emb_id:
                continue

            is_packed = _batch.get("_packed", False)
            if not is_packed:
                continue

            cu = _batch["cu_seqlens"]
            seq_lens = _batch["seq_lens"]
            B = int(seq_lens.shape[0])
            B_render = min(B, self.max_videos) if self.max_videos is not None else B
            ep_coverages: list[float] = []
            ep_successes: list[float] = []
            ep_frames_for_video: list[np.ndarray] = []

            for b in range(B):
                s = int(cu[b].item())
                e = int(cu[b + 1].item())
                T_ep = e - s
                sample: dict = {}
                for k, v in _batch.items():
                    if not torch.is_tensor(v):
                        continue
                    if v.dim() >= 1 and v.shape[0] == int(cu[-1].item()):
                        sample[k] = v[s:e]
                coverage, frames, _ = self._rollout_one(
                    sample, seq_len=T_ep, emb_id=emb_id, ep_idx=b,
                )
                ep_coverages.append(coverage)
                ep_successes.append(float(coverage >= self.coverage_threshold))
                if b < B_render and frames:
                    ep_frames_for_video.extend(frames)
                    if b < B_render - 1:
                        H, W, _ = frames[0].shape
                        sep = np.zeros((5, H, W, 3), dtype=np.uint8)
                        ep_frames_for_video.extend(list(sep))

            mean_cov = float(np.mean(ep_coverages)) if ep_coverages else 0.0
            success_rate = float(np.mean(ep_successes)) if ep_successes else 0.0
            metrics[f"Valid/emb{emb_id}_sim_coverage"] = torch.tensor(mean_cov, device=device)
            metrics[f"Valid/emb{emb_id}_sim_success_rate"] = torch.tensor(success_rate, device=device)

            if ep_frames_for_video:
                images_dict[emb_id] = np.stack(ep_frames_for_video, axis=0)

        return metrics, images_dict
