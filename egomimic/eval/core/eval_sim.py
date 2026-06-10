"""
Generic closed-loop sim evaluator.

Wraps the Tsimulation.pushshapes.PushShapesEnv simulator. The env is
treated as a dumb action-stepper: it takes one 2D action per step and
returns the next obs. **Model prediction lives in the algo.** The eval
class owns only the env loop, frame capture, and coverage metric.

Design:
  * SimRolloutEval (base) drives the env loop. For each step, it
    calls algo.inference_step(obs_zarr, t, emb_id) -> np.ndarray,
    feeds the action to env.step, and renders the frame.
  * Per-model subclasses (HPTSimEval, PackedSimEval) implement
    batch_to_env_init(batch, b_idx, emb_id): slice the val batch
    into the env init dict for replay mode. The base class is otherwise
    model-agnostic.
  * algo.inference_step owns ALL model state — KV cache, AR
    position, action-chunk buffer. The eval class never touches these.
    t=0 is the universal reset signal that tells the algo to wipe
    any prior rollout state and start fresh.
  * Return contract: (action_dim,) = one action for this env step
    (the legacy path; algos with internal queues dispense one per
    call). A flat (K*2,) return with K>1 is a time-ordered K-chunk
    executed open-loop: the env steps K times before the model is
    queried (and observes) again — upstream keyframe semantics.
  * Obs translation: _env_to_zarr_dict(obs_env) converts the env's
    native obs into a canonical zarr-key dict (same shape as what the
    dataset emits). The algo then applies its own keymap + transforms.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from egomimic.eval.core.eval_video import EvalVideo


# Env-output -> zarr-format conversion + state-split. PushShapes-specific
# glue; the canonical home is the embodiment package. Re-exported here so the
# legacy eval_sim._ENV_TO_ZARR / _state_to_init / _env_to_zarr_pushshapes
# names (used internally + by eval_dfot_self_rollout and scripts/verify_*) keep
# resolving unchanged.
from egomimic.rldb.embodiment.pushshapes_sim import (  # noqa: E402,F401
    _ENV_TO_ZARR,
    _env_to_zarr_pushshapes,
    _state_to_init,
)

# --------------------------------------------------------------------- #
# Base sim eval — owns env loop only.
# --------------------------------------------------------------------- #


class SimRolloutEval(EvalVideo):
    """Generic env-loop driver. Model-agnostic.

    Subclasses implement batch_to_env_init (replay-mode init) and
    _infer_n_episodes (how many rollouts to run per val pass).

    The algo's inference_step(obs_zarr, t, emb_id) -> np.ndarray is
    the single entry point for model prediction. t=0 resets algo
    state.
    """

    def __init__(
        self,
        env_kwargs: dict | None = None,
        embodiment_name: str = "pushshapes_sim",
        init_mode: str = "replay",
        init_seeds: list[int] | None = None,
        max_steps: int = 200,
        coverage_threshold: float = 0.7,
        coverage_agg: str = "final",
        n_eval_episodes: int | None = None,
        video_fps: int = 30,
        max_videos: int | None = None,
        limit_val_batches: int = 4,
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
        if self.embodiment_name not in _ENV_TO_ZARR:
            raise ValueError(
                f"No env-to-zarr converter for {self.embodiment_name!r}. "
                f"Add to _ENV_TO_ZARR in egomimic/rldb/embodiment/pushshapes_sim.py."
            )
        self.init_mode = str(init_mode)
        if self.init_mode not in {"replay", "random", "seeds"}:
            raise ValueError(
                f"init_mode must be replay/random/seeds, got {self.init_mode!r}"
            )
        self.init_seeds = list(init_seeds or [])
        if int(max_steps) <= 0:
            raise ValueError(
                f"max_steps must be > 0 (got {max_steps}). No fallback to dataloader seq_len."
            )
        self.max_steps = int(max_steps)
        self.coverage_threshold = float(coverage_threshold)
        # "final" = coverage at the last env step (legacy default);
        # "best" = best-so-far coverage over the rollout (upstream-pureDF
        # convention — score the peak, not where the policy happens to end).
        self.coverage_agg = str(coverage_agg)
        if self.coverage_agg not in {"final", "best"}:
            raise ValueError(
                f"coverage_agg must be final/best, got {self.coverage_agg!r}"
            )
        # None = legacy behavior (roll out only the rendered episodes,
        # i.e. min(batch episodes, max_videos)). When set, exactly this many
        # episodes are rolled out for the coverage metric (capped by the
        # batch's episode count in replay mode); rendering stays capped by
        # max_videos.
        self.n_eval_episodes = int(n_eval_episodes) if n_eval_episodes is not None else None
        self.video_fps = int(video_fps)
        self.limit_val_batches = int(limit_val_batches)
        self._env = None
        self._init_counter = 0

    def video_dir(self) -> str:
        return os.path.join(self.root_dir(), "videos", "sim")

    def _get_env(self):
        if self._env is None:
            kwargs = dict(self.env_kwargs)
            kwargs.setdefault("render_mode", "rgb_array")
            from Tsimulation.pushshapes import PushShapesEnv

            self._env = PushShapesEnv(**kwargs)
        return self._env

    def _init_env(self, env, env_init_dict: dict | None, ep_seed_offset: int) -> None:
        """Reset env. For replay mode, env_init_dict carries
        agent_pos + object_pose (+ optional goal_pose).
        For random/seeds modes, env_init_dict is None.
        """
        if self.init_mode == "random":
            env.reset(seed=ep_seed_offset)
        elif self.init_mode == "seeds":
            if not self.init_seeds:
                raise ValueError("init_mode='seeds' requires init_seeds")
            seed = self.init_seeds[self._init_counter % len(self.init_seeds)]
            self._init_counter += 1
            env.reset(seed=int(seed))
        else:
            if env_init_dict is None:
                raise ValueError(
                    "init_mode='replay' requires env_init_dict from batch_to_env_init"
                )
            env.reset(seed=ep_seed_offset)
            env.set_state(
                agent_pos=env_init_dict["agent_pos"],
                object_pose=env_init_dict["object_pose"],
                goal_pose=env_init_dict.get("goal_pose"),
            )

    def _env_to_zarr_dict(self, obs_env: dict, device: torch.device) -> dict:
        return _ENV_TO_ZARR[self.embodiment_name](obs_env, device)

    def batch_to_env_init(self, batch: Dict[str, Any], b_idx: int, emb_id: int) -> dict | None:
        raise NotImplementedError("subclass me")

    def _infer_n_episodes(self, batch: Dict[str, Any]) -> int:
        raise NotImplementedError("subclass me")

    def _draw_action_overlay(
        self,
        frame: np.ndarray,
        action_xy: np.ndarray,
        history: List[np.ndarray],
    ) -> np.ndarray:
        """Overlay the predicted action on the env's rendered frame.

        For pushshapes_sim: draws a fading trail of the last 10 predicted
        action positions (older = darker blue) plus the current action as a
        bright red dot with white outline. Action coordinates are in world
        space [0, 512]; we scale to pixel coords using frame.shape.

        For other embodiments: no-op (override to draw).
        """
        if self.embodiment_name != "pushshapes_sim":
            return frame
        try:
            import cv2
        except ImportError:
            return frame
        h, w = frame.shape[:2]
        scale = h / 512.0  # WORLD_SIZE
        out = frame.copy()
        # Trail: cyan polyline through last ~10 predicted positions (incl.
        # the current). Cyan is high-contrast vs the env's red pusher, blue
        # object, and green goal — no color collision.
        trail = history[-10:]
        if len(trail) >= 2:
            pts = []
            for a in trail:
                px, py = int(a[0] * scale), int(a[1] * scale)
                pts.append((px, py))
            for i in range(len(pts) - 1):
                cv2.line(out, pts[i], pts[i + 1], color=(255, 255, 0), thickness=2)  # cyan-ish (BGR/RGB readable)
        # Current action: bright YELLOW dot with black outline (high
        # contrast against the env's red pusher and any other env colors).
        cx, cy = int(action_xy[0] * scale), int(action_xy[1] * scale)
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(out, (cx, cy), radius=5, color=(0, 0, 0), thickness=-1)        # black halo
            cv2.circle(out, (cx, cy), radius=4, color=(0, 255, 255), thickness=-1)    # yellow fill
        return out

    @torch.no_grad()
    def _rollout_one(
        self,
        env_init_dict: dict | None,
        emb_id: int,
        ep_idx: int,
        render: bool = True,
    ) -> Tuple[float, List[np.ndarray]]:
        env = self._get_env()
        self._init_env(env, env_init_dict, ep_seed_offset=ep_idx)
        device = self.trainer.lightning_module.device
        algo = self.model
        if not hasattr(algo, "inference_step"):
            raise RuntimeError(
                f"Algo {type(algo).__name__} does not implement inference_step — "
                "required for SimRolloutEval. Must return np.float32 (action_dim,) "
                "in absolute frame from (obs_zarr, t, emb_id), or a flat "
                "(K*2,) time-ordered chunk to commit K steps open-loop."
            )

        frames: List[np.ndarray] = []
        action_history: List[np.ndarray] = []
        coverage = 0.0
        best_coverage = 0.0
        pending: List[np.ndarray] = []  # open-loop tail of a committed K-chunk
        for t in range(self.max_steps):
            if not pending:
                obs_env = env._get_obs()
                obs_zarr = self._env_to_zarr_dict(obs_env, device)
                ret = algo.inference_step(obs_zarr, t, emb_id, T_max=self.max_steps)
                ret = np.asarray(ret, dtype=np.float32).reshape(-1)
                if ret.size > 2 and ret.size % 2 == 0:
                    # K-chunk contract (D07): a flat (K*2,) return is K xy
                    # actions in time order, executed open-loop — the model is
                    # re-queried (and re-observes) only at the next keyframe,
                    # mirroring upstream df_pusht.py's commit-K-then-re-ground
                    # loop. Algos that dispense one action per call via an
                    # internal queue (DFoT._pp_queue, HNet._sim_action_queue,
                    # HPT chunk buffer) return (2,) and take the branch below.
                    pending = [row.copy() for row in ret.reshape(-1, 2)]
                else:
                    # legacy single-action contract: one (action_dim,) action
                    # per env step (odd sizes keep the historical [:2] cut).
                    pending = [ret[:2]]
            action_xy = pending.pop(0)
            action_history.append(action_xy.copy())
            _, _, terminated, _, info = env.step(action_xy)
            coverage = float(info.get("coverage", 0.0))
            best_coverage = max(best_coverage, coverage)
            frame = env.render() if render else None
            if frame is not None:
                frame = self._draw_action_overlay(frame, action_xy, action_history)
                frames.append(np.ascontiguousarray(frame))
            if terminated:
                break
        if self.coverage_agg == "best":
            return best_coverage, frames
        return coverage, frames

    def compute_metrics_and_viz(
        self, batch: Dict[int, Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, np.ndarray]]:
        device = self.trainer.lightning_module.device
        metrics: Dict[str, torch.Tensor] = {}
        images_dict: Dict[int, np.ndarray] = {}

        for emb_id, _batch in batch.items():
            if self.n_eval_episodes is not None:
                # Explicit episode budget: roll out exactly this many (capped
                # by batch episodes in replay mode — replay needs an init per
                # episode); rendering stays capped by max_videos below.
                B = self.n_eval_episodes
                if self.init_mode == "replay":
                    B = min(B, self._infer_n_episodes(_batch))
                B_render = min(B, self.max_videos) if self.max_videos is not None else B
                n_rollouts = B
            else:
                # Legacy default: only the rendered episodes are rolled out.
                B = self._infer_n_episodes(_batch)
                B_render = min(B, self.max_videos) if self.max_videos is not None else B
                n_rollouts = B_render
            if B == 0:
                continue

            ep_coverages: List[float] = []
            ep_successes: List[float] = []
            ep_frames: List[np.ndarray] = []

            for b in range(n_rollouts):
                env_init = (
                    self.batch_to_env_init(_batch, b, emb_id)
                    if self.init_mode == "replay"
                    else None
                )
                cov, frames = self._rollout_one(
                    env_init, emb_id, b, render=(b < B_render)
                )
                ep_coverages.append(cov)
                ep_successes.append(float(cov >= self.coverage_threshold))
                if frames and b < B_render:
                    ep_frames.extend(frames)
                    if b < B_render - 1:
                        H, W, _ = frames[0].shape
                        sep = np.zeros((5, H, W, 3), dtype=np.uint8)
                        ep_frames.extend(list(sep))

            mean_cov = float(np.mean(ep_coverages)) if ep_coverages else 0.0
            success_rate = float(np.mean(ep_successes)) if ep_successes else 0.0
            metrics[f"Valid/emb{emb_id}_sim_coverage"] = torch.tensor(
                mean_cov, device=device
            )
            metrics[f"Valid/emb{emb_id}_sim_success_rate"] = torch.tensor(
                success_rate, device=device
            )
            if ep_frames:
                images_dict[emb_id] = np.stack(ep_frames, axis=0)

        return metrics, images_dict


# --------------------------------------------------------------------- #
# Per-model subclasses.
# --------------------------------------------------------------------- #


class HPTSimEval(SimRolloutEval):
    """HPT per-frame batch layout: each row of state_agent_obj is one
    episode init point."""

    def _infer_n_episodes(self, batch: Dict[str, Any]) -> int:
        state = batch.get("state_agent_obj")
        if state is None or state.dim() < 2:
            return 0
        return min(int(state.shape[0]), self.limit_val_batches or 4)

    def batch_to_env_init(self, batch: Dict[str, Any], b_idx: int, emb_id: int) -> dict:
        state_b = batch["state_agent_obj"][b_idx]
        state_unnorm = self.model.norm_stats.unnormalize(
            {"state_agent_obj": state_b}, emb_id
        )["state_agent_obj"]
        state_np = state_unnorm.detach().cpu().numpy()
        agent_pos, object_pose = _state_to_init(state_np)
        goal_pose = None
        if "goal_pose" in batch:
            goal_seq = batch["goal_pose"][b_idx]
            goal_pose = tuple(
                float(x)
                for x in np.asarray(goal_seq.detach().cpu().numpy()).reshape(-1)[:3]
            )
        return {
            "agent_pos": agent_pos,
            "object_pose": object_pose,
            "goal_pose": goal_pose,
        }


class PackedSimEval(SimRolloutEval):
    """Closed-loop sim eval for any algo using packed batches.

    The batch is expected to carry ``cu_seqlens`` (and ideally ``seq_lens``)
    marking episode boundaries; each episode's frame-0 state is the env init.
    The algo just needs to implement ``inference_step(obs_zarr, t, emb_id)``
    — sim eval is otherwise algo-agnostic. Used by HNet, DFoT, and any
    future packed-batch policy.

    Kept under the (legacy) name ``HNetSimEval`` for backward compat — both
    names resolve to this class.
    """

    def _infer_n_episodes(self, batch: Dict[str, Any]) -> int:
        seq_lens = batch.get("seq_lens")
        if seq_lens is None or seq_lens.dim() < 1:
            return 0
        return min(int(seq_lens.shape[0]), self.limit_val_batches or 4)

    def batch_to_env_init(self, batch: Dict[str, Any], b_idx: int, emb_id: int) -> dict:
        cu = batch["cu_seqlens"]
        s = int(cu[b_idx].item())
        state_first = batch["state_agent_obj"][s]
        state_unnorm = self.model.norm_stats.unnormalize(
            {"state_agent_obj": state_first}, emb_id
        )["state_agent_obj"]
        state_np = state_unnorm.detach().cpu().numpy()
        agent_pos, object_pose = _state_to_init(state_np)
        goal_pose = None
        if "goal_pose" in batch:
            goal_seq = batch["goal_pose"][s]
            goal_pose = tuple(
                float(x)
                for x in np.asarray(goal_seq.detach().cpu().numpy()).reshape(-1)[:3]
            )
        return {
            "agent_pos": agent_pos,
            "object_pose": object_pose,
            "goal_pose": goal_pose,
        }


# Backward-compat alias — old configs / docs may reference HNetSimEval;
# the class is algo-agnostic so the rename to PackedSimEval is purely
# cosmetic. Existing eval_hnet_sim.yaml and downstream callers keep working.
HNetSimEval = PackedSimEval
