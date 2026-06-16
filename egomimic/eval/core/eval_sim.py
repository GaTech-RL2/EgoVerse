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
  * Obs translation: _env_to_zarr_dict(obs_env) converts the env's
    native obs into a canonical zarr-key dict (same shape as what the
    dataset emits). The algo then applies its own keymap + transforms.
"""

from __future__ import annotations

import os
import signal as _signal
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


class _RolloutTimeout(Exception):
    """Raised by the per-rollout SIGALRM watchdog (sim-eval hang guard)."""


def _rollout_alarm_handler(signum, frame):
    raise _RolloutTimeout()


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
        rollout_timeout_s: int = 120,
        report_max_coverage: bool = False,
        coverage_threshold: float = 0.7,
        video_fps: int = 30,
        max_videos: int | None = None,
        limit_val_batches: int = 4,
        viz_func: dict | None = None,
        transform_lists: dict | None = None,
        rng_pairing: bool = False,
        # --- EgoVerse2 sim-eval opt-ins (accepted for config parity). Defaults
        # reproduce gmm's autoregressive rollout exactly. The EV2 temporal-
        # ensemble / chunk-openloop / delta-action / goal rollout BRANCHES were
        # not grafted onto gmm's (diverged) rollout — setting any non-default
        # emits a warning and falls back to gmm's AR rollout. See
        # INTEGRATION_NOTES.md (#13). DAgger paths are intentionally omitted. ---
        temporal_ensemble: bool = False,
        te_m: float = 0.01,
        delta_action: bool = False,
        rollout_mode: str = "ar",
        chunk_k: int = 32,
        goal_in_obs: bool = False,
        fixed_goal=None,
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
        self.rollout_timeout_s = int(rollout_timeout_s)  # 0 disables the watchdog
        self.report_max_coverage = bool(report_max_coverage)  # peak vs final IoU
        self.coverage_threshold = float(coverage_threshold)
        self.video_fps = int(video_fps)
        self.limit_val_batches = int(limit_val_batches)
        # Reseed the sampler RNG per episode (inside fork_rng, so any outer
        # RNG stream — e.g. a training run whose val invokes this eval — is
        # untouched). Makes stochastic-head sampling noise paired across runs
        # for the same episode index. Off by default: numbers from unpaired
        # evals stay reproducible.
        self.rng_pairing = bool(rng_pairing)
        # EgoVerse2 sim-eval opt-ins (stored for config parity).
        self.temporal_ensemble = bool(temporal_ensemble)
        self.te_m = float(te_m)
        self.delta_action = bool(delta_action)
        self.rollout_mode = str(rollout_mode)
        self.chunk_k = int(chunk_k)
        self.goal_in_obs = bool(goal_in_obs)
        self.fixed_goal = fixed_goal
        if (
            temporal_ensemble
            or delta_action
            or goal_in_obs
            or rollout_mode != "ar"
            or fixed_goal is not None
        ):
            import warnings

            warnings.warn(
                "SimRolloutEval: EgoVerse2 opt-ins (temporal_ensemble / "
                "chunk-openloop / delta_action / goal rollout) are accepted for "
                "config parity but NOT active in this merged build — gmm's "
                "autoregressive rollout is used. See INTEGRATION_NOTES.md (#13).",
                RuntimeWarning,
            )
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
    ) -> Tuple[float, List[np.ndarray]]:
        if not self.rng_pairing:
            return self._rollout_one_impl(env_init_dict, emb_id, ep_idx)
        device = self.trainer.lightning_module.device
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            seed = 20000 + 97 * ep_idx
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            return self._rollout_one_impl(env_init_dict, emb_id, ep_idx)

    def _rollout_one_impl(
        self,
        env_init_dict: dict | None,
        emb_id: int,
        ep_idx: int,
    ) -> Tuple[float, List[np.ndarray]]:
        env = self._get_env()
        self._init_env(env, env_init_dict, ep_seed_offset=ep_idx)
        device = self.trainer.lightning_module.device
        algo = self.model
        if not hasattr(algo, "inference_step"):
            raise RuntimeError(
                f"Algo {type(algo).__name__} does not implement inference_step — "
                "required for SimRolloutEval. Must return np.float32 (action_dim,) "
                "in absolute frame from (obs_zarr, t, emb_id)."
            )

        frames: List[np.ndarray] = []
        action_history: List[np.ndarray] = []
        coverage = 0.0
        max_coverage = 0.0  # peak IoU over the rollout (for report_max_coverage)
        # Per-rollout SIGALRM watchdog: a single env.step() can wedge inside the
        # pymunk C solver (e.g. a NaN kinematic velocity) and never return, so the
        # max_steps cap never triggers -> the whole val hangs at 0% CPU. The alarm
        # aborts a stuck rollout (logs 0-coverage) and lets the val finish. Runs on
        # the main thread (Lightning calls this inline), so SIGALRM is valid here.
        prev_handler = None
        if self.rollout_timeout_s > 0:
            prev_handler = _signal.signal(_signal.SIGALRM, _rollout_alarm_handler)
            _signal.alarm(self.rollout_timeout_s)
        try:
            for t in range(self.max_steps):
                obs_env = env._get_obs()
                obs_zarr = self._env_to_zarr_dict(obs_env, device)
                action_xy = algo.inference_step(obs_zarr, t, emb_id, T_max=self.max_steps)
                action_xy = np.asarray(action_xy, dtype=np.float32).reshape(-1)[:2]
                if not np.all(np.isfinite(action_xy)):
                    # Non-finite action (late-training instability) poisons the
                    # pymunk solver — np.clip does NOT sanitize NaN, so a NaN
                    # velocity wedges _space.step. Abort the rollout as 0-coverage.
                    print(
                        f"[sim] WARNING: non-finite action t={t} emb{emb_id} "
                        f"ep{ep_idx} (raw={action_xy.tolist()}); abort 0-cov."
                    )
                    coverage = 0.0
                    break
                action_history.append(action_xy.copy())
                _, _, terminated, _, info = env.step(action_xy)
                coverage = float(info.get("coverage", 0.0))
                max_coverage = max(max_coverage, coverage)
                frame = env.render()
                if frame is not None:
                    frame = self._draw_action_overlay(frame, action_xy, action_history)
                    frames.append(np.ascontiguousarray(frame))
                if terminated:
                    break
        except _RolloutTimeout:
            print(
                f"[sim] WATCHDOG: rollout emb{emb_id} ep{ep_idx} exceeded "
                f"{self.rollout_timeout_s}s; logging 0-coverage and continuing."
            )
            coverage = 0.0
        finally:
            if self.rollout_timeout_s > 0:
                _signal.alarm(0)
                if prev_handler is not None:
                    _signal.signal(_signal.SIGALRM, prev_handler)
        # report_max_coverage: peak IoU over the rollout ("did it ever align?"),
        # vs the default final-step IoU which penalizes post-alignment overshoot.
        return (max_coverage if self.report_max_coverage else coverage), frames

    def compute_metrics_and_viz(
        self, batch: Dict[int, Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, np.ndarray]]:
        device = self.trainer.lightning_module.device
        metrics: Dict[str, torch.Tensor] = {}
        images_dict: Dict[int, np.ndarray] = {}
        self._last_per_ep_frames = {}
        self._last_per_ep_coverages = {}

        for emb_id, _batch in batch.items():
            B = self._infer_n_episodes(_batch)
            if B == 0:
                continue
            # Seed-mode inits are indexed per (embodiment, episode): episode i
            # of EVERY embodiment uses init_seeds[i], so inits stay paired
            # across embodiments and across runs regardless of batch makeup.
            self._init_counter = 0
            B_render = min(B, self.max_videos) if self.max_videos is not None else B

            ep_coverages: List[float] = []
            ep_successes: List[float] = []
            ep_frames: List[np.ndarray] = []
            per_ep_frames: List[np.ndarray] = []

            for b in range(B_render):
                env_init = (
                    self.batch_to_env_init(_batch, b, emb_id)
                    if self.init_mode == "replay"
                    else None
                )
                cov, frames = self._rollout_one(env_init, emb_id, b)
                ep_coverages.append(cov)
                ep_successes.append(float(cov >= self.coverage_threshold))
                if frames:
                    per_ep_frames.append(np.stack(frames, axis=0))
                    ep_frames.extend(frames)
                    if b < B_render - 1:
                        H, W, _ = frames[0].shape
                        sep = np.zeros((5, H, W, 3), dtype=np.uint8)
                        ep_frames.extend(list(sep))

            # Per-episode values to stdout so offline eval logs carry the full
            # DISTRIBUTION (means hide bimodality; success-rate alone hides
            # the near-miss mass). Parse with: grep '\[sim\] .* ep_coverages'.
            if ep_coverages:
                print(
                    f"[sim] emb{emb_id} ep_coverages: "
                    + ",".join(f"{c:.4f}" for c in ep_coverages)
                )
            self._last_per_ep_frames[emb_id] = per_ep_frames
            self._last_per_ep_coverages[emb_id] = list(ep_coverages)
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
