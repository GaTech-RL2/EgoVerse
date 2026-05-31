"""
Generic closed-loop sim evaluator.

Wraps ``Tsimulation.pushshapes.PushShapesEnv``. For each validation
episode, resets the env, then steps one frame at a time by delegating
inference to the algo's ``sim_init_state`` + ``sim_predict_step``.
"""

from __future__ import annotations
import os

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


from collections import deque as _deque_te


def _temporal_ensemble(chunk_buf, t, horizon, m):
    """ACT-style temporal ensemble. chunk_buf: deque of (pred_time, chunk (H,A)).
    Returns the (1,A) weighted-average of all overlapping chunks' predictions for
    step t, weights exp(-m*age) oldest->newest. Drops chunks older than horizon.
    Shared by the HPT chunk rollout and the H-Net chunk_te rollout."""
    while chunk_buf and chunk_buf[0][0] <= t - horizon:
        chunk_buf.popleft()
    preds = [ch[t - pt] for (pt, ch) in chunk_buf if 0 <= t - pt < ch.shape[0]]
    preds = torch.stack(preds, dim=0)
    w = torch.exp(-m * torch.arange(preds.shape[0], device=preds.device, dtype=preds.dtype))
    w = w / w.sum()
    return (preds * w[:, None]).sum(dim=0, keepdim=True)


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
        temporal_ensemble: bool = True,
        te_m: float = 0.01,
        delta_action: bool = False,
        rollout_mode: str = "ar",
        chunk_k: int = 32,
        goal_in_obs: bool = False,
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
        # Rollout-mode knobs (config-driven; no env-vars):
        #   temporal_ensemble / te_m -> HPT chunk rollout (base _rollout_one)
        #   delta_action             -> H-Net delta integration (HNetSimEval)
        self.temporal_ensemble = bool(temporal_ensemble)
        self.te_m = float(te_m)
        self.delta_action = bool(delta_action)
        self.rollout_mode = str(rollout_mode)   # 'ar' (single-step) | 'chunk_te'
        self.chunk_k = int(chunk_k)
        self.goal_in_obs = bool(goal_in_obs)    # feed goal_obs to a goal-conditioned model

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

            # The packed eval dataset returns RAW world-coord state (it does not
            # normalize like the per-frame train loader). Use it directly for
            # env init; unnormalizing raw values double-scales them into garbage
            # (agent ~46k instead of ~210), placing the object off-screen so
            # coverage is always 0.
            frame0 = state_seq[0].detach().cpu().numpy()
            agent_pos, object_pose = _state_to_init(frame0)

            goal_pose = None
            goal_seq = sample.get("goal_pose")
            if goal_seq is not None:
                goal_pose = tuple(
                    float(x) for x in goal_seq[0].detach().cpu().numpy().reshape(-1)[:3]
                )

            print(
                f"[ROLLOUT_DBG] sample_keys={list(sample.keys())} "
                f"goal_pose={goal_pose} agent={agent_pos} obj={object_pose}",
                flush=True,
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

        # The model is trained on obs WINDOWS of length action_horizon (32):
        # front_img_1 (B,32,3,96,96), state (B,32,5). A single-frame rollout obs
        # gives temporal dim 1, which mismatches training. Maintain a rolling
        # buffer of the last `horizon` env observations and feed the window.
        from collections import deque as _deque
        _H = 32
        _m = self.te_m                  # temporal-ensemble decay (config-driven)
        _te = self.temporal_ensemble    # ensemble on/off (config-driven)
        chunk_buf = _deque()                                   # (pred_time, chunk (H,A) norm)

        for t in range(T):
            obs_raw = obs_formatter(obs_env, device)
            obs_norm = algo.norm_stats.normalize(obs_raw, emb_id)
            # Fresh chunk every step (sim_predict_step stores it in
            # state["action_chunk"]); then TEMPORAL-ENSEMBLE (ACT-style):
            # weighted-average the predictions for the current step from all
            # recent overlapping chunks. Smooths the trajectory and blends in
            # earlier confident forward-motion predictions, so the policy
            # escapes the "stay put" fixed point single-action execution falls
            # into.
            state["action_chunk"] = None
            state["chunk_idx"] = 0
            a_t_single = algo.sim_predict_step(state, obs_norm, t, emb_id)  # (1,1,A)
            if _te:
                chunk_buf.append((t, state["action_chunk"][0].detach()))   # (H,A) norm
                a_t_norm = _temporal_ensemble(chunk_buf, t, _H, _m)
            else:
                a_t_norm = a_t_single.squeeze(0)
            a_t_world = (
                algo.norm_stats.unnormalize({ac_key: a_t_norm}, emb_id)[ac_key]
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

        if actions_taken:
            _a = np.asarray(actions_taken)
            _first = [[round(float(x), 1) for x in p] for p in actions_taken[:6]]
            print(
                f"[ROLLOUT_DBG] ep={ep_idx} steps={len(actions_taken)} "
                f"act_x[{_a[:, 0].min():.1f},{_a[:, 0].max():.1f}] "
                f"act_y[{_a[:, 1].min():.1f},{_a[:, 1].max():.1f}] "
                f"first6={_first} final_cov={last_coverage:.3f}",
                flush=True,
            )
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
            # Fixed-seed eval: roll out exactly len(init_seeds) reproducible
            # envs (identical scenes for every model). Reset the counter so each
            # eval deterministically uses seeds[0..N-1].
            if self.init_mode == "seeds":
                n_rollouts = len(self.init_seeds)
                self._init_counter = 0
            else:
                n_rollouts = B
            B_render = min(n_rollouts, self.max_videos) if self.max_videos is not None else n_rollouts
            ep_coverages: list[float] = []
            ep_successes: list[float] = []
            ep_frames_for_video: list[np.ndarray] = []

            for b in range(n_rollouts):
                if self.init_mode == "seeds":
                    sample = {}
                    T_ep = self.max_steps if self.max_steps is not None else 300
                else:
                    s = int(cu[b].item())
                    e = int(cu[b + 1].item())
                    T_ep = e - s
                    sample = {}
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


class HNetSimEval(SimRolloutEval):
    """Closed-loop sim rollout for the autoregressive H-Net.

    HPT predicts an ``action_horizon`` chunk per step and is temporally
    ensembled in the base ``_rollout_one``. H-Net is natively single-step
    autoregressive: it feeds its own predicted action back through a KV
    cache, so chunk/TE semantics do not apply. We override ``_rollout_one``
    to drive the algo's ``inference_step`` (t=0 allocates a fresh AR/KV
    state, t>0 steps the cached state) one env frame at a time.

    Everything else — replay init, goal set, packed-batch parsing, coverage
    metric, video assembly — is inherited from ``SimRolloutEval`` unchanged,
    so HPT and H-Net are compared on an identical env / init / metric and
    differ only in the rollout inference each architecture actually uses.
    """

    @torch.no_grad()
    def _rollout_one(self, sample, seq_len, emb_id, ep_idx):
        if getattr(self, "rollout_mode", "ar") == "chunk_te":
            return self._rollout_chunk_te(sample, seq_len, emb_id, ep_idx)
        algo = self.model
        env = self._get_env()
        # mode=train validation hands us a NORMALIZED batch (the algo's
        # process_batch_for_training ran first). The env init needs raw
        # world-coord state, so un-normalize state_agent_obj here. goal_pose
        # is carried through un-normalized by process_batch, use it as-is.
        sample = dict(sample)
        if "state_agent_obj" in sample:
            sample["state_agent_obj"] = algo.norm_stats.unnormalize(
                {"state_agent_obj": sample["state_agent_obj"]}, emb_id
            )["state_agent_obj"]
        obs_env = self._init_env(env, sample, ep_seed_offset=ep_idx, emb_id=emb_id)
        T_eff = self.max_steps if self.max_steps is not None else int(seq_len)
        # Both H-Net variants cap at action_horizon: fused pos_emb has
        # length 2*action_horizon and inference_step sizes the KV cache to
        # action_horizon. Rolling out past it overflows pos_emb / the cache.
        _policy = algo.nets["policy"]
        _ah = int(getattr(_policy, "action_horizon", 0) or 0)
        if _ah > 0:
            T_eff = min(T_eff, _ah)

        if not hasattr(algo, "inference_step"):
            raise RuntimeError(
                f"Algo {type(algo).__name__} has no inference_step; HNetSimEval "
                "needs the autoregressive single-step rollout hook."
            )
        device = self.trainer.lightning_module.device
        obs_formatter = _OBS_FORMATTERS[self.embodiment_name]

        frames: List[np.ndarray] = []
        actions_taken: List[np.ndarray] = []
        last_coverage = 0.0

        # Delta-action rollout: the model was trained on per-step deltas, so
        # integrate predictions onto a running absolute cursor seeded from the
        # env's initial agent position. delta_action=true (config) enables it.
        _delta = self.delta_action
        abs_xy = None
        if _delta:
            # Seed from the recorded initial cursor action[0] (carried raw as
            # init_action), NOT agent_pos -- the pusher and cursor can start
            # 200-400px apart, which would shift the whole integrated path off
            # screen. Fall back to agent_pos only if init_action is absent.
            _ia = sample.get("init_action")
            if _ia is not None:
                _ia0 = _ia[0]
                if hasattr(_ia0, "detach"):
                    _ia0 = _ia0.detach().cpu().numpy()
                abs_xy = np.asarray(_ia0, dtype=np.float64).reshape(-1)[:2]
            else:
                abs_xy = np.asarray(
                    obs_env["agent_pos"], dtype=np.float64
                ).reshape(-1)[:2]

        for t in range(T_eff):
            obs_raw = obs_formatter(obs_env, device)
            if self.goal_in_obs and "goal_pose" in obs_env:
                obs_raw["goal_obs"] = torch.as_tensor(
                    np.asarray(obs_env["goal_pose"], dtype=np.float32).reshape(-1)[:3],
                    device=device,
                ).reshape(1, 3)
            # inference_step: t=0 allocates a fresh AR/KV state; t>0 steps the
            # cached state. Returns the model's frame action as np.ndarray
            # (absolute, or a delta when delta_action=true).
            a_world = np.asarray(algo.inference_step(obs_raw, t, emb_id)).reshape(-1)
            if _delta:
                abs_xy = abs_xy + a_world[:2]
                action_xy = np.array(
                    [float(abs_xy[0]), float(abs_xy[1])], dtype=np.float64
                )
            else:
                action_xy = np.array(
                    [float(a_world[0]), float(a_world[1])], dtype=np.float64
                )

            obs_env, _, terminated, _, info = env.step(action_xy)
            last_coverage = float(info.get("coverage", 0.0))
            actions_taken.append(action_xy)

            frame = env.render()
            if frame is not None:
                frames.append(np.ascontiguousarray(frame))
            if terminated:
                break

        if actions_taken:
            _a = np.asarray(actions_taken)
            _first = [[round(float(x), 1) for x in p] for p in actions_taken[:6]]
            print(
                f"[HNET_ROLLOUT_DBG] ep={ep_idx} steps={len(actions_taken)} "
                f"act_x[{_a[:, 0].min():.1f},{_a[:, 0].max():.1f}] "
                f"act_y[{_a[:, 1].min():.1f},{_a[:, 1].max():.1f}] "
                f"first6={_first} final_cov={last_coverage:.3f}",
                flush=True,
            )
        return last_coverage, frames, actions_taken

    @torch.no_grad()
    def _rollout_chunk_te(self, sample, seq_len, emb_id, ep_idx):
        """H-Net chunk + temporal-ensemble rollout (HPT-style, H4 test). At each
        env step, generate a K-step plan from the current obs and temporally
        ensemble overlapping plans. No retrain; eval-only on the AR checkpoint."""
        algo = self.model
        policy = algo.nets["policy"]
        env = self._get_env()
        sample = dict(sample)
        if "state_agent_obj" in sample:
            sample["state_agent_obj"] = algo.norm_stats.unnormalize(
                {"state_agent_obj": sample["state_agent_obj"]}, emb_id
            )["state_agent_obj"]
        obs_env = self._init_env(env, sample, ep_seed_offset=ep_idx, emb_id=emb_id)
        T_eff = self.max_steps if self.max_steps is not None else int(seq_len)
        _ah = int(getattr(policy, "action_horizon", 0) or 0)
        if _ah > 0:
            T_eff = min(T_eff, _ah)
        device = self.trainer.lightning_module.device
        ac_key = getattr(algo, "resolved_ac_keys", algo.ac_keys)[emb_id]
        obs_formatter = _OBS_FORMATTERS[self.embodiment_name]
        K = max(1, int(getattr(self, "chunk_k", 32)))
        chunk_buf = _deque_te()
        frames: List[np.ndarray] = []
        actions_taken: List[np.ndarray] = []
        last_coverage = 0.0
        for t in range(T_eff):
            obs_raw = obs_formatter(obs_env, device)
            if self.goal_in_obs and "goal_pose" in obs_env:
                obs_raw["goal_obs"] = torch.as_tensor(
                    np.asarray(obs_env["goal_pose"], dtype=np.float32).reshape(-1)[:3],
                    device=device,
                ).reshape(1, 3)
            obs_norm = algo.norm_stats.normalize(obs_raw, emb_id)
            chunk = policy.generate(obs_norm, batch_size=1, device=device, T=min(K, T_eff - t))
            chunk_buf.append((t, chunk[0].detach()))            # (K, A) normalized
            a_norm = _temporal_ensemble(chunk_buf, t, K, self.te_m)
            a_world = (
                algo.norm_stats.unnormalize({ac_key: a_norm}, emb_id)[ac_key]
                .detach().cpu().numpy().reshape(-1)
            )
            action_xy = np.array([float(a_world[0]), float(a_world[1])], dtype=np.float64)
            obs_env, _, terminated, _, info = env.step(action_xy)
            last_coverage = float(info.get("coverage", 0.0))
            actions_taken.append(action_xy)
            frame = env.render()
            if frame is not None:
                frames.append(np.ascontiguousarray(frame))
            if terminated:
                break
        if actions_taken:
            _a = np.asarray(actions_taken)
            print(
                f"[HNET_CHUNKTE_DBG] ep={ep_idx} steps={len(actions_taken)} "
                f"act_x[{_a[:,0].min():.1f},{_a[:,0].max():.1f}] final_cov={last_coverage:.3f}",
                flush=True,
            )
        return last_coverage, frames, actions_taken
