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
        fixed_goal: list | None = None,
        run_full_horizon: bool = False,
    ):
        self.run_full_horizon = run_full_horizon
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
        # FIXED-GOAL + RANDOM-INIT eval: when set (list [x,y,theta]), the seeds/
        # random branches reset() (random object/goal/pusher) then override ONLY
        # the goal via set_state(goal_pose=...), leaving the randomly-sampled T
        # object + pusher untouched. This gives "fixed goal at training target,
        # randomized initial T" — the correct eval for a FIXED-GOAL dataset.
        self.fixed_goal = (
            [float(x) for x in fixed_goal] if fixed_goal is not None else None
        )

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
            self._apply_fixed_goal(env, ep_seed_offset)
        elif self.init_mode == "seeds":
            if not self.init_seeds:
                raise ValueError("init_mode='seeds' requires init_seeds")
            seed = self.init_seeds[self._init_counter % len(self.init_seeds)]
            self._init_counter += 1
            env.reset(seed=int(seed))
            self._apply_fixed_goal(env, int(seed))

        return env._get_obs()

    def _apply_fixed_goal(self, env, seed_for_dbg: int) -> None:
        """If self.fixed_goal is set, override ONLY the goal (keep the random
        object + pusher from reset). Prints a [ROLLOUT_DBG] line so we can verify
        the goal is fixed while the sampled T/pusher vary across seeds."""
        if self.fixed_goal is None:
            return
        gp = tuple(self.fixed_goal[:3])
        env.set_state(goal_pose=gp)  # agent_pos/object_pose left None -> random T+pusher kept
        try:
            obs0 = env._get_obs()
            agent = np.asarray(obs0["agent_pos"]).reshape(-1).tolist()
            objp = np.asarray(obs0["object_pose"]).reshape(-1).tolist()
        except Exception:
            agent = objp = None
        print(
            f"[ROLLOUT_DBG] FIXED_GOAL seed={seed_for_dbg} goal_pose={list(gp)} "
            f"rand_agent={[round(float(x),2) for x in agent] if agent else None} "
            f"rand_obj={[round(float(x),2) for x in objp] if objp else None}",
            flush=True,
        )

    @torch.no_grad()
    def _rollout_one(
        self,
        sample: Dict[str, torch.Tensor],
        seq_len: int,
        emb_id: int,
        ep_idx: int,
    ) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
        """One closed-loop sim rollout. Returns (final_coverage, frames, actions)."""
        if getattr(self, "rollout_mode", "ar") == "chunk_openloop":
            return self._rollout_chunk_openloop(sample, seq_len, emb_id, ep_idx)

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
            last_coverage = max(last_coverage, float(info.get("coverage", 0.0)))
            actions_taken.append(action_xy)

            frame = env.render()
            if frame is not None:
                frames.append(np.ascontiguousarray(frame))

            if (terminated and not self.run_full_horizon) or last_coverage >= self.coverage_threshold:
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
        # --- env-gated per-episode rollout video dump (ROLLOUT_VIDEO_DIR) [ar] ---
        # Same additive pattern as the chunk_openloop/chunk_te dumps; no
        # behaviour change when ROLLOUT_VIDEO_DIR is unset.
        _rvd = os.environ.get("ROLLOUT_VIDEO_DIR")
        if _rvd and frames:
            try:
                import torchvision.io as tvio
                os.makedirs(_rvd, exist_ok=True)
                path = os.path.join(
                    _rvd, f"roll_ep{ep_idx}_cov{last_coverage:.3f}.mp4"
                )
                vid = torch.from_numpy(np.stack(frames, axis=0)).to(torch.uint8)
                tvio.write_video(path, vid, fps=int(self.video_fps), video_codec="h264")
                print(
                    f"[ROLLOUT_VIDEO] ep={ep_idx} wrote {path} "
                    f"({len(frames)} frames) cov={last_coverage:.3f}",
                    flush=True,
                )
            except Exception as e:  # pragma: no cover
                print(f"[ROLLOUT_VIDEO] ep={ep_idx} FAILED: {e}", flush=True)
        return last_coverage, frames, actions_taken

    @torch.no_grad()
    def _rollout_chunk_openloop(
        self,
        sample: Dict[str, torch.Tensor],
        seq_len: int,
        emb_id: int,
        ep_idx: int,
    ) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
        """Chunk execution for HPT-style policies.

        Re-observe only at chunk boundaries. The policy cache predicts a chunk,
        executes each action from that chunk, and replans after ``chunk_k`` env
        steps. This avoids the base AR/TE loop clearing the chunk every frame.
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
        obs_formatter = _OBS_FORMATTERS[self.embodiment_name]
        K = max(1, int(getattr(self, "chunk_k", 32)))

        frames: List[np.ndarray] = []
        actions_taken: List[np.ndarray] = []
        last_coverage = 0.0
        t = 0

        while t < T:
            obs_raw = obs_formatter(obs_env, device)
            obs_norm = algo.norm_stats.normalize(obs_raw, emb_id)

            state["action_chunk"] = None
            state["chunk_idx"] = 0
            n = min(K, T - t)
            terminated = False

            for _ in range(n):
                a_t_norm = algo.sim_predict_step(state, obs_norm, t, emb_id).squeeze(0)
                a_t_world = (
                    algo.norm_stats.unnormalize({ac_key: a_t_norm}, emb_id)[ac_key]
                    .detach().cpu().numpy().reshape(-1)
                )
                action_xy = np.array(
                    [float(a_t_world[0]), float(a_t_world[1])], dtype=np.float64
                )

                obs_env, _, terminated, _, info = env.step(action_xy)
                last_coverage = max(last_coverage, float(info.get("coverage", 0.0)))
                actions_taken.append(action_xy)

                frame = env.render()
                if frame is not None:
                    frames.append(np.ascontiguousarray(frame))

                t += 1
                if (terminated and not self.run_full_horizon) or last_coverage >= self.coverage_threshold:
                    break

            if (terminated and not self.run_full_horizon) or last_coverage >= self.coverage_threshold:
                break

        if actions_taken:
            _a = np.asarray(actions_taken)
            _first = [[round(float(x), 1) for x in p] for p in actions_taken[:6]]
            print(
                f"[HPT_CHUNKOL_DBG] ep={ep_idx} K={K} steps={len(actions_taken)} "
                f"act_x[{_a[:, 0].min():.1f},{_a[:, 0].max():.1f}] "
                f"act_y[{_a[:, 1].min():.1f},{_a[:, 1].max():.1f}] "
                f"first6={_first} final_cov={last_coverage:.3f}",
                flush=True,
            )

        _ctd = os.environ.get("CHUNK_TRAJ_DIR")
        if _ctd and ep_idx < 4 and actions_taken:
            try:
                os.makedirs(_ctd, exist_ok=True)
                _acts = np.asarray(actions_taken, dtype=np.float64)
                np.savez(
                    os.path.join(_ctd, f"hpt_chunktraj_ep{ep_idx}.npz"),
                    actions=_acts,
                    K=int(K),
                    coverage=float(last_coverage),
                )
                print(
                    f"[HPT_CHUNK_TRAJ] ep{ep_idx} saved actions{_acts.shape} "
                    f"K={K} cov={last_coverage:.3f}",
                    flush=True,
                )
            except Exception as e:  # pragma: no cover
                print(f"[HPT_CHUNK_TRAJ] ep={ep_idx} FAILED: {e}", flush=True)

        _rvd = os.environ.get("ROLLOUT_VIDEO_DIR")
        if _rvd and frames:
            try:
                import torchvision.io as tvio
                os.makedirs(_rvd, exist_ok=True)
                path = os.path.join(_rvd, f"roll_ep{ep_idx}_cov{last_coverage:.3f}.mp4")
                vid = torch.from_numpy(np.stack(frames, axis=0)).to(torch.uint8)
                tvio.write_video(path, vid, fps=int(self.video_fps), video_codec="h264")
                print(
                    f"[ROLLOUT_VIDEO] ep={ep_idx} wrote {path} "
                    f"({len(frames)} frames) cov={last_coverage:.3f}",
                    flush=True,
                )
            except Exception as e:  # pragma: no cover
                print(f"[ROLLOUT_VIDEO] ep={ep_idx} FAILED: {e}", flush=True)

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


def _tf_action_dump(ev, sample, emb_id, ep_idx):
    """Teacher-forced dump done THE RIGHT WAY: pull a raw val batch and run it
    through the model's own ``process_batch_for_training`` (zarr-key resolution
    + norm_stats), so obs are exactly what the model trained on. Then plain
    teacher-forced ``policy.forward`` and save predicted-vs-GT actions (world
    coords) for the first TF_DUMP_N episodes. Runs once (ep_idx==0). Guarded by
    env TF_DUMP_DIR. Eval-only diagnostic; no behaviour change when unset."""
    import os
    import numpy as np
    import torch
    d = os.environ.get("TF_DUMP_DIR")
    if not d or ep_idx != 0:
        return
    try:
        algo = ev.model
        policy = algo.nets["policy"]
        n = int(os.environ.get("TF_DUMP_N", "4"))
        dl = ev.trainer.datamodule.val_dataloader()
        if isinstance(dl, (list, tuple)):
            dl = dl[0]
        raw = next(iter(dl))
        for _ in range(4):
            if hasattr(raw, 'items'):
                break
            if isinstance(raw, (tuple, list)) and len(raw):
                raw = raw[0]
        proc = algo.process_batch_for_training(raw)
        eid = emb_id if emb_id in proc else list(proc.keys())[0]
        pb = proc[eid]
        ackey = getattr(algo, "resolved_ac_keys", algo.ac_keys)[eid]
        cu = torch.as_tensor(pb["cu_seqlens"]).long().reshape(-1) if "cu_seqlens" in pb else None
        nep = (len(cu) - 1) if cu is not None else 1
        from egomimic.models.hnet_nets.action_heads import action_head_decode
        os.makedirs(d, exist_ok=True)
        # OBS-ONLY mode (env TF_DUMP_OBSONLY=1): predict each frame's action from
        # the OBSERVATION ALONE via policy.generate (the SAME path the chunk
        # open-loop rollout uses), taking chunk[0] as the predicted action a_t.
        # This matches how a token_dropout_p=1.0 chunk model was trained (action
        # tokens always BOS), unlike the default GT-actions forward below which is
        # OOD for such a model. Additive; no behaviour change when unset.
        obsonly = os.environ.get("TF_DUMP_OBSONLY") == "1"
        dev = next(policy.parameters()).device
        for e in range(min(n, nep)):
            s0, e0 = (int(cu[e]), int(cu[e + 1])) if cu is not None else (0, None)
            obs = {k: pb[k][s0:e0].unsqueeze(0) for k in ("front_img_1", "state_agent_obj") if k in pb}
            gt = pb[ackey][s0:e0]
            with torch.no_grad():
                if obsonly:
                    T_ck = int(os.environ.get("TF_DUMP_K", "32"))
                    Tep = gt.shape[0]
                    preds_t = []
                    for ti in range(Tep):
                        # single-frame obs (B=1): state (1,5) dim==2, img (1,3,96,96)
                        # dim==4 -- exactly the shape policy.generate expects (mirrors
                        # _rollout_chunk_openloop's obs_norm). pb is already normalized
                        # and on-device.
                        obs_t = {
                            k: pb[k][s0:e0][ti : ti + 1]
                            for k in ("front_img_1", "state_agent_obj")
                            if k in pb
                        }
                        chunk = policy.generate(obs_t, batch_size=1, device=dev, T=T_ck)
                        chunk = chunk[0] if chunk.dim() == 3 else chunk  # (T_ck, A)
                        preds_t.append(chunk[0])  # first action of the plan = a_t
                    pred = torch.stack(preds_t, dim=0)  # (Tep, A)
                else:
                    pred, _ = policy(gt.unsqueeze(0), obs)
                    pred = action_head_decode(policy, pred)
                    pred = pred[0]
            predw = algo.norm_stats.unnormalize({ackey: pred}, eid)[ackey].detach().cpu().numpy()
            gtw = algo.norm_stats.unnormalize({ackey: gt}, eid)[ackey].detach().cpu().numpy()
            np.savez(os.path.join(d, f"tf_ep{e}.npz"), pred=predw, gt=gtw)
            print(f"[TF_DUMP] ep{e} obsonly={obsonly} pred{predw.shape} gt{gtw.shape}", flush=True)
    except Exception as ex:
        import traceback
        print(f"[TF_DUMP_ERR] {ex}\n{traceback.format_exc()}", flush=True)


# ---------- DAgger data collection (policy rollout + expert relabel) ----------
# Guarded by env DAGGER_COLLECT_DIR. Executes the POLICY's action (with optional
# exploration noise) so the rollout drifts; records the EXPERT's scripted_action
# as the LABEL at each visited (drifted) state -> writes the training zarr format.
_DAGGER = {"writer": None, "n": 0}


def _dagger_enabled():
    import os
    return bool(os.environ.get("DAGGER_COLLECT_DIR"))


def _dagger_start(env):
    import os
    if not _dagger_enabled():
        return False
    if _DAGGER["writer"] is None:
        import atexit
        from Tsimulation.collect.zarr_writer import ZarrDemoWriter
        d = os.environ["DAGGER_COLLECT_DIR"]
        os.makedirs(d, exist_ok=True)
        _DAGGER["writer"] = ZarrDemoWriter(
            path=d, env_args={"collector": "dagger"},
            image_size=int(os.environ.get("DAGGER_IMG", "96")), fps=30,
        )
        atexit.register(_dagger_close)
    try:
        _DAGGER["writer"].start_episode(init_state=env.get_episode_init())
        return True
    except Exception as e:
        import traceback
        print(f"[DAGGER_ERR] start {e}\n{traceback.format_exc()}", flush=True)
        return False


def _dagger_record(obs_env):
    import numpy as np
    w = _DAGGER["writer"]
    if w is None:
        return
    try:
        from Tsimulation.collect.scripted_collect import scripted_action
        agent = np.asarray(obs_env["agent_pos"], dtype=np.float64).reshape(-1)[:2]
        obj = np.asarray(obs_env["object_pose"], dtype=np.float64).reshape(-1)
        goal = np.asarray(obs_env["goal_pose"], dtype=np.float64).reshape(-1)
        exp = scripted_action(agent_xy=agent, object_xy=obj[:2], goal_xy=goal[:2], world_size=512.0)
        w.add_step(
            image=obs_env["image"], pusher_obs_pose=obs_env["agent_pos"],
            object_obs_pose=obs_env["object_pose"], pusher_cmd_pose=exp,
            action=exp, reward=0.0, goal_pose=obs_env["goal_pose"],
        )
    except Exception as e:
        import traceback
        print(f"[DAGGER_ERR] record {e}\n{traceback.format_exc()}", flush=True)


def _dagger_noise(action_xy):
    import os
    import numpy as np
    sig = float(os.environ.get("DAGGER_NOISE", "0") or 0)
    if sig <= 0:
        return action_xy
    a = np.asarray(action_xy, dtype=np.float64) + np.random.normal(0.0, sig, size=2)
    return np.clip(a, 0.0, 512.0)


def _dagger_commit():
    w = _DAGGER["writer"]
    if w is None:
        return
    try:
        if w.steps_in_episode > 0:
            idx = w.commit_episode()
            if idx is not None and idx >= 0:
                _DAGGER["n"] += 1
                print(f"[DAGGER] committed episode (total={_DAGGER['n']})", flush=True)
        else:
            w.abort_episode()
    except Exception as e:
        print(f"[DAGGER_ERR] commit {e}", flush=True)


def _dagger_close():
    w = _DAGGER["writer"]
    if w is not None:
        try:
            w.close()
        except Exception:
            pass
        _DAGGER["writer"] = None
        print(f"[DAGGER] closed writer, {_DAGGER['n']} episodes", flush=True)


def _probe_dump(ev, sample, emb_id, ep_idx):
    """Image-understanding probe dump: run the FROZEN image encoder on val frames
    and save (features, GT object(T) xy, GT goal xy). A linear probe trained on
    this tells us whether the vision actually encodes where the T / goal are.
    Guarded by env PROBE_DUMP_DIR. Runs once (ep_idx==0)."""
    import os
    import numpy as np
    import torch
    d = os.environ.get("PROBE_DUMP_DIR")
    if not d or ep_idx != 0:
        return
    try:
        algo = ev.model
        policy = algo.nets["policy"]
        enc = policy.cond_encoder.img_encoders["front_img_1"]
        nb = int(os.environ.get("PROBE_BATCHES", "6"))
        dl = ev.trainer.datamodule.val_dataloader()
        if isinstance(dl, (list, tuple)):
            dl = dl[0]
        it = iter(dl)
        F, OBJ, GOAL = [], [], []
        for _ in range(nb):
            raw = next(it)
            for _u in range(4):
                if hasattr(raw, "items"):
                    break
                if isinstance(raw, (tuple, list)) and len(raw):
                    raw = raw[0]
            proc = algo.process_batch_for_training(raw)
            eid = emb_id if emb_id in proc else list(proc.keys())[0]
            pb = proc[eid]
            img = pb["front_img_1"]
            imgb = img.unsqueeze(1) if img.dim() == 4 else img      # (N, 1, C, H, W)
            with torch.no_grad():
                feat = enc(imgb)
            feat = feat.reshape(feat.shape[0], -1)                  # (N, Fdim)
            raw_st = algo.norm_stats.unnormalize(
                {"state_agent_obj": pb["state_agent_obj"]}, eid
            )["state_agent_obj"]
            obj = raw_st.reshape(raw_st.shape[0], -1)[:, 2:4]        # object (T) xy
            g = pb.get("goal_pose")
            gxy = (torch.as_tensor(g).reshape(g.shape[0], -1)[:, :2]
                   if g is not None else torch.zeros(obj.shape[0], 2, device=obj.device))
            n = min(feat.shape[0], obj.shape[0], gxy.shape[0])
            F.append(feat[:n].float().cpu().numpy())
            OBJ.append(obj[:n].float().cpu().numpy())
            GOAL.append(gxy[:n].float().cpu().numpy())
        os.makedirs(d, exist_ok=True)
        F = np.concatenate(F); OBJ = np.concatenate(OBJ); GOAL = np.concatenate(GOAL)
        np.savez(os.path.join(d, "probe.npz"), feats=F, obj=OBJ, goal=GOAL)
        print(f"[PROBE] saved feats{F.shape} obj{OBJ.shape} goal{GOAL.shape}", flush=True)
    except Exception as e:
        import traceback
        print(f"[PROBE_ERR] {e}\n{traceback.format_exc()}", flush=True)


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
        _tf_action_dump(self, sample, emb_id, ep_idx)
        _probe_dump(self, sample, emb_id, ep_idx)
        # --- env-gated SPLIT ROLLOUT diagnostic (SPLIT_DIAG_DIR) ---
        # Determines whether the ~4s failure is WORLD-STATE (covariate shift) or
        # HIDDEN ROLLOUT-STATE (a bug), by snapshotting the full env state at step
        # 120 (S120), then restarting a FRESH rollout from set_state(S120) with
        # fresh model/rollout buffers, and comparing the continuation actions
        # (rolloutA[120:240]) vs the restart actions (rolloutB[0:120]). Additive;
        # no behaviour change when SPLIT_DIAG_DIR is unset.
        if os.environ.get("SPLIT_DIAG_DIR"):
            return self._rollout_split(sample, seq_len, emb_id, ep_idx)
        if getattr(self, "rollout_mode", "ar") == "chunk_te":
            return self._rollout_chunk_te(sample, seq_len, emb_id, ep_idx)
        if getattr(self, "rollout_mode", "ar") == "chunk_openloop":
            return self._rollout_chunk_openloop(sample, seq_len, emb_id, ep_idx)
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
        _dagger_on = _dagger_start(env)
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

            if _dagger_on:
                _dagger_record(obs_env)
                action_xy = _dagger_noise(action_xy)
            obs_env, _, terminated, _, info = env.step(action_xy)
            last_coverage = max(last_coverage, float(info.get("coverage", 0.0)))
            actions_taken.append(action_xy)

            frame = env.render()
            if frame is not None:
                frames.append(np.ascontiguousarray(frame))
            if (terminated and not self.run_full_horizon) or last_coverage >= self.coverage_threshold:
                break

        if _dagger_on:
            _dagger_commit()
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
        # --- env-gated per-episode rollout video dump (ROLLOUT_VIDEO_DIR) [ar] ---
        # Same additive pattern as the chunk_openloop/chunk_te dumps; no
        # behaviour change when ROLLOUT_VIDEO_DIR is unset.
        _rvd = os.environ.get("ROLLOUT_VIDEO_DIR")
        if _rvd and frames:
            try:
                import torchvision.io as tvio
                os.makedirs(_rvd, exist_ok=True)
                path = os.path.join(
                    _rvd, f"roll_ep{ep_idx}_cov{last_coverage:.3f}.mp4"
                )
                vid = torch.from_numpy(np.stack(frames, axis=0)).to(torch.uint8)
                tvio.write_video(path, vid, fps=int(self.video_fps), video_codec="h264")
                print(
                    f"[ROLLOUT_VIDEO] ep={ep_idx} wrote {path} "
                    f"({len(frames)} frames) cov={last_coverage:.3f}",
                    flush=True,
                )
            except Exception as e:  # pragma: no cover
                print(f"[ROLLOUT_VIDEO] ep={ep_idx} FAILED: {e}", flush=True)
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
            last_coverage = max(last_coverage, float(info.get("coverage", 0.0)))
            actions_taken.append(action_xy)
            frame = env.render()
            if frame is not None:
                frames.append(np.ascontiguousarray(frame))
            if (terminated and not self.run_full_horizon) or last_coverage >= self.coverage_threshold:
                break
        # --- env-gated executed-trajectory dump (CHUNK_TRAJ_DIR); first ~4 eps ---
        # Mirrors _rollout_chunk_openloop's dump: guarded by an env var, wrapped
        # in try/except, additive; no behaviour change when unset.
        _ctd = os.environ.get("CHUNK_TRAJ_DIR")
        if _ctd and ep_idx < 4 and actions_taken:
            try:
                os.makedirs(_ctd, exist_ok=True)
                _acts = np.asarray(actions_taken, dtype=np.float64)  # (T,2) WORLD
                np.savez(
                    os.path.join(_ctd, f"chunktraj_te_ep{ep_idx}.npz"),
                    actions=_acts,
                    K=int(K),
                    coverage=float(last_coverage),
                )
                print(
                    f"[CHUNK_TRAJ_TE] ep{ep_idx} saved actions{_acts.shape} "
                    f"K={K} cov={last_coverage:.3f}",
                    flush=True,
                )
            except Exception as _ex:
                import traceback
                print(f"[CHUNK_TRAJ_TE_ERR] {_ex}\n{traceback.format_exc()}", flush=True)
        if actions_taken:
            _a = np.asarray(actions_taken)
            print(
                f"[HNET_CHUNKTE_DBG] ep={ep_idx} steps={len(actions_taken)} "
                f"act_x[{_a[:,0].min():.1f},{_a[:,0].max():.1f}] final_cov={last_coverage:.3f}",
                flush=True,
            )
        # --- env-gated per-episode rollout video dump (ROLLOUT_VIDEO_DIR) [chunk_te] ---
        _rvd = os.environ.get("ROLLOUT_VIDEO_DIR")
        if _rvd and frames:
            try:
                import torchvision.io as tvio
                os.makedirs(_rvd, exist_ok=True)
                path = os.path.join(_rvd, f"roll_ep{ep_idx}_cov{last_coverage:.3f}.mp4")
                vid = torch.from_numpy(np.stack(frames, axis=0)).to(torch.uint8)
                tvio.write_video(path, vid, fps=int(self.video_fps), video_codec="h264")
                print(f"[ROLLOUT_VIDEO] ep={ep_idx} wrote {path} ({len(frames)} frames) cov={last_coverage:.3f}", flush=True)
            except Exception as _ex:
                import traceback
                print(f"[ROLLOUT_VIDEO_ERR] {_ex}", flush=True)
        return last_coverage, frames, actions_taken

    @torch.no_grad()
    def _rollout_chunk_openloop(self, sample, seq_len, emb_id, ep_idx):
        """Chunked open-loop rollout. Observe ONE frame, predict the full
        K-action plan, EXECUTE ALL K raw (one per env.step), then re-observe
        and predict the next chunk. No temporal ensemble, no chunk_buf, no
        mid-chunk re-observation -- the chunked-policy execution semantics
        the chunk-A / chunk-B variants are trained for."""
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
        # OBS-HISTORY (N): condition each chunk on the last N chunk-anchor obs
        # (the frames the policy re-observes, one per replanning step). N=1 ->
        # single-frame (memoryless), byte-identical to before. Episode start
        # pads by repeating the first observation. Must mirror the trainer's
        # sim history (chunk-anchor granularity, oldest->newest).
        N = max(1, int(getattr(algo, "n_obs_history", 1)))
        obs_hist: List[dict] = []
        frames: List[np.ndarray] = []
        actions_taken: List[np.ndarray] = []
        last_coverage = 0.0
        t = 0
        while t < T_eff:
            # Observe the current frame and predict the full K-action plan.
            obs_raw = obs_formatter(obs_env, device)
            if self.goal_in_obs and "goal_pose" in obs_env:
                obs_raw["goal_obs"] = torch.as_tensor(
                    np.asarray(obs_env["goal_pose"], dtype=np.float32).reshape(-1)[:3],
                    device=device,
                ).reshape(1, 3)
            obs_norm = algo.norm_stats.normalize(obs_raw, emb_id)
            n = min(K, T_eff - t)
            if N > 1:
                obs_hist.append(obs_norm)
                if len(obs_hist) > N:
                    obs_hist = obs_hist[-N:]
                win = ([obs_hist[0]] * (N - len(obs_hist))) + obs_hist  # pad-by-repeat (oldest)
                obs_gen = {
                    k: torch.stack([f[k] for f in win], dim=1)  # (1, N, ...)
                    for k in obs_norm
                }
            else:
                obs_gen = obs_norm
            chunk = policy.generate(obs_gen, batch_size=1, device=device, T=n)  # (n, A) norm
            chunk = chunk[0] if chunk.dim() == 3 else chunk                       # (n, A)
            chunk_world = (
                algo.norm_stats.unnormalize({ac_key: chunk}, emb_id)[ac_key]
                .detach().cpu().numpy()
            )  # (n, A)
            # Execute ALL n actions raw, one per env.step (no re-observe mid-chunk).
            terminated = False
            for j in range(chunk_world.shape[0]):
                a_world = chunk_world[j].reshape(-1)
                action_xy = np.array([float(a_world[0]), float(a_world[1])], dtype=np.float64)
                obs_env, _, terminated, _, info = env.step(action_xy)
                last_coverage = max(last_coverage, float(info.get("coverage", 0.0)))
                actions_taken.append(action_xy)
                frame = env.render()
                if frame is not None:
                    frames.append(np.ascontiguousarray(frame))
                t += 1
                if (terminated and not self.run_full_horizon) or last_coverage >= self.coverage_threshold:
                    break
            if (terminated and not self.run_full_horizon) or last_coverage >= self.coverage_threshold:
                break
        # --- env-gated executed-trajectory dump (CHUNK_TRAJ_DIR); first ~4 eps ---
        # Mirrors the _tf_action_dump / _probe_dump pattern: guarded by an env
        # var, wrapped in try/except, additive; no behaviour change when unset.
        _ctd = os.environ.get("CHUNK_TRAJ_DIR")
        if _ctd and ep_idx < 4 and actions_taken:
            try:
                os.makedirs(_ctd, exist_ok=True)
                _acts = np.asarray(actions_taken, dtype=np.float64)  # (T,2) WORLD
                np.savez(
                    os.path.join(_ctd, f"chunktraj_ep{ep_idx}.npz"),
                    actions=_acts,
                    K=int(K),
                    coverage=float(last_coverage),
                )
                print(
                    f"[CHUNK_TRAJ] ep{ep_idx} saved actions{_acts.shape} "
                    f"K={K} cov={last_coverage:.3f}",
                    flush=True,
                )
            except Exception as _ex:
                import traceback
                print(f"[CHUNK_TRAJ_ERR] {_ex}\n{traceback.format_exc()}", flush=True)
        if actions_taken:
            _a = np.asarray(actions_taken)
            print(
                f"[HNET_CHUNKOL_DBG] ep={ep_idx} K={K} steps={len(actions_taken)} "
                f"act_x[{_a[:,0].min():.1f},{_a[:,0].max():.1f}] "
                f"act_y[{_a[:,1].min():.1f},{_a[:,1].max():.1f}] final_cov={last_coverage:.3f}",
                flush=True,
            )
        # --- env-gated per-episode rollout video dump (ROLLOUT_VIDEO_DIR) ---
        # Mirrors _rollout_split's video writing exactly (same tvio import,
        # (T,H,W,3) uint8, fps). Additive, wrapped in try/except; no behaviour
        # change when ROLLOUT_VIDEO_DIR is unset.
        _rvd = os.environ.get("ROLLOUT_VIDEO_DIR")
        if _rvd and frames:
            try:
                import torchvision.io as tvio
                os.makedirs(_rvd, exist_ok=True)
                path = os.path.join(
                    _rvd, f"roll_ep{ep_idx}_cov{last_coverage:.3f}.mp4"
                )
                vid = torch.from_numpy(np.stack(frames, axis=0)).to(torch.uint8)  # (T,H,W,3)
                tvio.write_video(path, vid, fps=int(self.video_fps), video_codec="h264")
                print(
                    f"[ROLLOUT_VIDEO] ep={ep_idx} wrote {path} "
                    f"({len(frames)} frames) cov={last_coverage:.3f}",
                    flush=True,
                )
            except Exception as _ex:
                import traceback
                print(f"[ROLLOUT_VIDEO_ERR] {_ex}\n{traceback.format_exc()}", flush=True)
        return last_coverage, frames, actions_taken

    # ------------------------------------------------------------------ #
    # SPLIT ROLLOUT DIAGNOSTIC
    # ------------------------------------------------------------------ #
    @staticmethod
    def _snapshot_env(env):
        """Full world state = the SAME (agent_pos, object_pose, goal_pose) tuple
        that VecPushShapes.set_states / _init_env feed to env.set_state. Read live
        from the env's public properties at the current step."""
        return (
            tuple(float(x) for x in env.agent_pos),
            tuple(float(x) for x in env.object_pose),
            tuple(float(x) for x in env.goal_pose),
        )

    @torch.no_grad()
    def _split_run_openloop(self, env, n_steps, K, ac_key, emb_id, device,
                            obs_formatter, snap_at=None, frames=None):
        """chunk_openloop loop (predict K from one obs, execute all K, re-observe),
        run for n_steps from the env's CURRENT state. Mirrors _rollout_chunk_openloop
        exactly. Optionally snapshot the full env state right AFTER executing the
        step with index == snap_at. Returns (actions (n,2) WORLD, S_snap or None)."""
        algo = self.model
        policy = algo.nets["policy"]
        acts = []
        S_snap = None
        t = 0
        while t < n_steps:
            obs_env = env._get_obs()
            obs_raw = obs_formatter(obs_env, device)
            obs_norm = algo.norm_stats.normalize(obs_raw, emb_id)
            n = min(K, n_steps - t)
            chunk = policy.generate(obs_norm, batch_size=1, device=device, T=n)
            chunk = chunk[0] if chunk.dim() == 3 else chunk
            chunk_world = (
                algo.norm_stats.unnormalize({ac_key: chunk}, emb_id)[ac_key]
                .detach().cpu().numpy()
            )
            for j in range(chunk_world.shape[0]):
                a = chunk_world[j].reshape(-1)
                action_xy = np.array([float(a[0]), float(a[1])], dtype=np.float64)
                env.step(action_xy)
                acts.append(action_xy)
                if frames is not None:
                    fr = env.render()
                    if fr is not None:
                        frames.append(np.ascontiguousarray(fr))
                if snap_at is not None and t == snap_at:
                    S_snap = self._snapshot_env(env)
                t += 1
        return np.asarray(acts, dtype=np.float64), S_snap

    @torch.no_grad()
    def _split_run_te(self, env, n_steps, K, ac_key, emb_id, device,
                      obs_formatter, snap_at=None, frames=None):
        """chunk_te loop (per-step generate + temporal ensemble over a fresh
        chunk_buf), run for n_steps from the env's CURRENT state. Mirrors
        _rollout_chunk_te exactly: a FRESH empty chunk_buf is created here, so a
        restart begins with an empty TE buffer (the warmup state). Optionally
        snapshot env state after step index == snap_at."""
        algo = self.model
        policy = algo.nets["policy"]
        chunk_buf = _deque_te()
        acts = []
        S_snap = None
        for t in range(n_steps):
            obs_env = env._get_obs()
            obs_raw = obs_formatter(obs_env, device)
            obs_norm = algo.norm_stats.normalize(obs_raw, emb_id)
            chunk = policy.generate(obs_norm, batch_size=1, device=device, T=min(K, n_steps - t))
            chunk_buf.append((t, chunk[0].detach()))
            a_norm = _temporal_ensemble(chunk_buf, t, K, self.te_m)
            a_world = (
                algo.norm_stats.unnormalize({ac_key: a_norm}, emb_id)[ac_key]
                .detach().cpu().numpy().reshape(-1)
            )
            action_xy = np.array([float(a_world[0]), float(a_world[1])], dtype=np.float64)
            env.step(action_xy)
            acts.append(action_xy)
            if frames is not None:
                fr = env.render()
                if fr is not None:
                    frames.append(np.ascontiguousarray(fr))
            if snap_at is not None and t == snap_at:
                S_snap = self._snapshot_env(env)
        return np.asarray(acts, dtype=np.float64), S_snap

    @torch.no_grad()
    def _rollout_split(self, sample, seq_len, emb_id, ep_idx):
        """Split-rollout diagnostic for ONE episode and ONE rollout mode (read
        from self.rollout_mode). Steps:
          A) demo-init replay rollout for SPLIT+CONT steps. Render -> rolloutA.
             Snapshot full env state S120 right after step (SPLIT-1).
          B) Fresh env: reset(seed=0) + set_state(S120), fresh model/rollout
             buffers, run CONT steps. Render -> rolloutB.
          C) Compare actions A[SPLIT:SPLIT+CONT] (continuation) vs B[0:CONT]
             (restart). Report mean per-step |diff| in WORLD px, split first-K
             vs after-K. Dump npz + 2 videos per (mode, ep).
        Gated by SPLIT_DIAG_DIR; only runs for ep_idx < SPLIT_DIAG_NEPS (default 1).
        """
        import torchvision.io as tvio
        d = os.environ["SPLIT_DIAG_DIR"]
        n_eps = int(os.environ.get("SPLIT_DIAG_NEPS", "1"))
        SPLIT = int(os.environ.get("SPLIT_DIAG_SPLIT", "120"))   # step index of S120
        CONT = int(os.environ.get("SPLIT_DIAG_CONT", "120"))     # continuation length
        mode = getattr(self, "rollout_mode", "chunk_openloop")
        # One-shot guard: the eval harness may call us across several val
        # batches (debug trainer forces limit_val_batches>1), each re-using
        # ep_idx 0..B-1. Run each (mode, ep_idx) EXACTLY ONCE so videos/npz are
        # the FIRST batch's episodes and aren't overwritten by later batches.
        if not hasattr(self, "_split_done"):
            self._split_done = set()
        key = (mode, ep_idx)
        if ep_idx >= n_eps or key in self._split_done:
            # Skip extra/duplicate episodes cheaply (return an empty rollout).
            return 0.0, [], []
        self._split_done.add(key)
        os.makedirs(d, exist_ok=True)
        algo = self.model
        policy = algo.nets["policy"]
        env = self._get_env()
        device = self.trainer.lightning_module.device
        ac_key = getattr(algo, "resolved_ac_keys", algo.ac_keys)[emb_id]
        obs_formatter = _OBS_FORMATTERS[self.embodiment_name]
        K = max(1, int(getattr(self, "chunk_k", 32)))
        # action_horizon cap: rolloutA needs SPLIT+CONT steps -> ensure room.
        _ah = int(getattr(policy, "action_horizon", 0) or 0)
        runner = (self._split_run_openloop if mode == "chunk_openloop"
                  else self._split_run_te)

        # mode=train validation hands a NORMALIZED batch; env init needs RAW
        # world-coord state. Un-normalize like the other rollout methods do.
        sample = dict(sample)
        if "state_agent_obj" in sample:
            sample["state_agent_obj"] = algo.norm_stats.unnormalize(
                {"state_agent_obj": sample["state_agent_obj"]}, emb_id
            )["state_agent_obj"]

        # --- (A) demo-init rollout, SPLIT+CONT steps, snapshot S120 ---
        self._init_env(env, sample, ep_seed_offset=ep_idx, emb_id=emb_id)
        framesA = []
        n_A = SPLIT + CONT
        if _ah > 0 and n_A > _ah:
            print(f"[SPLIT_DIAG_WARN] n_A={n_A} > action_horizon={_ah}; "
                  f"capping (generate raises past horizon)", flush=True)
        actsA, S120 = runner(env, n_A, K, ac_key, emb_id, device, obs_formatter,
                             snap_at=SPLIT - 1, frames=framesA)
        if S120 is None:
            print(f"[SPLIT_DIAG_ERR] ep={ep_idx} mode={mode}: did not reach "
                  f"snap step {SPLIT-1} (only {len(actsA)} steps).", flush=True)
            return 0.0, framesA, list(actsA)
        print(f"[SPLIT_DIAG] ep={ep_idx} mode={mode} S120="
              f"agent={tuple(round(x,2) for x in S120[0])} "
              f"obj={tuple(round(x,3) for x in S120[1])} "
              f"goal={tuple(round(x,3) for x in S120[2])} "
              f"actsA={actsA.shape}", flush=True)

        # --- (B) fresh restart from S120, fresh buffers, CONT steps ---
        env.reset(seed=_REPLAY_RESET_SEED)
        env.set_state(agent_pos=S120[0], object_pose=S120[1], goal_pose=S120[2])
        # Verify set_state reproduced the world exactly.
        Schk = self._snapshot_env(env)
        rep_err = max(
            max(abs(a - b) for a, b in zip(Schk[0], S120[0])),
            max(abs(a - b) for a, b in zip(Schk[1], S120[1])),
            max(abs(a - b) for a, b in zip(Schk[2], S120[2])),
        )
        framesB = []
        actsB, _ = runner(env, CONT, K, ac_key, emb_id, device, obs_formatter,
                          snap_at=None, frames=framesB)
        print(f"[SPLIT_DIAG] ep={ep_idx} mode={mode} set_state_repro_err={rep_err:.3e} "
              f"actsB={actsB.shape}", flush=True)

        # --- (C) compare continuation A[SPLIT:] vs restart B[0:] ---
        contA = actsA[SPLIT:SPLIT + CONT]            # continuation past S120
        restB = actsB[:CONT]                          # restart from S120
        m = min(contA.shape[0], restB.shape[0])
        contA, restB = contA[:m], restB[:m]
        per_step = np.linalg.norm(contA - restB, axis=1)   # WORLD px per step
        first_k = float(per_step[:K].mean()) if m > 0 else float("nan")
        after_k = float(per_step[K:].mean()) if m > K else float("nan")
        overall = float(per_step.mean()) if m > 0 else float("nan")
        print(
            f"[SPLIT_DIAG_RESULT] ep={ep_idx} mode={mode} m={m} K={K} "
            f"set_state_repro_err={rep_err:.3e} "
            f"mean_abs_action_diff_px: overall={overall:.4f} "
            f"first{K}={first_k:.4f} after{K}={after_k:.4f}",
            flush=True,
        )

        # --- dump npz + videos ---
        tag = f"{mode}_ep{ep_idx}"
        np.savez(
            os.path.join(d, f"split_{tag}.npz"),
            actsA=actsA, actsB=actsB,
            contA=contA, restB=restB, per_step=per_step,
            S120_agent=np.asarray(S120[0]), S120_obj=np.asarray(S120[1]),
            S120_goal=np.asarray(S120[2]),
            set_state_repro_err=np.asarray(rep_err),
            SPLIT=SPLIT, CONT=CONT, K=K,
            first_k=first_k, after_k=after_k, overall=overall,
        )
        for nm, frs in (("rolloutA", framesA), ("rolloutB", framesB)):
            if frs:
                path = os.path.join(d, f"{nm}_{tag}.mp4")
                vid = torch.from_numpy(np.stack(frs, axis=0)).to(torch.uint8)  # (T,H,W,3)
                tvio.write_video(path, vid, fps=int(self.video_fps), video_codec="h264")
                print(f"[SPLIT_DIAG_VIDEO] wrote {path} ({len(frs)} frames)", flush=True)
        # Return rolloutA as the "rollout" for the harness (coverage unused here).
        return 0.0, framesA, list(actsA)
