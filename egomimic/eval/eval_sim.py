"""
Generic closed-loop sim evaluator.

Wraps the ``Tsimulation.pushshapes.PushShapesEnv`` simulator. For each
validation episode, resets the env (init mode is config-driven), then
steps the env one frame at a time by delegating inference to the algo's
own ``sim_init_state`` + ``sim_predict_step`` methods.

The eval loop itself is algo-agnostic:

  1. Reset env (replay frame-0 state from the val episode, or fresh seed).
  2. ``state = algo.sim_init_state(batch_size, T_max, device, emb_id)``.
  3. For each sim step ``t``:
       obs_env = env._get_obs()
       obs_norm = format & normalize obs for this embodiment
       a_t_norm = algo.sim_predict_step(state, obs_norm, t, emb_id)
       a_t_world = algo.norm_stats.unnormalize(a_t_norm)
       env.step(a_t_world)
       render frame
  4. Record ``env._coverage()`` at episode end.

Algos that want to support sim eval implement two methods (see
``egomimic.algo.hnet.HNet.sim_init_state`` for the reference impl):

  ``sim_init_state(batch_size, T_max, device, emb_id) -> Any``
      Build the inference state object (KV cache + position pointers for
      AR transformers; action-chunk buffer for diffusion policies). The
      returned object is opaque to the eval loop — only the algo's own
      ``sim_predict_step`` reads it.

  ``sim_predict_step(state, obs_norm, t, emb_id) -> Tensor (1, 1, A)``
      Given fresh single-frame normalized obs and the inference state,
      return the action prediction for sim step ``t`` (in normalized
      action space — the eval loop applies ``norm_stats.unnormalize``
      before passing it to ``env.step``). May mutate ``state`` in place
      to advance KV / chunk pointers.

Coverage / success / videos are computed once per episode regardless of
the algo.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from egomimic.eval.eval_video import EvalVideo


# Maps the env's ``_get_obs()`` output into the algo's expected obs dict.
# Per-embodiment because different embodiments have different obs keys.
def _format_pushshapes_obs(obs_env: dict, device: torch.device) -> dict:
    """Format PushShapesEnv obs → model batch (B=1, …).

    Mirrors what the dataset produces for pushshapes_sim:
      observations.state -> ``state_agent_obj`` (5,) = concat(pusher, obj)
      observations.images.front_img_1 -> ``front_img_1`` (3, H, W) float [0,1]
    """
    state_5 = np.concatenate(
        [obs_env["agent_pos"], obs_env["object_pose"]], axis=0
    ).astype(np.float32)
    image_chw = np.transpose(obs_env["image"], (2, 0, 1)).astype(np.float32) / 255.0
    return {
        "state_agent_obj": torch.from_numpy(state_5).unsqueeze(0).to(device),
        "front_img_1": torch.from_numpy(image_chw).unsqueeze(0).to(device),
    }


# Embodiment-name → obs-formatter. Add new entries when you wire a new
# embodiment's data pipeline.
_OBS_FORMATTERS = {
    "pushshapes_sim": _format_pushshapes_obs,
}


def _state_to_init(state: np.ndarray) -> tuple:
    """Split the dataset's (5,) ``observations.state`` vector into
    PushShapesEnv ``set_state`` args. Mirrors ``zarr_writer``:
    state = concat(pusher_xy, object_xyangle).
    """
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] < 5:
        raise ValueError(f"expected state of len >= 5, got {state.shape}")
    agent_pos = (float(state[0]), float(state[1]))
    object_pose = (float(state[2]), float(state[3]), float(state[4]))
    return agent_pos, object_pose


class SimRolloutEval(EvalVideo):
    """Closed-loop sim rollout eval. Algo-agnostic — delegates inference to
    ``algo.sim_init_state`` + ``algo.sim_predict_step``.

    Args (from hydra yaml):
      env_kwargs: dict of kwargs passed to ``PushShapesEnv(**env_kwargs)``.
      embodiment_name: name of the embodiment driving this env (matches the
        key in ``algo.ac_keys`` / ``algo.norm_stats``). Used to pick the
        obs formatter and the action key for unnormalize.
      init_mode: ``"replay"`` reads frame-0 state from the val batch and
        seeds the env with it; ``"random"`` calls ``env.reset(seed=ep_idx)``;
        ``"seeds"`` cycles through ``init_seeds``.
      init_seeds: list[int] for ``init_mode='seeds'``.
      max_steps: optional per-episode step cap. When ``None`` and
        ``init_mode='replay'`` the recorded episode length is used.
      coverage_threshold: success threshold for ``_sim_success_rate``.
      video_fps: encoder fps for the saved validation videos.
      limit_val_batches: inherited from ``EvalVideo``.
    """

    def __init__(
        self,
        env_kwargs: dict | None = None,
        embodiment_name: str = "pushshapes_sim",
        init_mode: str = "replay",
        init_seeds: list[int] | None = None,
        max_steps: int | None = None,
        coverage_threshold: float = 0.7,
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
        )
        self.env_kwargs = dict(env_kwargs or {})
        self.embodiment_name = str(embodiment_name)
        if self.embodiment_name not in _OBS_FORMATTERS:
            raise ValueError(
                f"No obs formatter registered for embodiment {self.embodiment_name!r}. "
                f"Add an entry to ``_OBS_FORMATTERS`` in ``eval_sim.py``."
            )
        self.init_mode = str(init_mode)
        if self.init_mode not in {"replay", "random", "seeds"}:
            raise ValueError(
                f"init_mode must be one of replay/random/seeds, got {self.init_mode!r}"
            )
        self.init_seeds = list(init_seeds or [])
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.coverage_threshold = float(coverage_threshold)
        self.video_fps = int(video_fps)
        self.max_videos = int(max_videos) if max_videos is not None else None
        self._env = None
        self._init_counter = 0

    def video_dir(self):
        # Distinct subdir so sim rollouts don't collide with composite
        # val viz output (both default to <root>/videos/epoch_N/<emb>/...).
        import os as _os

        return _os.path.join(self.root_dir(), "videos", "sim")

    # ------------------------------------------------------------------ #

    def _get_env(self):
        if self._env is None:
            kwargs = dict(self.env_kwargs)
            kwargs.setdefault("render_mode", "rgb_array")
            from Tsimulation.pushshapes import PushShapesEnv

            self._env = PushShapesEnv(**kwargs)
        return self._env

    def _init_env(self, env, sample: dict, ep_seed_offset: int, emb_id: int) -> None:
        """Reset + (optionally) set_state. Encapsulates init_mode.

        Replay mode reads ``state_agent_obj`` and ``goal_pose`` from the
        batch and **unnormalizes** them before calling ``env.set_state``:
        the batch was passed through ``process_batch_for_training`` which
        normalizes obs, but the env expects world coordinates.
        """
        if self.init_mode == "replay":
            state_seq = sample.get("state_agent_obj")
            if state_seq is None:
                raise KeyError("init_mode='replay' requires 'state_agent_obj' in batch")

            # Build a single-key dict and unnormalize → world coords.
            unnorm = self.model.norm_stats.unnormalize(
                {"state_agent_obj": state_seq}, emb_id
            )
            frame0 = unnorm["state_agent_obj"][0].detach().cpu().numpy()
            agent_pos, object_pose = _state_to_init(frame0)

            goal_pose = None
            goal_seq = sample.get("goal_pose")
            if goal_seq is not None:
                # ``goal_pose`` isn't a normalized key (see
                # ``MultiDataset.norm_stats`` keys); pass straight through.
                goal_pose = tuple(
                    float(x)
                    for x in np.asarray(goal_seq[0].detach().cpu().numpy()).reshape(-1)[
                        :3
                    ]
                )
            env.reset(seed=ep_seed_offset)
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

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _rollout_one(
        self,
        sample: Dict[str, torch.Tensor],
        seq_len: int,
        emb_id: int,
        ep_idx: int,
    ) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
        """One closed-loop sim rollout. Returns ``(final_coverage, frames,
        actions_taken)``.
        """
        algo = self.model
        device = self.trainer.lightning_module.device
        env = self._get_env()

        self._init_env(env, sample, ep_seed_offset=ep_idx, emb_id=emb_id)
        T_eff = self.max_steps if self.max_steps is not None else int(seq_len)

        if not hasattr(algo, "sim_init_state") or not hasattr(algo, "sim_predict_step"):
            raise RuntimeError(
                f"Algo {type(algo).__name__} does not implement "
                "``sim_init_state`` / ``sim_predict_step`` — required for "
                "SimRolloutEval."
            )
        state = algo.sim_init_state(
            batch_size=1,
            T_max=T_eff,
            device=device,
            emb_id=emb_id,
        )
        # The algo may have capped T_max internally (e.g. by its pos_emb).
        T = state.get("T_max", T_eff)

        # ``resolved_ac_keys`` is HNet's id-keyed map; HPT's ``ac_keys``
        # is dual-keyed (both name and id) so both work via the same access.
        ac_key = getattr(algo, "resolved_ac_keys", algo.ac_keys)[emb_id]
        obs_formatter = _OBS_FORMATTERS[self.embodiment_name]

        frames: List[np.ndarray] = []
        actions_taken: List[np.ndarray] = []
        last_coverage = 0.0

        for t in range(T):
            obs_env = env._get_obs()
            obs_raw = obs_formatter(obs_env, device)
            obs_norm = algo.norm_stats.normalize(obs_raw, emb_id)

            a_t_norm = algo.sim_predict_step(state, obs_norm, t, emb_id)
            # Unnormalize → world coords, drop batch dim.
            a_t_world = (
                algo.norm_stats.unnormalize({ac_key: a_t_norm.squeeze(0)}, emb_id)[
                    ac_key
                ]
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            action_xy = np.array(
                [float(a_t_world[0]), float(a_t_world[1])], dtype=np.float32
            )

            _, _, terminated, _, info = env.step(action_xy)
            last_coverage = float(info.get("coverage", 0.0))
            actions_taken.append(action_xy)

            frame = env.render()
            if frame is not None:
                frames.append(np.ascontiguousarray(frame))

            if terminated:
                break

        return last_coverage, frames, actions_taken

    # ------------------------------------------------------------------ #

    def compute_metrics_and_viz(
        self, batch: Dict[int, Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, np.ndarray]]:
        metrics: Dict[str, torch.Tensor] = {}
        images_dict: Dict[int, np.ndarray] = {}
        device = self.trainer.lightning_module.device

        for emb_id, _batch in batch.items():
            is_packed = _batch.get("_packed", False)

            # Non-packed (HPT per-frame loader): each row of the batch is one
            # action-chunk sample. Synthesize a packed-style batch by
            # treating the first dimension as the episode index and the
            # action-horizon T as the sub-episode length. ``_init_env``
            # uses the first frame of ``state_agent_obj`` as the env init
            # state — for HPT that's the obs at the chunk's start.
            if not is_packed:
                state_key = "state_agent_obj"
                if state_key not in _batch:
                    continue
                state = _batch[state_key]
                if state.dim() < 2:
                    continue
                # Cap to a sensible number of rollouts in the smoke path —
                # SimRolloutEval doesn't currently know about --n-episodes
                # for non-packed input, and 128 × max_steps rollouts is huge.
                B_hpt = min(
                    int(state.shape[0]), int(getattr(self, "limit_val_batches", 4) or 4)
                )
                state = state[:B_hpt]
                # Treat each row as a 1-frame episode for init purposes.
                cu = torch.arange(B_hpt + 1, device=state.device, dtype=torch.long)
                seq_lens = torch.ones(B_hpt, device=state.device, dtype=torch.long)
                # Re-pack the per-frame batch by taking frame 0 of each row.
                _packed_batch: dict = {}
                for k, v in _batch.items():
                    if torch.is_tensor(v) and v.dim() >= 2 and v.shape[0] >= B_hpt:
                        _packed_batch[k] = v[:B_hpt, 0]
                    else:
                        _packed_batch[k] = v
                _packed_batch["cu_seqlens"] = cu
                _packed_batch["seq_lens"] = seq_lens
                _batch = _packed_batch
                cu = _batch["cu_seqlens"]
                seq_lens = _batch["seq_lens"]
                B = int(seq_lens.shape[0])
                ep_coverages = []
            else:
                cu = _batch["cu_seqlens"]
                seq_lens = _batch["seq_lens"]
                B = int(seq_lens.shape[0])
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
                    sample,
                    seq_len=T_ep,
                    emb_id=emb_id,
                    ep_idx=b,
                )
                ep_coverages.append(coverage)
                ep_successes.append(float(coverage >= self.coverage_threshold))
                if frames and (self.max_videos is None or b < self.max_videos):
                    ep_frames_for_video.extend(frames)
                    if b < B - 1 and (
                        self.max_videos is None or b + 1 < self.max_videos
                    ):
                        H, W, _ = frames[0].shape
                        sep = np.zeros((5, H, W, 3), dtype=np.uint8)
                        ep_frames_for_video.extend(list(sep))

            mean_cov = float(np.mean(ep_coverages)) if ep_coverages else 0.0
            success_rate = float(np.mean(ep_successes)) if ep_successes else 0.0
            metrics[f"Valid/emb{emb_id}_sim_coverage"] = torch.tensor(
                mean_cov, device=device
            )
            metrics[f"Valid/emb{emb_id}_sim_success_rate"] = torch.tensor(
                success_rate, device=device
            )

            if ep_frames_for_video:
                images_dict[emb_id] = np.stack(ep_frames_for_video, axis=0)

        return metrics, images_dict
