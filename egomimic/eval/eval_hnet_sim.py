"""
Closed-loop sim eval for the H-Net pushshapes policy.

Wraps the ``Tsimulation.pushshapes.PushShapesEnv`` simulator. For each
validation sample, resets the env (init mode is config-driven — see
``init_config``), then steps the env one frame at a time using the model's
predicted action. Each step:

  1. Get fresh obs from the env (image + state).
  2. Encode obs and run a single transformer step through the H-Net.
  3. Use the predicted action to drive ``env.step``.
  4. Render the frame for the video buffer.

At episode end we record the env's ``_coverage()`` (the fraction of the
goal polygon covered by the object polygon) as the primary success
metric.

Compatible with both :class:`HNetPolicy` (HNet algo, stage-based) and
:class:`FlatFusedPolicy` (HNetFused algo) — the only requirements are
that the policy exposes a ``cond_encoder``, ``action_in``, ``action_out``,
``bos``, ``pos_emb``, and a ``hnet`` / ``backbone`` attribute with
``allocate_inference_cache(batch_size, max_seqlen, device, dtype)`` and
``step(x, params, cond=...)``. Each sub-class of the policy that
implements a custom AR generate path is handled via a small adapter
(see ``_AR_ADAPTERS``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from egomimic.eval.eval_video import EvalVideo
from egomimic.models.hnet_nets.context import HNetContext


def _state_to_init(state: np.ndarray) -> tuple:
    """Split a (5,) ``observations.state`` vector into env-init args.

    Mirrors ``zarr_writer``: state = concat(pusher_xy, object_xyangle).
    Returns ``(agent_pos, object_pose)``. ``goal_pose`` is a separate
    zarr key.
    """
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] < 5:
        raise ValueError(f"expected state of len >= 5, got {state.shape}")
    agent_pos = (float(state[0]), float(state[1]))
    object_pose = (float(state[2]), float(state[3]), float(state[4]))
    return agent_pos, object_pose


class HNetSimEval(EvalVideo):
    """Closed-loop sim rollout evaluator for H-Net on PushShapes.

    Args (set from the hydra yaml):
      env_kwargs: passed verbatim to ``PushShapesEnv(**env_kwargs)``.
      init_mode: ``"replay"`` (default) reads each val episode's frame-0
        state + goal from the batch; ``"random"`` calls ``env.reset(seed)``
        with a fresh seed; ``"seeds"`` cycles through ``init_seeds``.
      init_seeds: list[int] used when ``init_mode == "seeds"``.
      max_steps: optional per-episode step cap (defaults to the recorded
        episode length when in replay mode, else 200).
      coverage_threshold: fraction of goal coverage that counts as
        "success".
      video_fps: encoder fps for the saved videos.
      limit_val_batches: inherited from ``EvalVideo`` — caps how many
        validation batches are processed.
      viz_func: per-embodiment viz overlay — typically the same one used
        for teacher-forced eval, but we pass the sim-rolled-out
        trajectory and the sim's own rendered frames.
    """

    def __init__(
        self,
        env_kwargs: dict | None = None,
        init_mode: str = "replay",
        init_seeds: list[int] | None = None,
        max_steps: int | None = None,
        coverage_threshold: float = 0.7,
        video_fps: int = 30,
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
        self.init_mode = str(init_mode)
        if self.init_mode not in {"replay", "random", "seeds"}:
            raise ValueError(
                f"init_mode must be one of replay/random/seeds, got "
                f"{self.init_mode!r}"
            )
        self.init_seeds = list(init_seeds or [])
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.coverage_threshold = float(coverage_threshold)
        self.video_fps = int(video_fps)
        self._env = None  # lazily constructed; reused across episodes.
        self._init_counter = 0  # for seeds-mode cycling.

    # ------------------------------------------------------------------ #

    def _get_env(self):
        if self._env is None:
            # ``render_mode="rgb_array"`` lets us call ``env.render()`` for
            # the saved video without touching pygame's display loop.
            kwargs = dict(self.env_kwargs)
            kwargs.setdefault("render_mode", "rgb_array")
            from Tsimulation.pushshapes import PushShapesEnv

            self._env = PushShapesEnv(**kwargs)
        return self._env

    def _init_env(self, env, sample: dict, ep_seed_offset: int) -> None:
        """Reset + (optionally) set_state. Encapsulates init_mode."""
        if self.init_mode == "replay":
            # ``observations.state`` is shape (T, 5) per sample after the
            # dataset unpacks; pick frame 0. Same for ``goal_pose`` if it
            # was loaded (would require keymap entry — handled below).
            state_seq = sample.get("state_agent_obj")
            if state_seq is None:
                raise KeyError("init_mode='replay' requires 'state_agent_obj' in batch")
            frame0 = state_seq[0].detach().cpu().numpy()
            agent_pos, object_pose = _state_to_init(frame0)

            goal_seq = sample.get("goal_pose")
            goal_pose = None
            if goal_seq is not None:
                # goal_pose is constant within an episode; frame 0 is fine.
                goal_pose = tuple(
                    float(x)
                    for x in np.asarray(goal_seq[0].detach().cpu().numpy()).reshape(-1)[
                        :3
                    ]
                )

            # reset() randomizes the goal if not overridden; do reset first
            # then set_state to overwrite.
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
        """Roll out one sim episode using the model. Returns final coverage,
        the sim's rendered RGB frames, and the per-step predicted actions
        (unnormalized, world coords).
        """
        algo = self.model
        policy = algo.nets["policy"]
        device = self.trainer.lightning_module.device
        env = self._get_env()
        T = self.max_steps if self.max_steps is not None else int(seq_len)

        self._init_env(env, sample, ep_seed_offset=ep_idx)
        # ``action_horizon`` upper-bounds pos_emb length; cap T accordingly.
        # FlatFusedPolicy's pos_emb is 2*action_horizon long (interleaved
        # cond/action), so we cap at action_horizon either way and the
        # adapter halves it for the flat case if needed.
        T = min(T, policy.action_horizon)

        # Pick the right backbone path. HNetPolicy has ``policy.hnet``
        # (HNetCore stage tree, step takes (x, ctx)). FlatFusedPolicy has
        # ``policy.backbone`` (Isotropic stack, step takes (x, params, cond)).
        hnet_core = getattr(policy, "hnet", None)
        flat_backbone = getattr(policy, "backbone", None)
        is_flat = hnet_core is None and flat_backbone is not None
        backbone = hnet_core if hnet_core is not None else flat_backbone
        if backbone is None:
            raise RuntimeError("policy must expose ``hnet`` or ``backbone`` attribute")

        dtype = next(policy.parameters()).dtype
        # Flat policy has a 2T-token interleaved stream → cache sized 2T.
        cache_max_seqlen = (2 * T) if is_flat else T
        params = backbone.allocate_inference_cache(
            batch_size=1,
            max_seqlen=cache_max_seqlen,
            device=device,
            dtype=dtype,
        )

        # Initial token: BOS + pos_emb[0]. For flat the BOS sits at
        # position 1 of the interleaved stream (cond goes at position 0),
        # so we handle that inside the step loop instead.
        if not is_flat:
            cur = (policy.bos.expand(1, 1, policy.d_model) + policy.pos_emb[:, 0:1]).to(
                dtype
            )
        else:
            # Will set cur = bos + pos_emb[1] AFTER feeding the cond token
            # at pos 0 each step.
            cur = None

        frames: List[np.ndarray] = []
        actions_taken: List[np.ndarray] = []
        last_coverage = 0.0
        a_prev_emb = (
            (policy.bos.expand(1, 1, policy.d_model)).to(dtype) if is_flat else None
        )

        for t in range(T):
            # 1) Read current obs from the sim, format for the model.
            obs_env = env._get_obs()
            state_5 = np.concatenate(
                [obs_env["agent_pos"], obs_env["object_pose"]], axis=0
            ).astype(np.float32)  # (5,)
            image_chw = (
                np.transpose(obs_env["image"], (2, 0, 1)).astype(np.float32) / 255.0
            )  # (3, H, W)

            obs_norm = algo.norm_stats.normalize(
                {
                    "state_agent_obj": torch.from_numpy(state_5)
                    .unsqueeze(0)
                    .to(device),
                    "front_img_1": torch.from_numpy(image_chw).unsqueeze(0).to(device),
                },
                emb_id,
            )
            cond_dict_seq = policy.cond_encoder.encode(obs_norm, T_action=1)
            # cond[fused_cond] is (1, 1, d_cond). For step paths we want
            # the (1, d_cond) view (AdaLN) or (1, 1, d_cond) keep-as-is
            # (cross-attn unsqueezes internally). The block's step
            # path accepts the 2D form.
            cond_2d = {k: v.squeeze(1) for k, v in cond_dict_seq.items()}

            if is_flat:
                # FlatFusedPolicy AR semantics: at each outer step we run
                # TWO inner steps — cond token then action token. The
                # backbone is an Isotropic with no cond at the block level
                # (cond is fused via the input token), so step takes (x,
                # params) only.
                fused = cond_dict_seq["fused_cond"].squeeze(1)  # (1, d_cond)
                c_tok = (
                    policy.cond_in(fused).unsqueeze(1)
                    + policy.pos_emb[:, 2 * t : 2 * t + 1]
                ).to(dtype)
                _ = backbone.step(c_tok, params)
                a_tok = (a_prev_emb + policy.pos_emb[:, 2 * t + 1 : 2 * t + 2]).to(
                    dtype
                )
                h = backbone.step(a_tok, params)
            else:
                ctx = HNetContext(
                    cond_dict=cond_2d,
                    aux=[],
                    inference_params=params,
                )
                h = backbone.step(cur, ctx)

            a_t_norm = policy.action_out(h)  # (1, 1, action_dim) — normalized

            # 4) unnormalize and apply to sim.
            ac_key = algo.resolved_ac_keys[emb_id]
            a_t_world = (
                algo.norm_stats.unnormalize(
                    {ac_key: a_t_norm.squeeze(0)},
                    emb_id,  # squeeze batch dim
                )[ac_key]
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            action_xy = np.array(
                [float(a_t_world[0]), float(a_t_world[1])],
                dtype=np.float32,
            )

            _, _, terminated, _, info = env.step(action_xy)
            last_coverage = float(info.get("coverage", 0.0))
            actions_taken.append(action_xy)

            # 5) render the frame (use env's full-world surface for clarity;
            #    image_size could be tiny).
            frame = env.render()
            if frame is not None:
                frames.append(np.ascontiguousarray(frame))

            if terminated:
                break

            # 6) Prepare next-step input token.
            if is_flat:
                # Next outer step's a_prev = action_in(a_t_norm); the cond
                # for that step will be re-encoded from fresh sim obs.
                a_prev_emb = policy.action_in(a_t_norm).to(dtype)
            elif t + 1 < T:
                cur = (
                    policy.action_in(a_t_norm) + policy.pos_emb[:, t + 1 : t + 2]
                ).to(dtype)

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
            if not is_packed:
                # Padded mode is not used by current pushshapes setup.
                continue

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
                # Slice this sub-sequence's per-frame data.
                sample = {}
                for k, v in _batch.items():
                    if not torch.is_tensor(v):
                        continue
                    # Packed-tensor keys have first-dim = T_total.
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
                # Store frames for the video (one stack per episode, prefixed
                # with a 5-frame "title card" of black to separate eps).
                if frames:
                    ep_frames_for_video.extend(frames)
                    # Black separator between episodes.
                    if b < B - 1:
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
