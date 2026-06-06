"""DFoT self-rollout (world-model) evaluator.

For the joint-obs+action DFoT variant (``ObsActionDFoTOuterStage``), the
model denoises ``concat([state, action])`` jointly per step. The state
portion of the bundle IS the model's predicted future world state.
This evaluator turns that predicted-state trajectory into an actual
RENDERED video frame-by-frame by using the PushShapesEnv as a stateless
renderer:

  1. Pick the first ``max_videos`` episodes from the val batch.
  2. Take ONLY the t=0 image as the AdaLN cond (no GT state / action
     used at inference). Broadcast the cond across ``rollout_steps``
     tokens.
  3. Run chunk-mode DDIM sampling -> predicted bundle
     ``(rollout_steps, bundle_dim)``.
  4. Slice the state portion via ``outer_stage`` config:
       ``state = bundle[..., :sum(bundle_obs_dims)]``
     and unnormalize per obs key to recover world-frame state.
  5. For each step, ``env.set_state(agent_pos, object_pose)`` and
     ``env.render()`` to produce a frame. Stack -> mp4.

The first 5 dims of the bundle's obs portion are
``(agent_x, agent_y, obj_x, obj_y, obj_theta)`` (the pushshapes
state layout) — same split used by ``eval_sim._state_to_init``. If the
bundle obs is shorter than 5 dims, the env can't be rendered and the
eval falls back to the path-overlay viz from the previous version.

The starting frame (rendered from the GT initial state) is prepended
to each video so viewers can see the launch condition before the model
takes over.
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
import torch

from egomimic.eval.core.eval_sim import _state_to_init
from egomimic.eval.core.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


def _img_chw_to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    x = img_chw.detach().cpu().float().numpy()
    if x.max() <= 1.5:
        x = x * 255.0
    return x.clip(0, 255).astype(np.uint8)


def _path_overlay_frame(
    img_chw_uint8: np.ndarray,
    path_xy: np.ndarray,
    cur_idx: int,
    world_size: float,
    upscale_to: int = 512,
) -> np.ndarray:
    """Fallback viz when the env renderer is unavailable: trail overlay
    on the starting image. Same as the previous implementation."""
    img_hwc = np.transpose(img_chw_uint8, (1, 2, 0))
    img_hwc = cv2.resize(
        img_hwc, (upscale_to, upscale_to), interpolation=cv2.INTER_NEAREST
    )
    H, W, _ = img_hwc.shape
    out = np.ascontiguousarray(img_hwc.copy())
    sx = W / float(world_size)
    sy = H / float(world_size)
    pts = []
    for p in path_xy[: cur_idx + 1]:
        cx = int(round(float(p[0]) * sx))
        cy = int(round(float(p[1]) * sy))
        if 0 <= cx < W and 0 <= cy < H:
            pts.append((cx, cy))
    for i in range(1, len(pts)):
        cv2.line(out, pts[i - 1], pts[i], (180, 220, 255), thickness=2)
    if pts:
        cv2.circle(out, pts[-1], 8, (0, 0, 0), thickness=-1)
        cv2.circle(out, pts[-1], 6, (255, 255, 255), thickness=-1)
    return out


class DFoTSelfRolloutEval(EvalVideo):
    """World-model self-rollout for joint obs+action DFoT.

    Args:
        rollout_steps: number of bundle tokens to generate from the
            single starting image. mp4 length = rollout_steps (+ one
            launch-condition frame at the start).
        world_size: physics world extent for pixel scaling (PushShapes
            uses 512).
        image_key: which obs key carries the env image.
        agent_state_slice: slice into the predicted state vector that
            holds (agent_x, agent_y). Default ``[0, 2]``.
        env_kwargs: forwarded to ``PushShapesEnv(**)`` for rendering
            (object_shape, pusher_shape, obstacle_level, image_size, ...).
            ``render_mode`` defaults to ``"rgb_array"``.
        cfg_scale: CFG scale for sampling. 1.0 = off.
        upscale_to: per-frame side after upscaling (env's image_size is
            usually 96; we upscale for clarity).
        max_videos: per-pass cap on rendered episodes.
        limit_val_batches: per-pass cap on val batches.
        embodiment_name: which embodiment the packed batch carries.
        viz_func / transform_lists: forwarded to ``EvalVideo`` base.
    """

    def __init__(
        self,
        rollout_steps: int = 64,
        world_size: float = 512.0,
        image_key: str = "front_img_1",
        agent_state_slice: slice | tuple = (0, 2),
        env_kwargs: dict | None = None,
        cfg_scale: float = 1.0,
        upscale_to: int = 512,
        max_videos: int = 2,
        limit_val_batches: int = 4,
        embodiment_name: str = "pushshapes_sim",
        video_subdir: str = "videos_self_rollout",
        viz_func=None,
        transform_lists=None,
    ):
        super().__init__(
            limit_val_batches=limit_val_batches,
            viz_func=viz_func,
            transform_lists=transform_lists,
            max_videos=max_videos,
        )
        # Per-eval output namespace so DFoTValEval and DFoTSelfRolloutEval
        # can be run side-by-side at the top of an EvalList without
        # overwriting each other's validation_video_*.mp4 files.
        self._video_subdir = str(video_subdir)
        self.rollout_steps = int(rollout_steps)
        self.world_size = float(world_size)
        self.image_key = image_key
        if isinstance(agent_state_slice, slice):
            self.agent_state_slice = agent_state_slice
        else:
            start, stop = agent_state_slice
            self.agent_state_slice = slice(int(start), int(stop))
        self.env_kwargs = dict(env_kwargs or {})
        self.cfg_scale = float(cfg_scale)
        self.upscale_to = int(upscale_to)
        self.embodiment_name = embodiment_name
        self._env = None  # lazy-init on first call

    # Override the base ``EvalVideo.video_dir`` so we don't collide with
    # DFoTValEval's mp4 filenames at val time.
    def video_dir(self):
        import os

        return os.path.join(self.root_dir(), self._video_subdir)

    # ------------------------------------------------------------------
    # Env: lazy PushShapesEnv for state-driven rendering. No physics
    # step — we use set_state + render() only.
    # ------------------------------------------------------------------

    def _get_env(self):
        if self._env is not None:
            return self._env
        try:
            from Tsimulation.pushshapes import PushShapesEnv
        except Exception:
            return None
        kw = dict(self.env_kwargs)
        kw.setdefault("render_mode", "rgb_array")
        self._env = PushShapesEnv(**kw)
        return self._env

    # ------------------------------------------------------------------
    # Bundle rollout from one starting image.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _rollout_from_image(
        self, algo, img_chw_norm: torch.Tensor, emb_id: int
    ) -> torch.Tensor:
        device = img_chw_norm.device
        T = self.rollout_steps
        # CondEncoderModule's image branch broadcasts (B, C, H, W) over
        # T_action; a 5-D tensor is taken as already-temporal and skips
        # the broadcast, which mismatches the backbone at T=rollout_steps.
        img_bchw = img_chw_norm.unsqueeze(0)
        obs_for_cond = {self.image_key: img_bchw}
        cond_dict = algo.cond_encoder.encode(obs_for_cond, T_action=T)
        cond = cond_dict.get(algo.cond_output_key)
        if cond is None:
            raise RuntimeError(
                "cond_encoder produced no fused_cond — DFoTSelfRolloutEval "
                "requires image conditioning."
            )
        bundle = algo._sample_chunk(B=1, T=T, cond=cond, device=device)
        return bundle.squeeze(0)  # (T, bundle_dim)

    # ------------------------------------------------------------------
    # Frame renderer: per-step set_state -> render -> resize.
    # ------------------------------------------------------------------

    def _render_states_to_frames(
        self,
        state_world_seq: np.ndarray,  # (T, state_dim) — agent_xy + obj_xytheta
        prepend_state: np.ndarray | None = None,
    ) -> List[np.ndarray] | None:
        env = self._get_env()
        if env is None:
            return None
        if not hasattr(env, "set_state"):
            return None

        frames: List[np.ndarray] = []
        # reset() must happen once before set_state(); we only need this
        # once per episode rollout since set_state overrides everything.
        try:
            env.reset()
        except Exception:
            return None

        def _render_state(s_vec: np.ndarray) -> np.ndarray | None:
            if s_vec.shape[0] < 5:
                return None
            agent_pos, obj_pose = _state_to_init(s_vec)
            env.set_state(agent_pos=agent_pos, object_pose=obj_pose)
            rgb = env.render()
            if rgb is None:
                return None
            if rgb.shape[0] != self.upscale_to:
                rgb = cv2.resize(
                    rgb,
                    (self.upscale_to, self.upscale_to),
                    interpolation=cv2.INTER_NEAREST,
                )
            return np.ascontiguousarray(rgb)

        if prepend_state is not None:
            f0 = _render_state(prepend_state)
            if f0 is not None:
                frames.append(f0)

        for t in range(state_world_seq.shape[0]):
            f = _render_state(state_world_seq[t])
            if f is None:
                return None
            frames.append(f)
        return frames

    # ------------------------------------------------------------------
    # Main entrypoint.
    # ------------------------------------------------------------------

    def compute_metrics_and_viz(self, batch):
        algo = self.model
        metrics: Dict[str, torch.Tensor] = {}
        images_dict: Dict[int, np.ndarray] = {}

        emb_id = get_embodiment_id(self.embodiment_name)
        if emb_id not in batch:
            return metrics, images_dict
        _batch = batch[emb_id]
        ac_key = algo.resolved_ac_keys[emb_id]

        if self.image_key not in _batch:
            return metrics, images_dict
        imgs = _batch[self.image_key]

        outer = algo.outer_stage
        action_slice = getattr(outer, "action_slice", slice(0, algo.action_dim))
        obs_end = action_slice.start or 0
        has_obs_in_bundle = obs_end > 0
        obs_keys = list(getattr(outer, "bundle_obs_keys", []))
        obs_dims = list(getattr(outer, "bundle_obs_dims", []))

        # Pick starting images + (optionally) GT starting states.
        if not _batch.get("_packed", False):
            B = imgs.shape[0]
            n = min(B, self.max_videos or B)
            start_imgs = [imgs[i, 0] for i in range(n)]
            gt_starts = []
            for i in range(n):
                if obs_keys and obs_keys[0] in _batch:
                    gt_starts.append(_batch[obs_keys[0]][i, 0])
                else:
                    gt_starts.append(None)
        else:
            cu = _batch["cu_seqlens"].to(imgs.device, dtype=torch.long)
            n = min(int(cu.shape[0] - 1), self.max_videos or 99999)
            start_imgs = [imgs[int(cu[i].item())] for i in range(n)]
            gt_starts = []
            for i in range(n):
                if obs_keys and obs_keys[0] in _batch:
                    gt_starts.append(_batch[obs_keys[0]][int(cu[i].item())])
                else:
                    gt_starts.append(None)

        frames_for_emb: List[np.ndarray] = []
        for img_chw, gt_start in zip(start_imgs, gt_starts):
            bundle_pred = self._rollout_from_image(algo, img_chw, emb_id)

            if has_obs_in_bundle and obs_keys:
                obs_pred = bundle_pred[..., :obs_end]
                splits = list(torch.split(obs_pred, obs_dims, dim=-1))
                obs_unnorm_dict = {k: v for k, v in zip(obs_keys, splits)}
                obs_unnorm = algo.norm_stats.unnormalize(obs_unnorm_dict, emb_id)
                state_world = obs_unnorm[obs_keys[0]]  # (T, dim_0)

                # Prepend GT initial state so the video starts from the
                # actual launch condition (rendered).
                gt_start_world = None
                if gt_start is not None:
                    gt_start_norm = {obs_keys[0]: gt_start.unsqueeze(0)}
                    gt_start_world_t = algo.norm_stats.unnormalize(
                        gt_start_norm, emb_id
                    )[obs_keys[0]].squeeze(0)
                    gt_start_world = gt_start_world_t.detach().cpu().numpy()

                state_world_np = state_world.detach().cpu().numpy()
                rendered = self._render_states_to_frames(
                    state_world_np,
                    prepend_state=gt_start_world,
                )
                if rendered is not None:
                    frames_for_emb.extend(rendered)
                    continue
                # Renderer unavailable -> fall back to overlay.
                xy = state_world[..., self.agent_state_slice]
                xy_np = xy.detach().cpu().numpy()
                img_uint8 = _img_chw_to_uint8(img_chw)
                for t in range(self.rollout_steps):
                    frames_for_emb.append(
                        _path_overlay_frame(
                            img_uint8,
                            xy_np,
                            t,
                            self.world_size,
                            self.upscale_to,
                        )
                    )
            else:
                # Vanilla DFoT (bundle = action only) — no state to render.
                act_pred = bundle_pred[..., action_slice]
                act_unnorm = algo.norm_stats.unnormalize({ac_key: act_pred}, emb_id)[
                    ac_key
                ]
                xy = act_unnorm[..., :2]
                xy_np = xy.detach().cpu().numpy()
                img_uint8 = _img_chw_to_uint8(img_chw)
                for t in range(self.rollout_steps):
                    frames_for_emb.append(
                        _path_overlay_frame(
                            img_uint8,
                            xy_np,
                            t,
                            self.world_size,
                            self.upscale_to,
                        )
                    )

        if frames_for_emb:
            images_dict[emb_id] = np.stack(frames_for_emb, axis=0)
        return metrics, images_dict
