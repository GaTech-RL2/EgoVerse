"""Teacher-forced action OVERLAY eval for the PIXEL obs+action policy (D06).

The dense-signal yardstick that must exist before any model-level debugging:
replay a val episode with GT observation history through the pixel_policy
inference path (clean-context pinning + windowed joint denoise — the same
mechanism as ``DFoT._inference_step_pixel_policy``, but grounded on GT frames
instead of env renders), decode the predicted action from the action planes
(channel-masked spatial mean -> ``norm_stats`` unnormalize -> world px), and
score it against the GT action trajectory.

Mirrors the upstream pureDF evidence base (``df_pusht.py:145-182`` teacher-
forced ``test_step`` + ``df_pusht_overlay.py`` world-px RMSE / collapse check),
written in this repo's eval-class idiom (cf. ``eval_dfot_policy.py``).

Outputs per validation pass:
  * metrics — world-px action MSE / RMSE pooled over all replayed steps,
    pred-std collapse check (``overlay_pred_std_px`` == 0 => collapsed policy),
    GT std for reference, and the replayed step count.
  * artifacts — per-episode pred-vs-GT trajectory overlay plot (PNG) +
    per-step error arrays (NPZ) under ``video_dir()/epoch_N/``, and the same
    overlay panel as a short validation video via the ``EvalVideo`` buffer.

``action_history="pred"`` (default) feeds the model's own committed actions
back into the context — upstream-faithful (GT obs, autoregressive action).
``action_history="gt"`` is the fully teacher-forced variant.

K-chunk stacking (D07): when the outer stage carries ``action_chunk_k = K > 1``
(action planes = ``K*action_dim`` per keyframe, batch rows ``K*A`` wide at
keyframe stride K), each predicted keyframe decodes to a K-action chunk which
is unrolled in time order — metrics/plots score per-step world-px error over
the unrolled ``K*T`` action sequence, and the npz ``steps`` array holds
unrolled per-action indices (``keyframe*K + chunk_i``). ``K=1`` (or a stage
without the knob) is byte-identical to the legacy per-frame behavior.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from egomimic.eval.core.eval_video import EvalVideo
from egomimic.eval.dfot._base import DFoTVideoEvalMixin
from egomimic.models.diffusion.diffusion.discrete_diffusion import DiscreteDiffusion
from egomimic.models.diffusion.sampling import vanilla_schedule
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


_ACTION_HISTORY_MODES = ("pred", "gt")


# --------------------------------------------------------------------- #
# Pure decode/metric helpers — unit-testable without a model.
# --------------------------------------------------------------------- #


def actions_to_planes(actions: torch.Tensor, action_channels: int, h: int, w: int) -> torch.Tensor:
    """``(N, A) -> (N, C_a, H, W)`` broadcast planes. Mirrors
    ``PixelObsActionDFoTOuterStage._action_to_planes`` exactly (one action
    component per constant plane)."""
    n = actions.shape[0]
    a = actions[..., :action_channels]
    return a.reshape(n, action_channels, 1, 1).expand(n, action_channels, h, w)


def decode_action_planes(
    bundle: torch.Tensor,
    image_channels: int,
    action_channels: int,
    action_dim: int,
    chunk_k: int = 1,
) -> torch.Tensor:
    """``(..., C, H, W) -> (..., A)``: mask the action planes out of the joint
    bundle ON THE CHANNEL AXIS and spatial-mean each plane (prediction error
    averages out; a constant plane decodes exactly).

    The channel axis is third-from-last regardless of leading batch/time dims,
    so this is safe for (T,C,H,W) and (B,T,C,H,W) bundles alike — the D02
    failure mode (``[..., slice]`` hitting image WIDTH) cannot occur here.

    ``chunk_k > 1`` (D07 K-stacking): each frame's planes carry a stacked
    K-action chunk (``K*action_dim`` channels, time-major like upstream
    ``pusht_dataset.py``); the decode unstacks to ``(..., K, action_dim)``.
    ``chunk_k=1`` keeps the legacy ``(..., A)`` shape and values exactly.
    """
    planes = bundle[..., image_channels : image_channels + action_channels, :, :]
    pooled = planes.mean(dim=(-2, -1))
    if chunk_k <= 1:
        return pooled[..., :action_dim]
    return pooled[..., : chunk_k * action_dim].reshape(
        *pooled.shape[:-1], chunk_k, action_dim
    )


def overlay_metrics(pred_world: np.ndarray, gt_world: np.ndarray) -> Dict[str, "np.ndarray | np.floating"]:
    """World-px overlay scores for aligned (N, A) pred/GT action arrays.

    Returns per-step squared error plus the scalar MSE / RMSE and the
    collapse check: ``pred_std_px`` is the per-dim std across steps averaged
    over dims — 0 means the policy emits a constant action (collapsed).
    """
    if pred_world.shape != gt_world.shape:
        raise ValueError(
            f"pred/gt shape mismatch: {pred_world.shape} vs {gt_world.shape}"
        )
    per_step_sq = ((pred_world - gt_world) ** 2).mean(axis=-1)  # (N,)
    mse = float(per_step_sq.mean()) if per_step_sq.size else 0.0
    return {
        "per_step_sq_err": per_step_sq,
        "mse_px": np.float64(mse),
        "rmse_px": np.float64(np.sqrt(mse)),
        "pred_std_px": np.float64(pred_world.std(axis=0).mean()) if len(pred_world) else np.float64(0.0),
        "gt_std_px": np.float64(gt_world.std(axis=0).mean()) if len(gt_world) else np.float64(0.0),
    }


class DFoTPixelPolicyOverlayEval(DFoTVideoEvalMixin, EvalVideo):
    """GT-grounded replay overlay for ``pixel_mode="policy"`` outer stages."""

    def __init__(
        self,
        n_context_frames: int = 4,
        commit_k: int = 1,
        n_chunk_steps: int = 50,
        action_history: str = "pred",
        max_episodes: int = 2,
        max_replay_steps: int | None = None,
        embodiment_name: str = "pushshapes_sim",
        image_key: str = "front_img_1",
        limit_val_batches: int = 1,
        max_videos: int = 2,
        video_subdir: str = "videos_pixel_overlay",
        viz_func=None,
        transform_lists=None,
    ):
        super().__init__(
            limit_val_batches=limit_val_batches, viz_func=viz_func,
            transform_lists=transform_lists, max_videos=max_videos,
        )
        self.store_dfot_knobs(
            embodiment_name=embodiment_name, image_key=image_key,
            video_subdir=video_subdir, n_chunk_steps=n_chunk_steps,
        )
        self.n_context_frames = int(n_context_frames)
        self.commit_k = max(1, int(commit_k))
        self.action_history = str(action_history)
        if self.action_history not in _ACTION_HISTORY_MODES:
            raise ValueError(
                f"action_history must be one of {_ACTION_HISTORY_MODES}; "
                f"got {self.action_history!r}"
            )
        self.max_episodes = int(max_episodes)
        self.max_replay_steps = int(max_replay_steps) if max_replay_steps is not None else None

    # ------------------------------------------------------------------ #
    def _ddim_from_v(self, diff, x, v, cur, nxt):
        """One eta=0 DDIM step from a v-prediction. Mirrors
        ``DFoT._struct_ddim_step`` / ``DFoTPolicyActionEval._ddim_from_v``."""
        T = x.shape[1]
        pad = x.dim() - 2
        kBT = cur.clamp_min(0).long().unsqueeze(0)
        # Objective dispatch (mirrors DiscreteDiffusion.model_predictions).
        # Hardcoded pred_v here + a pred_x0-trained model negates x0 at
        # zero-terminal SNR (mirror-through-center bug, found 2026-06-10).
        obj = getattr(diff, "objective", "pred_v")
        if obj == "pred_x0":
            x0 = v
            eps = diff.predict_noise_from_start(x, kBT, v)
        elif obj == "pred_noise":
            eps = v
            x0 = diff.predict_start_from_noise(x, kBT, v)
        else:
            x0 = diff.predict_start_from_v(x, kBT, v)
            eps = diff.predict_noise_from_v(x, kBT, v)
        an = diff.alphas_cumprod[nxt.clamp_min(0).long()]
        an = torch.where(nxt < 0, torch.ones_like(an), an)
        an = an.reshape(1, T, *([1] * pad))
        c = (1.0 - an).clamp_min(0.0).sqrt()
        return x0 * an.sqrt() + eps * c

    @torch.no_grad()
    def _predict_window(self, algo, ctx_bundle: torch.Tensor, k: int, device) -> torch.Tensor:
        """Pin ``ctx_bundle`` (1, n_ctx, C, H, W) clean, denoise the next ``k``
        joint [RGB + action-plane] frames. Returns (k, C, H, W). Same windowed
        sampler as ``DFoT._inference_step_pixel_policy``."""
        diff = algo.diffusion
        n_ctx = ctx_bundle.shape[1]
        T = n_ctx + k
        C, H, W = ctx_bundle.shape[2:]
        dts = int(diff.timesteps) if isinstance(diff, DiscreteDiffusion) else None
        clean = -1 if dts is not None else 0.0
        sched = vanilla_schedule(self.n_chunk_steps, T, discrete_timesteps=dts).to(device).clone()
        sched[:, :n_ctx] = clean

        x = torch.randn(1, T, C, H, W, device=device)
        x[:, :n_ctx] = ctx_bundle
        for s in range(sched.shape[0] - 1):
            klev = sched[s].clamp_min(0).long().unsqueeze(0)
            v = algo.backbone(x, klev, external_cond=None)
            x = self._ddim_from_v(diff, x, v, sched[s], sched[s + 1])
            x[:, :n_ctx] = ctx_bundle
        return x[0, n_ctx:]

    # ------------------------------------------------------------------ #
    def _render_overlay_panel(
        self,
        pred_world: np.ndarray,
        gt_world: np.ndarray,
        per_step_sq: np.ndarray,
        title: str,
        png_path: str,
    ) -> np.ndarray:
        """Matplotlib pred-vs-GT trajectory + per-step error panel. Saves the
        PNG and returns the rendered RGB array for the validation video."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax_tr, ax_err) = plt.subplots(
            1, 2, figsize=(8.0, 4.0), dpi=96, constrained_layout=True
        )
        ax_tr.plot(gt_world[:, 0], gt_world[:, 1], "-o", color="green",
                   markersize=2, linewidth=1, label="GT")
        ax_tr.plot(pred_world[:, 0], pred_world[:, 1], "-o", color="darkorange",
                   markersize=2, linewidth=1, label="pred")
        ax_tr.set_title(title, fontsize=8)
        ax_tr.set_aspect("equal", adjustable="box")
        ax_tr.invert_yaxis()  # image-style y-down world coords
        ax_tr.legend(fontsize=7)
        ax_err.plot(np.sqrt(per_step_sq), color="crimson", linewidth=1)
        ax_err.set_title("per-step error (px)", fontsize=8)
        ax_err.set_xlabel("replay step", fontsize=7)
        fig.savefig(png_path)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        return np.ascontiguousarray(rgba[..., :3])

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compute_metrics_and_viz(self, batch) -> Tuple[Dict[str, torch.Tensor], Dict[int, np.ndarray]]:
        algo = self.model
        metrics: Dict[str, torch.Tensor] = {}
        images: Dict[int, np.ndarray] = {}
        emb_id = get_embodiment_id(self.embodiment_name)
        if emb_id not in batch:
            return metrics, images
        _batch = batch[emb_id]
        if self.image_key not in _batch:
            return metrics, images
        outer = algo.outer_stage
        if getattr(outer, "pixel_mode", None) != "policy":
            return metrics, images  # not the pixel obs+action policy
        device = self.trainer.lightning_module.device
        ac_key = algo.resolved_ac_keys[emb_id]
        imgs = _batch[self.image_key]
        is_packed = _batch.get("_packed", False)

        Ci = int(outer._image_channels)
        Ca = int(outer._action_channels)
        A = int(outer.action_dim)
        # D07 K-stacking: each (key)frame carries a stacked K-action chunk
        # (K*A plane channels, batch action rows are K*A wide). K=1 = legacy.
        K = max(1, int(getattr(outer, "action_chunk_k", 1)))
        n_ctx = max(1, self.n_context_frames)
        k = self.commit_k

        if is_packed:
            cu = _batch["cu_seqlens"].to(imgs.device, dtype=torch.long)
            ne = min(int(cu.shape[0] - 1), self.max_episodes)
            spans = [(int(cu[i].item()), int(cu[i + 1].item() - cu[i].item())) for i in range(ne)]
        else:
            ne = min(imgs.shape[0], self.max_episodes)
            spans = [(ep, imgs.shape[1]) for ep in range(ne)]

        epoch_dir = os.path.join(self.video_dir(), f"epoch_{self.trainer.current_epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        pooled_pred: List[np.ndarray] = []
        pooled_gt: List[np.ndarray] = []
        panel_frames: List[np.ndarray] = []
        for ep, (start, L) in enumerate(spans):
            if self.max_replay_steps is not None:
                L = min(L, n_ctx + self.max_replay_steps)
            if L < n_ctx + k:
                continue
            if is_packed:
                sl = slice(start, start + L)
                img_seq, act_seq = imgs[sl], _batch[ac_key][sl]
            else:
                img_seq, act_seq = imgs[start, :L], _batch[ac_key][start, :L]
            img_seq = img_seq.to(device).float()
            if img_seq.max() > 1.5:
                img_seq = img_seq / 255.0
            # D03 seam: pin GT frames in the model's image range.
            if str(getattr(outer, "image_range", "01")) == "pm1":
                img_seq = img_seq * 2.0 - 1.0
            act_seq = act_seq.to(device).float()  # normalized actions

            h, w = img_seq.shape[-2:]
            # action history fed into the context: GT, or overwritten by the
            # model's own committed predictions ("pred" = upstream-faithful).
            act_used = act_seq.clone()
            kf_steps: List[int] = []   # predicted (key)frame indices in the sequence
            pred_steps: List[int] = []  # unrolled per-action step indices (kf*K + i)
            pred_rows: List[torch.Tensor] = []  # one (K*A,) chunk row per keyframe
            for a in range(n_ctx, L - k + 1, k):
                ctx_rgb = img_seq[a - n_ctx : a]
                ctx_act = act_used[a - n_ctx : a]
                ctx_bundle = torch.cat(
                    [ctx_rgb, actions_to_planes(ctx_act, Ca, h, w)], dim=1
                ).unsqueeze(0)  # (1, n_ctx, Ci+Ca, H, W)
                pred = self._predict_window(algo, ctx_bundle, k, device)
                # (k, A) at K=1; (k, K, A) chunk unstack at K>1.
                pred_norm = decode_action_planes(pred, Ci, Ca, A, chunk_k=K)
                if self.action_history == "pred":
                    act_used[a : a + k, : K * A] = pred_norm.reshape(k, K * A)
                for j in range(k):
                    kf_steps.append(a + j)
                    pred_steps.extend((a + j) * K + i for i in range(K))
                    pred_rows.append(pred_norm[j].reshape(K * A))
            if not pred_rows:
                continue

            # Unroll keyframe chunks to per-action rows in time order: the
            # score is per-step world-px error over the K*T action sequence.
            # Unnormalizing the unrolled (N*K, A) rows applies the per-
            # underlying-dim norm_stats to every chunk slot (== tiling the
            # A-dim stats across K). K=1 reduces to the legacy (N, A) path.
            pred_norm_all = torch.stack(pred_rows, dim=0).reshape(-1, A)  # (N*K, A)
            gt_norm_all = act_seq[kf_steps, : K * A].reshape(-1, A)
            pred_world = algo.norm_stats.unnormalize({ac_key: pred_norm_all}, emb_id)[ac_key]
            gt_world = algo.norm_stats.unnormalize({ac_key: gt_norm_all}, emb_id)[ac_key]
            pred_world = pred_world.detach().cpu().numpy()
            gt_world = gt_world.detach().cpu().numpy()
            pooled_pred.append(pred_world)
            pooled_gt.append(gt_world)

            ep_m = overlay_metrics(pred_world, gt_world)
            np.savez(
                os.path.join(epoch_dir, f"overlay_emb{emb_id}_ep{ep}.npz"),
                pred_world=pred_world, gt_world=gt_world,
                steps=np.asarray(pred_steps, dtype=np.int64),
                per_step_sq_err=ep_m["per_step_sq_err"],
            )
            panel = self._render_overlay_panel(
                pred_world, gt_world, ep_m["per_step_sq_err"],
                title=(
                    f"ep{ep} rmse={float(ep_m['rmse_px']):.1f}px "
                    f"pred_std={float(ep_m['pred_std_px']):.1f} (0=COLLAPSED)"
                ),
                png_path=os.path.join(epoch_dir, f"overlay_emb{emb_id}_ep{ep}.png"),
            )
            panel_frames.extend([panel] * 30)  # ~1 s dwell per episode @30fps

        if pooled_pred:
            m = overlay_metrics(
                np.concatenate(pooled_pred, axis=0), np.concatenate(pooled_gt, axis=0)
            )
            metrics[f"Valid/emb{emb_id}_overlay_action_mse_px"] = torch.tensor(
                float(m["mse_px"]), device=device)
            metrics[f"Valid/emb{emb_id}_overlay_action_rmse_px"] = torch.tensor(
                float(m["rmse_px"]), device=device)
            metrics[f"Valid/emb{emb_id}_overlay_pred_std_px"] = torch.tensor(
                float(m["pred_std_px"]), device=device)
            metrics[f"Valid/emb{emb_id}_overlay_gt_std_px"] = torch.tensor(
                float(m["gt_std_px"]), device=device)
            metrics[f"Valid/emb{emb_id}_overlay_n_steps"] = torch.tensor(
                float(len(m["per_step_sq_err"])), device=device)
        if panel_frames:
            images[emb_id] = np.stack(panel_frames, axis=0)
        return metrics, images
