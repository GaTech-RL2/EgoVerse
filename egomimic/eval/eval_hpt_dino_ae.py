"""Evaluator for the two-stage HPT-DINO action-expert algorithm.

Routes per-stage:
  - Stage 1: MSE + cosine similarity between predicted and GT EOS DINO,
    plus a "classic" DINO PCA visualization rendered side-by-side
    (GT | predicted) per sample. Works for both spatial patch-token
    embeddings ((B, N, D), N>1) and pooled CLS embeddings ((B, 1, D)).
  - Stage 2: action MSE / paired metrics and a trajectory overlay on
    front_img_t, delegating to ``algo.viz_func[embodiment_name]`` like
    the standard HPT eval.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import MeanSquaredError

from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment
from egomimic.utils.egomimicUtils import frechet_gaussian_over_time

_PCA_PNG_PER_STEP = 8  # how many per-batch samples to dump as standalone PNGs


def _pad_width(arr: np.ndarray, target_w: int) -> np.ndarray:
    """Right-pad a (B, H, W, 3) uint8 array along the width dim with zeros."""
    if arr.shape[2] >= target_w:
        return arr
    pad = np.zeros(
        (arr.shape[0], arr.shape[1], target_w - arr.shape[2], arr.shape[3]),
        dtype=arr.dtype,
    )
    return np.concatenate([arr, pad], axis=2)


def _pca_basis(features: torch.Tensor, n_components: int = 3):
    """Fit PCA basis on a (M, D) tensor. Returns (mean, components) of shape (1, D), (D, n_components).

    If the data has fewer than ``n_components`` non-trivial directions (e.g. M
    < n_components), the extra component slots are zero-padded.
    """
    M, D = features.shape
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    k = max(1, min(n_components, M, D))
    try:
        _, _, V = torch.pca_lowrank(centered, q=k, center=False)
        components = V[:, :k]
    except Exception:
        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        components = Vh[:k].T
    if components.shape[1] < n_components:
        pad = torch.zeros(
            D, n_components - components.shape[1], device=components.device
        )
        components = torch.cat([components, pad], dim=1)
    return mean, components


def _project_to_rgb(
    tokens: torch.Tensor, mean: torch.Tensor, components: torch.Tensor
) -> torch.Tensor:
    """Project (M, D) -> (M, 3) via PCA basis, then rescale per-image to [0,1]."""
    proj = (tokens - mean) @ components
    return proj


def _per_sample_normalize(x: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max normalize a (B, N, 3) tensor along N. Patch only."""
    lo = x.amin(dim=1, keepdim=True)
    hi = x.amax(dim=1, keepdim=True)
    return (x - lo) / (hi - lo + 1e-6)


def dino_pca_viz(
    pred: torch.Tensor,
    gt: torch.Tensor,
    target_size: int = 224,
) -> np.ndarray | None:
    """Side-by-side GT | predicted DINO PCA visualization (patch-token only).

    Args:
        pred, gt: tensors of shape (B, N, D) with ``N > 1``. CLS-only inputs
            (``N == 1``) carry no spatial structure and PCA-to-RGB on a single
            vector is degenerate, so this function returns ``None`` for them
            — the caller should rely on numeric metrics for CLS embeddings.
        target_size: each panel is upsampled to (target_size, target_size).

    Returns:
        np.uint8 array (B, target_size, 2*target_size, 3) on success, or
        ``None`` when ``N == 1``.
    """
    if pred.ndim == 2:
        pred = pred.unsqueeze(1)
    if gt.ndim == 2:
        gt = gt.unsqueeze(1)
    pred = pred.detach().float().cpu()
    gt = gt.detach().float().cpu()
    assert pred.shape == gt.shape, f"pred {pred.shape} vs gt {gt.shape}"

    B, N, D = gt.shape
    if N <= 1:
        return None

    fit = gt.reshape(-1, D)
    mean, components = _pca_basis(fit, n_components=3)

    pred_proj = _project_to_rgb(pred.reshape(-1, D), mean, components).reshape(B, N, 3)
    gt_proj = _project_to_rgb(gt.reshape(-1, D), mean, components).reshape(B, N, 3)
    pred_proj = _per_sample_normalize(pred_proj)
    gt_proj = _per_sample_normalize(gt_proj)

    side = int(round(N**0.5))
    if side * side != N:
        side = max(1, int(N**0.5))
        keep = side * side
        pred_proj = pred_proj[:, :keep]
        gt_proj = gt_proj[:, :keep]
    pred_grid = pred_proj.reshape(B, side, side, 3)
    gt_grid = gt_proj.reshape(B, side, side, 3)

    def _resize_nearest(grid: torch.Tensor) -> torch.Tensor:
        x = grid.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(target_size, target_size), mode="nearest")
        return x.permute(0, 2, 3, 1)

    pred_img = _resize_nearest(pred_grid)
    gt_img = _resize_nearest(gt_grid)
    side_by_side = torch.cat([gt_img, pred_img], dim=2)
    return (side_by_side.clamp(0, 1) * 255).to(torch.uint8).numpy()


class HPTDinoAEEvalVideo(EvalVideo):
    """Eval class for the two-stage HPT-DINO action-expert algorithm."""

    def compute_metrics_and_viz(self, batch):
        algo = self.model
        stage = algo.current_stage()
        # forward_eval already runs the combined (Stage 1 + Stage 2) inference
        # path during Stage 2 training, so a single forward call gives us
        # everything we need.
        preds = algo.forward_eval(batch)

        if stage == 1:
            return self._stage1_metrics_and_viz(batch, preds)
        return self._stage2_metrics_and_viz(batch, preds)

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Stage 1 writes PNGs to ``videos/epoch_<n>/<embodiment>/pca/`` and
        skips the video buffer entirely. Stage 2 falls through to the standard
        ``EvalVideo`` video-buffering path.
        """
        algo = self.model
        if algo.current_stage() != 1:
            super().on_validation_step(batch, batch_idx, dataloader_idx)
            return

        metrics, images_dict = self.compute_metrics_and_viz(batch)
        device = self.trainer.lightning_module.device
        metrics = {
            k: (v.to(device) if torch.is_tensor(v) else torch.tensor(v, device=device))
            for k, v in metrics.items()
        }
        epoch = self.trainer.current_epoch
        for emb_id, images in images_dict.items():
            outdir = os.path.join(
                self.video_dir(),
                f"epoch_{epoch}",
                str(get_embodiment(emb_id)),
                "pca",
            )
            os.makedirs(outdir, exist_ok=True)
            n = min(_PCA_PNG_PER_STEP, len(images))
            for i in range(n):
                img = images[i]  # uint8 HxWx3 RGB
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                path = os.path.join(outdir, f"step{batch_idx:04d}_sample{i:02d}.png")
                cv2.imwrite(path, bgr)
        self.trainer.lightning_module.log_dict(metrics, sync_dist=True)

    # ------------------------------------------------------------------
    # Stage 1: DINO prediction + PCA viz
    # ------------------------------------------------------------------

    def _stage1_metrics_and_viz(self, batch, preds):
        """Stage-1-only eval. Uses forward_eval predictions directly."""
        algo = self.model
        metrics: dict[str, torch.Tensor | float] = {}
        images_dict: dict = {}
        target_key = algo.shared_ac_key_stage1

        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            pk = f"{embodiment_name}_{target_key}"
            if pk not in preds or target_key not in _batch:
                continue
            self._add_dino_metrics_and_viz(
                metrics,
                images_dict,
                embodiment_id,
                embodiment_name,
                _batch,
                preds,
                target_key,
                pk,
            )
        return metrics, images_dict

    # ------------------------------------------------------------------
    # Stage 2: combined inference -> stage 1 PCA viz + stage 2 video viz
    # ------------------------------------------------------------------

    def _stage2_metrics_and_viz(self, batch, preds):
        """Stage-2 eval. Reports BOTH Stage-1 metrics/PCA-viz and Stage-2
        action metrics/trajectory video, since ``forward_eval`` runs the
        combined inference path during Stage 2 training.
        """
        algo = self.model
        metrics: dict[str, torch.Tensor | float] = {}
        images_dict: dict = {}
        mse_mod = MeanSquaredError()

        ac_key = algo.shared_ac_key_stage2
        dino_key = algo.shared_ac_key_stage1

        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()

            # ---- Stage-1 metrics from the combined-inference EOS DINO prediction.
            dino_pk = f"{embodiment_name}_{dino_key}"
            dino_image = None
            if dino_pk in preds and dino_key in _batch:
                dino_image = self._add_dino_metrics_and_viz(
                    metrics,
                    None,
                    embodiment_id,
                    embodiment_name,
                    _batch,
                    preds,
                    dino_key,
                    dino_pk,
                    metric_prefix="combined",
                )

            # ---- Stage-2 metrics: predicted actions vs GT.
            pk = f"{embodiment_name}_{ac_key}"
            if pk not in preds:
                if dino_image is not None:
                    images_dict[embodiment_id] = dino_image
                continue

            unb = algo.data_schematic.unnormalize_data(_batch, embodiment_id)
            ref = unb.get(ac_key)
            if ref is None:
                pieces = [unb[k] for k in algo.action_keys[embodiment_id] if k in unb]
                if not pieces:
                    if dino_image is not None:
                        images_dict[embodiment_id] = dino_image
                    continue
                ref = torch.cat(pieces, dim=-1)

            pred = preds[pk]
            T = ref.shape[1]
            D = ref.shape[-1]
            pred = pred[:, :T, :D]

            metrics[f"Valid/{pk}_paired_mse_avg"] = mse_mod(
                pred.cpu(), ref.cpu()
            ).item()
            metrics[f"Valid/{pk}_final_mse_avg"] = mse_mod(
                pred[:, -1].cpu(), ref[:, -1].cpu()
            ).item()
            try:
                fd = frechet_gaussian_over_time(pred, ref)
                metrics[f"Valid/{pk}_frechet_gauss_avg"] = fd.mean().item()
            except Exception:
                pass

            traj_image = self._stage2_viz(preds, unb)
            # Stack PCA viz (top) + trajectory viz (bottom) when both available
            # so the video frame carries both halves of the combined inference.
            if dino_image is not None:
                images_dict[embodiment_id] = self._stack_viz(dino_image, traj_image)
            else:
                images_dict[embodiment_id] = traj_image

        return metrics, images_dict

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_dino_metrics_and_viz(
        self,
        metrics: dict,
        images_dict: dict | None,
        embodiment_id,
        embodiment_name: str,
        batch: dict,
        preds: dict,
        target_key: str,
        pk: str,
        metric_prefix: str = "",
    ):
        algo = self.model
        unb = algo.data_schematic.unnormalize_data(batch, embodiment_id)
        gt = unb[target_key]
        pred = preds[pk]
        if gt.ndim == 2:
            gt = gt.unsqueeze(1)
        if pred.ndim == 2:
            pred = pred.unsqueeze(1)
        T = gt.shape[1]
        pred = pred[:, :T]

        prefix = f"{metric_prefix}_" if metric_prefix else ""
        mse_mod = MeanSquaredError()
        metrics[f"Valid/{pk}_{prefix}dino_mse"] = mse_mod(pred.cpu(), gt.cpu()).item()
        flat_pred = pred.reshape(pred.shape[0], -1)
        flat_gt = gt.reshape(gt.shape[0], -1)
        cos = F.cosine_similarity(flat_pred, flat_gt, dim=-1)
        metrics[f"Valid/{pk}_{prefix}dino_cosine_avg"] = cos.mean().item()
        metrics[f"Valid/{pk}_{prefix}dino_cosine_min"] = cos.min().item()

        # Retrieval-style discriminability: with B>1, compute pred[i] · gt[j]
        # and check whether the matching pair (i==j) has the highest similarity.
        # If predictions collapse to a single vector, top-1 retrieval drops to 0.
        if flat_pred.shape[0] > 1:
            pn = F.normalize(flat_pred, dim=-1)
            gn = F.normalize(flat_gt, dim=-1)
            sim = pn @ gn.T  # (B, B)
            B = sim.shape[0]
            argmax = sim.argmax(dim=1)
            top1 = (argmax == torch.arange(B, device=sim.device)).float().mean()
            diag = sim.diagonal()
            off = sim[~torch.eye(B, dtype=torch.bool, device=sim.device)]
            metrics[f"Valid/{pk}_{prefix}dino_retrieval_top1"] = top1.item()
            metrics[f"Valid/{pk}_{prefix}dino_diag_minus_off"] = (
                diag.mean() - off.mean()
            ).item()

        try:
            fd = frechet_gaussian_over_time(pred, gt)
            metrics[f"Valid/{pk}_{prefix}dino_frechet_avg"] = fd.mean().item()
        except Exception:
            pass

        viz = dino_pca_viz(pred, gt)  # None for CLS-only embeddings
        if images_dict is not None and viz is not None:
            images_dict[embodiment_id] = viz
        return viz

    @staticmethod
    def _stack_viz(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
        """Vertically stack two (B, H, W, 3) viz arrays. Pads widths to match."""
        if top.shape[0] != bottom.shape[0]:
            n = min(top.shape[0], bottom.shape[0])
            top, bottom = top[:n], bottom[:n]
        Wt, Wb = top.shape[2], bottom.shape[2]
        if Wt != Wb:
            target_w = max(Wt, Wb)
            top = _pad_width(top, target_w)
            bottom = _pad_width(bottom, target_w)
        return np.concatenate([top, bottom], axis=1)

    def _stage2_viz(self, predictions, batch):
        algo = self.model
        if algo.viz_func is None:
            # No viz function configured — emit a single black frame to keep the
            # video writer happy.
            B = next(iter(predictions.values())).shape[0]
            return np.zeros((B, 224, 224, 3), dtype=np.uint8)
        embodiment_id = batch["embodiment"][0].item()
        embodiment_name = get_embodiment(embodiment_id).lower()
        return algo.viz_func[embodiment_name](predictions, batch)
