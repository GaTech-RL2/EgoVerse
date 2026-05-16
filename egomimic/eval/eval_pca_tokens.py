"""
PCATokenEval — PCA scatter visualisation of the "highest stage" tokens.

For each val episode we run the algo's teacher-forced ``forward_packed``
and capture the per-token activations going into ``policy.action_out``
(via a forward pre-hook). These are the highest-level tokens in the
network — same shape as the predicted action sequence in token space,
``(T_total, d_model)`` packed.

PCA is fit on ALL tokens across the val batch (global 2D embedding).
Then per video frame ``t`` of each episode, we render a 2D scatter
with that episode's tokens up to time ``t`` drawn as a connected
"trail" so a viewer scrubbing through the video can see how the
representation evolves over time.

Output: ``(N_total, H, W, 3)`` where ``H``/``W`` is the figure size
(default 384×384 to match the env render). Length matches
``HNetEvalVideo`` (one frame per real timestep across all episodes,
plus 5-frame separators).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from egomimic.eval.eval_video import EvalVideo


def _pca_fit_transform(X: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Plain top-k PCA via SVD. Returns (projected, components).

    Centers ``X`` and returns ``(X - mean) @ V_k``, plus the components
    ``V_k`` (shape ``(d, k)``) for re-projection of other data.
    """
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    # ``full_matrices=False`` so VT has shape (min(N, d), d).
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V_k = Vt[:k].T  # (d, k)
    return Xc @ V_k, V_k


def _render_pca_frame(
    fig_h: int,
    fig_w: int,
    global_xy: np.ndarray,
    trail_xy: np.ndarray,
    current_xy: np.ndarray | None,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    title: str | None = None,
) -> np.ndarray:
    """Render one PCA scatter frame. Returns ``(H, W, 3)`` uint8."""
    dpi = 100
    fig, ax = plt.subplots(
        figsize=(fig_w / dpi, fig_h / dpi), dpi=dpi, facecolor="white"
    )
    # All val tokens — light grey background cloud.
    ax.scatter(global_xy[:, 0], global_xy[:, 1], s=2, c="#cccccc", alpha=0.4)
    # Trail — coloured by time.
    if trail_xy.shape[0] > 0:
        n = trail_xy.shape[0]
        colors = plt.cm.viridis(np.linspace(0.0, 1.0, n))
        ax.plot(trail_xy[:, 0], trail_xy[:, 1], color="#444444", lw=0.8, alpha=0.6)
        ax.scatter(trail_xy[:, 0], trail_xy[:, 1], s=8, c=colors, edgecolors="none")
    if current_xy is not None:
        ax.scatter(
            current_xy[0],
            current_xy[1],
            s=80,
            c="red",
            edgecolors="black",
            lw=1.5,
            marker="o",
            zorder=5,
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8)
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


class PCATokenEval(EvalVideo):
    """PCA scatter trail of the model's final-layer tokens.

    Args (yaml):
      panel_h, panel_w: pixel size of each scatter frame (default 384²).
      n_components: PCA dims (default 2; 3 not visualised here).
      limit_val_batches: inherited.
    """

    def __init__(
        self,
        panel_h: int = 384,
        panel_w: int = 384,
        n_components: int = 2,
        limit_val_batches: int = 4,
        viz_func: dict | None = None,
        transform_lists: dict | None = None,
    ):
        super().__init__(
            limit_val_batches=limit_val_batches,
            viz_func=viz_func,
            transform_lists=transform_lists,
        )
        self.panel_h = int(panel_h)
        self.panel_w = int(panel_w)
        self.n_components = int(n_components)

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _capture_tokens(
        self, batch: Dict[int, Dict[str, Any]]
    ) -> Dict[int, np.ndarray]:
        """Returns ``{emb_id: tokens (T_total, d_model)}`` captured from
        the input to ``policy.action_out``. Done via a forward pre-hook.
        """
        algo = self.model
        out: Dict[int, np.ndarray] = {}
        for emb_id, _batch in batch.items():
            if not _batch.get("_packed", False):
                continue
            policy = algo.nets["policy"]
            captured: list[torch.Tensor] = []

            def _hook(_module, inputs):
                # inputs is a tuple; first arg is the (T_total, d_model)
                # tensor that action_out projects to action_dim.
                captured.append(inputs[0].detach().float().cpu())

            handle = policy.action_out.register_forward_pre_hook(_hook)
            try:
                ac_key = algo.resolved_ac_keys[emb_id]
                obs = algo._build_obs(_batch, emb_id)
                actions = _batch[ac_key]
                cu = _batch["cu_seqlens"]
                max_seqlen = int(_batch["max_seq_len"])
                policy.forward_packed(actions, obs, cu, max_seqlen)
            finally:
                handle.remove()

            if not captured:
                continue
            tokens = captured[-1]  # (T_total, d_model)  — last call (handles
            # nested hook fires defensively).
            # FlatFusedPolicy may run action_out on a (B, T, d_model) tensor;
            # flatten if so.
            if tokens.dim() == 3:
                tokens = tokens.reshape(-1, tokens.shape[-1])
            out[emb_id] = tokens.numpy()
        return out

    # ------------------------------------------------------------------ #

    def compute_metrics_and_viz(
        self, batch: Dict[int, Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, np.ndarray]]:
        metrics: Dict[str, torch.Tensor] = {}
        images_dict: Dict[int, np.ndarray] = {}
        tokens_by_emb = self._capture_tokens(batch)

        for emb_id, _batch in batch.items():
            tokens = tokens_by_emb.get(emb_id)
            if tokens is None:
                continue
            cu = _batch["cu_seqlens"]
            seq_lens = _batch["seq_lens"]
            B = int(seq_lens.shape[0])
            T_total = tokens.shape[0]
            # Sanity: pack length should match.
            if int(cu[-1].item()) != T_total:
                # Shapes diverge — give up on this emb to avoid mis-indexing.
                continue

            # Global PCA fit on all tokens.
            X_proj, _ = _pca_fit_transform(tokens, k=self.n_components)
            # Compute axis limits once, with a 5% pad.
            x_min, x_max = X_proj[:, 0].min(), X_proj[:, 0].max()
            y_min, y_max = X_proj[:, 1].min(), X_proj[:, 1].max()
            pad_x = 0.05 * (x_max - x_min + 1e-9)
            pad_y = 0.05 * (y_max - y_min + 1e-9)
            xlim = (x_min - pad_x, x_max + pad_x)
            ylim = (y_min - pad_y, y_max + pad_y)

            # Per-frame: for each episode, for each timestep t, render the
            # trail up to and including t.
            frames: List[np.ndarray] = []
            for b in range(B):
                s = int(cu[b].item())
                e = int(cu[b + 1].item())
                T_ep = e - s
                ep_xy = X_proj[s:e]  # (T_ep, 2)
                for t in range(T_ep):
                    frame = _render_pca_frame(
                        fig_h=self.panel_h,
                        fig_w=self.panel_w,
                        global_xy=X_proj,
                        trail_xy=ep_xy[: t + 1],
                        current_xy=ep_xy[t],
                        xlim=xlim,
                        ylim=ylim,
                        title=f"PCA ep={b} t={t}",
                    )
                    frames.append(frame)
                if b < B - 1:
                    sep = np.zeros((5, self.panel_h, self.panel_w, 3), dtype=np.uint8)
                    frames.extend(list(sep))

            images_dict[emb_id] = np.stack(frames, axis=0)

            # Aggregate metric: total variance explained by top-k PCs.
            try:
                # Cheap: top-k singular values squared / total Frobenius.
                mean = tokens.mean(axis=0, keepdims=True)
                Xc = tokens - mean
                S = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
                total = float((S * S).sum())
                topk = float((S[: self.n_components] ** 2).sum())
                ratio = topk / max(total, 1e-9)
                metrics[f"Valid/emb{emb_id}_pca_top{self.n_components}_explained"] = (
                    torch.tensor(ratio, device=self.trainer.lightning_module.device)
                )
            except Exception:
                pass

        return metrics, images_dict
