"""
PCATokenEval — PCA scatter visualisation of the **chunker output**
tokens (high-level / post-compression representation).

For each val episode we run the algo's teacher-forced ``forward_packed``
and capture two streams via forward hooks on the inner-most
``ComputeStage.main_network``:

  * ``inner_tokens``: ``(T_chunked, d_inner)`` — one token per chunk
    (i.e. one token per *committed* boundary). These are the
    high-level representations the model "thinks at" between chunker
    decisions; they only update when a new boundary fires.
  * ``boundary_mask`` from ``ctx.aux`` — used to map each video frame
    ``t`` to its containing chunk index (cumsum of mask up to ``t``).

PCA is fit on ALL chunker tokens across the val batch. Per video frame
the PCA scatter shows that episode's chunks visited so far (a trail
that only EXTENDS when a boundary fires; otherwise the "current point"
stays put). This is the right "PCA jumps on boundaries, not on every
frame" behaviour the earlier ``action_out``-hook version got wrong.

For policies without a chunker (e.g. FlatFusedPolicy) the hook gracefully
falls back to the input of ``policy.action_out`` (one token per frame)
so the panel still renders something, just frame-by-frame.

Output: ``(N_total, H, W, 3)`` where ``H``/``W`` is the figure size
(default 384×384). Length matches the other panels (one frame per real
timestep across all episodes, plus 5-frame separators).
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
        max_videos: int | None = None,
    ):
        super().__init__(
            limit_val_batches=limit_val_batches,
            viz_func=viz_func,
            transform_lists=transform_lists,
            max_videos=max_videos,
        )
        self.panel_h = int(panel_h)
        self.panel_w = int(panel_w)
        self.n_components = int(n_components)

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _capture_tokens(self, batch: Dict[int, Dict[str, Any]]) -> Dict[int, dict]:
        """For each emb returns a dict with:

          * ``inner_tokens``: ``(T_chunked, d_inner)`` — chunker output.
            Falls back to ``(T_total, d_model)`` (one per frame) when
            there's no chunker / no inner ComputeStage.
          * ``boundary_mask``: ``(T_total,) bool`` — per-frame chunker
            decisions, used to map frames to chunk indices. ``None`` for
            policies without a chunker.

        We hook ``ComputeStage.main_network`` forward to grab the
        chunker output activations, and read ``boundary_mask`` from
        ``ctx.aux`` via the same ``forward_packed`` call.
        """
        algo = self.model
        out: Dict[int, dict] = {}
        for emb_id, _batch in batch.items():
            if not _batch.get("_packed", False):
                continue
            policy = algo.nets["policy"]

            inner_module = self._find_inner_main_network(policy)
            captured_inner: list[torch.Tensor] = []
            captured_outer: list[torch.Tensor] = []

            def _hook_inner(_module, inputs, output):
                # ``main_network`` returns the chunker-space tokens
                # (T_chunked, d_inner) in packed mode.
                t = output.detach().float().cpu()
                if t.dim() == 3:
                    t = t.reshape(-1, t.shape[-1])
                captured_inner.append(t)

            def _hook_outer(_module, inputs):
                # ``action_out`` input: (T_total, d_model). Fallback only.
                t = inputs[0].detach().float().cpu()
                if t.dim() == 3:
                    t = t.reshape(-1, t.shape[-1])
                captured_outer.append(t)

            handles = []
            if inner_module is not None:
                handles.append(inner_module.register_forward_hook(_hook_inner))
            handles.append(policy.action_out.register_forward_pre_hook(_hook_outer))

            aux_list = []
            try:
                ac_key = algo.resolved_ac_keys[emb_id]
                obs = algo._build_obs(_batch, emb_id)
                actions = _batch[ac_key]
                cu = _batch["cu_seqlens"]
                max_seqlen = int(_batch["max_seq_len"])
                domain_name = getattr(algo, "domain_by_id", {}).get(emb_id)
                _pred, aux = policy.forward_packed(
                    actions, obs, cu, max_seqlen, embodiment_id=domain_name
                )
                aux_list = aux
            finally:
                for h in handles:
                    h.remove()

            # Pick inner if available; else fallback to outer per-frame.
            if captured_inner:
                tokens = captured_inner[-1]
                # Extract boundary_mask from the first chunker stage.
                bmask = None
                for entry in aux_list:
                    bp = entry.get("bpred") if isinstance(entry, dict) else None
                    if bp is None:
                        continue
                    bmask = bp.boundary_mask.detach().cpu().to(torch.bool)
                    break
                out[emb_id] = {
                    "inner_tokens": tokens.numpy(),
                    "boundary_mask": bmask.numpy() if bmask is not None else None,
                    "per_frame_fallback": False,
                }
            elif captured_outer:
                out[emb_id] = {
                    "inner_tokens": captured_outer[-1].numpy(),
                    "boundary_mask": None,
                    "per_frame_fallback": True,
                }
        return out

    @staticmethod
    def _find_inner_main_network(policy):
        """Walk ``policy.hnet.stages`` from the inside out to find the
        innermost ``ComputeStage``'s ``main_network``. Returns ``None``
        for flat policies / policies without a chunker.
        """
        hnet = getattr(policy, "hnet", None)
        if hnet is None:
            return None
        stages = getattr(hnet, "stages", None)
        if not stages:
            return None
        # Walk in reverse; first stage with a ``main_network`` attribute is
        # the innermost ComputeStage.
        for stage in reversed(stages):
            mn = getattr(stage, "main_network", None)
            if mn is not None:
                return mn
        return None

    # ------------------------------------------------------------------ #

    def compute_metrics_and_viz(
        self, batch: Dict[int, Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, np.ndarray]]:
        metrics: Dict[str, torch.Tensor] = {}
        images_dict: Dict[int, np.ndarray] = {}
        capture_by_emb = self._capture_tokens(batch)

        # ------------------------------------------------------------------ #
        # Pass 1: collect per-emb token streams + metadata. Then fit ONE PCA
        # on the union so all embodiments live in the same projected space —
        # required for cross-embodiment alignment viz (otherwise each emb
        # has its own basis and a "circle vs stick" comparison is
        # meaningless).
        # ------------------------------------------------------------------ #
        emb_records: Dict[int, dict] = {}
        token_chunks: List[np.ndarray] = []
        offsets: Dict[int, tuple[int, int]] = {}  # emb_id -> (start, end) in concat
        running = 0
        for emb_id, _batch in batch.items():
            captured = capture_by_emb.get(emb_id)
            if captured is None:
                continue
            tokens = captured["inner_tokens"]
            if tokens.size == 0:
                continue
            emb_records[emb_id] = {
                "tokens": tokens,
                "bmask": captured.get("boundary_mask"),
                "per_frame_fallback": captured.get("per_frame_fallback", False),
                "cu": _batch["cu_seqlens"],
                "seq_lens": _batch["seq_lens"],
            }
            token_chunks.append(tokens)
            offsets[emb_id] = (running, running + tokens.shape[0])
            running += tokens.shape[0]

        if not token_chunks:
            return metrics, images_dict

        # ONE PCA fit on the union of all embs' tokens.
        all_tokens = np.concatenate(token_chunks, axis=0)
        X_all, _ = _pca_fit_transform(all_tokens, k=self.n_components)
        # Shared xlim / ylim so all per-emb panels render in the same axes.
        x_min, x_max = X_all[:, 0].min(), X_all[:, 0].max()
        y_min, y_max = X_all[:, 1].min(), X_all[:, 1].max()
        pad_x = 0.05 * (x_max - x_min + 1e-9)
        pad_y = 0.05 * (y_max - y_min + 1e-9)
        xlim = (x_min - pad_x, x_max + pad_x)
        ylim = (y_min - pad_y, y_max + pad_y)

        for emb_id, _batch in batch.items():
            rec = emb_records.get(emb_id)
            if rec is None:
                continue
            tokens = rec["tokens"]
            bmask = rec["bmask"]
            per_frame_fallback = rec["per_frame_fallback"]
            cu = rec["cu"]
            seq_lens = rec["seq_lens"]
            B = int(seq_lens.shape[0])
            T_total = int(cu[-1].item())

            # This emb's slice of the shared projection.
            s, e = offsets[emb_id]
            X_proj = X_all[s:e]

            # Build a packed frame-index → chunk-index lookup. The chunker
            # output has one token per True in ``boundary_mask`` (cumsum
            # gives the running chunk index). The chunker stage commits a
            # chunk at each fire, so frame ``t`` belongs to chunk
            # ``cumsum(bmask)[t] - 1``.
            if not per_frame_fallback and bmask is not None:
                cs = np.cumsum(bmask.astype(np.int64))
                global_chunk_idx = np.maximum(cs - 1, 0)
            else:
                # Fallback: one token per frame.
                global_chunk_idx = np.arange(T_total)

            frames: List[np.ndarray] = []
            B_render = min(B, self.max_videos) if self.max_videos is not None else B
            for b in range(B_render):
                s = int(cu[b].item())
                e = int(cu[b + 1].item())
                ep_chunk_idx = global_chunk_idx[s:e]
                if ep_chunk_idx.size == 0:
                    continue
                ep_chunk_idx = np.clip(ep_chunk_idx, 0, tokens.shape[0] - 1)
                T_ep = ep_chunk_idx.shape[0]
                title_prefix = f"PCA ep={b}"
                for t in range(T_ep):
                    cur_idx = int(ep_chunk_idx[t])
                    # Trail: unique chunks visited up to t, in order.
                    seen, uniq = set(), []
                    for ci in ep_chunk_idx[: t + 1].tolist():
                        if ci not in seen:
                            seen.add(int(ci))
                            uniq.append(int(ci))
                    trail_xy = X_proj[np.array(uniq, dtype=np.int64)]
                    frame = _render_pca_frame(
                        fig_h=self.panel_h,
                        fig_w=self.panel_w,
                        global_xy=X_proj,
                        trail_xy=trail_xy,
                        current_xy=X_proj[cur_idx],
                        xlim=xlim,
                        ylim=ylim,
                        title=f"{title_prefix} t={t} chunk={cur_idx}",
                    )
                    frames.append(frame)
                if b < B - 1:
                    sep = np.zeros((5, self.panel_h, self.panel_w, 3), dtype=np.uint8)
                    frames.extend(list(sep))

            if frames:
                images_dict[emb_id] = np.stack(frames, axis=0)

            try:
                mean = tokens.mean(axis=0, keepdims=True)
                Xc = tokens - mean
                S = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
                total = float((S * S).sum())
                topk = float((S[: self.n_components] ** 2).sum())
                ratio = topk / max(total, 1e-9)
                metrics[f"Valid/emb{emb_id}_pca_top{self.n_components}_explained"] = (
                    torch.tensor(ratio, device=self.trainer.lightning_module.device)
                )
                metrics[f"Valid/emb{emb_id}_pca_n_tokens"] = torch.tensor(
                    float(tokens.shape[0]),
                    device=self.trainer.lightning_module.device,
                )
            except Exception:
                pass

        return metrics, images_dict
