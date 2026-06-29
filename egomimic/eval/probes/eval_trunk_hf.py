"""TrunkHFEval — per-embodiment high-frequency diagnostic for the SHARED trunk.

For each embodiment, logs the per-token first-difference L2 norm (high-frequency
content) of the TRUNK reconstruction (``up`` from msublatent_up) vs the linear
RESIDUAL (``residual_scale * residual_proj(x)``), plus their ratio.

Reading:
  * ``hf_ratio = HFtrunk / HFresid``.
  * ``> ~1``  -> the trunk carries the per-token detail (emb is living on the trunk).
  * ``< 1``   -> the residual carries the per-token detail = the emb is BYPASSING
                the trunk (trunk is only a smooth baseline).

Why HF and not magnitude: a heavily under-segmented chunk produces a high-MAGNITUDE
but SMOOTH trunk reconstruction (one chunk spread over hundreds of tokens), so raw
norm hides the bypass; the first-difference isolates information-bearing per-token
variation. Forward-only (teacher-forced) -> no DECHUNK-AR needed.

Watch the two embodiments' hf_ratio CONVERGE across training; divergence is the
shared-trunk capture (one emb grabs the trunk, the other bypasses via residual).
See vault: 30_Projects/RL2/Transformer HNet/Findings.md.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from egomimic.eval.core.eval_video import EvalVideo


def _hf(x: torch.Tensor, cu: torch.Tensor) -> float:
    """Mean per-token first-difference L2 norm WITHIN sequences (skips the
    cross-sequence jumps at cu_seqlens starts)."""
    if x.shape[0] < 2:
        return float("nan")
    d = (x[1:] - x[:-1]).norm(dim=-1)  # (T-1,)
    mask = torch.ones_like(d, dtype=torch.bool)
    starts = cu[1:-1].long() - 1
    starts = starts[(starts >= 0) & (starts < d.shape[0])]
    mask[starts] = False
    return d[mask].mean().item() if bool(mask.any()) else float("nan")


class TrunkHFEval(EvalVideo):
    """Logs per-emb {trunk_hf, resid_hf, hf_ratio}. No viz."""

    def __init__(
        self,
        limit_val_batches: int = 2,
        viz_func: dict | None = None,
        transform_lists: dict | None = None,
        max_videos: int | None = 0,
    ):
        super().__init__(
            limit_val_batches=limit_val_batches,
            viz_func=viz_func,
            transform_lists=transform_lists,
            max_videos=max_videos,
        )

    @staticmethod
    def _is_chunker(m) -> bool:
        return (
            hasattr(m, "residual_proj")
            and hasattr(m, "residual_scale")
            and hasattr(m, "msublatent_up")
        )

    @torch.no_grad()
    def compute_metrics_and_viz(
        self, batch: Dict[int, Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, np.ndarray]]:
        metrics: Dict[str, torch.Tensor] = {}
        algo = self.model
        root = getattr(algo, "nets", None)
        if root is None or not hasattr(root, "named_modules"):
            root = getattr(algo, "policy", None)
        if root is None:
            return metrics, {}
        stages = [(n, m) for n, m in root.named_modules() if self._is_chunker(m)]
        if not stages:
            return metrics, {}

        cap: Dict[str, torch.Tensor] = {}

        def mk(key):
            def h(mod, i, o):
                cap[key] = (
                    (o[0] if isinstance(o, (tuple, list)) else o).detach().float()
                )

            return h

        handles = []
        for n, m in stages:
            handles.append(m.msublatent_up.register_forward_hook(mk(n + "::up")))
            handles.append(m.residual_proj.register_forward_hook(mk(n + "::res")))
        try:
            for emb_id, _batch in batch.items():
                if not isinstance(_batch, dict) or not _batch.get("_packed", False):
                    continue
                ac_key = algo.resolved_ac_keys[emb_id]
                obs = algo._build_obs(_batch, emb_id)
                actions = _batch[ac_key]
                cu = _batch["cu_seqlens"]
                ms = int(_batch["max_seq_len"])
                dom = getattr(algo, "domain_by_id", {}).get(emb_id)
                cap.clear()
                algo.policy.forward_packed(actions, obs, cu, ms, embodiment_id=dom)
                cu_t = cu if torch.is_tensor(cu) else torch.as_tensor(cu)
                for n, m in stages:
                    up = cap.get(n + "::up")
                    res = cap.get(n + "::res")
                    if up is None or res is None:
                        continue  # this sub_stage didn't run for this emb
                    sc = (
                        float(m.residual_scale.detach())
                        if torch.is_tensor(m.residual_scale)
                        else float(m.residual_scale)
                    )
                    res = sc * res
                    cu_d = cu_t.to(up.device)
                    hft, hfr = _hf(up, cu_d), _hf(res, cu_d)
                    metrics[f"Valid/emb{emb_id}_trunk_hf"] = torch.tensor(hft)
                    metrics[f"Valid/emb{emb_id}_resid_hf"] = torch.tensor(hfr)
                    metrics[f"Valid/emb{emb_id}_hf_ratio"] = torch.tensor(
                        hft / (hfr + 1e-9)
                    )
        finally:
            for h in handles:
                h.remove()
        return metrics, {}
