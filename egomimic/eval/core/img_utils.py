"""Shared image conversion helpers for evaluators (DESIGN.md §2 ``eval/core``).

Hoisted in dedup collapse c6: six evaluators
(:mod:`eval.dfot.eval_dfot_video_rollout`,
:mod:`eval.dfot.eval_dfot_pixel_video_rollout`,
:mod:`eval.dfot.eval_dfot_spatial_video_rollout`,
:mod:`eval.dfot.eval_dfot_policy`,
:mod:`eval.dfot.eval_dfot_bundle_anchored`,
:mod:`eval.core.eval_vae_recon`) each carried a byte-for-byte equivalent
``(C, H, W)`` float-in-[0,1] -> ``(H, W, C)`` uint8-in-[0,255] converter under
three different private names (``_img_chw_to_uint8`` / ``_u8`` /
``_to_uint8_hwc``). They now all delegate here.

NOTE: ``eval.dfot.eval_dfot_self_rollout._img_chw_to_uint8`` is intentionally
NOT collapsed onto this — it uses a different ``x.max() <= 1.5`` auto-scale
heuristic and clips to [0, 255], so it is a genuinely distinct function.
"""

import numpy as np
import torch


def img_chw_to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    """``(C, H, W)`` float in [0, 1] -> ``(H, W, C)`` uint8 in [0, 255]."""
    x = img_chw.detach().cpu().float().numpy()
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)
    return np.transpose(x, (1, 2, 0))
