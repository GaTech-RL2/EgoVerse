"""Permanent equality gate for dedup collapse c6.

Six evaluators carried byte-equivalent ``(C,H,W) float [0,1] -> (H,W,C) uint8``
converters under three private names. They were collapsed onto the single
``egomimic.eval.core.img_utils.img_chw_to_uint8``. This test pins that the
canonical fn reproduces EACH original body exactly (``np.array_equal``) on a
fixed random float tensor, and import-smokes every touched eval module so a
future edit that diverges any of them fails here.
"""

import importlib

import numpy as np
import torch

from egomimic.eval.core.img_utils import img_chw_to_uint8


# --- the original bodies, transcribed verbatim from pre-c6 sources ---

def _orig_4liner(img_chw):
    # video_rollout / pixel_video / spatial_video / vae_recon form
    x = img_chw.detach().cpu().float().numpy()
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)
    return np.transpose(x, (1, 2, 0))


def _orig_2liner(img_chw):
    # policy_action (_img_chw_to_uint8) / bundle_anchored (_u8) form
    x = np.clip(img_chw.detach().cpu().float().numpy(), 0.0, 1.0)
    return np.transpose((x * 255.0).astype(np.uint8), (1, 2, 0))


def test_canonical_matches_every_original_body():
    torch.manual_seed(0)
    # exercise the full clip range incl. out-of-[0,1] values
    t = torch.rand(3, 17, 11) * 1.4 - 0.2
    canon = img_chw_to_uint8(t)
    assert canon.dtype == np.uint8
    assert canon.shape == (17, 11, 3)
    for orig in (_orig_4liner, _orig_2liner):
        assert np.array_equal(canon, orig(t.clone()))


def test_touched_eval_modules_import():
    # COMBINE A: the pixel/spatial video-rollout modules were collapsed into
    # the family-agnostic ``eval_dfot_video_rollout`` (which re-exports them as
    # compat aliases). The converter all six shared still lives in img_utils.
    mods = [
        "egomimic.eval.core.img_utils",
        "egomimic.eval.core.eval_vae_recon",
        "egomimic.eval.dfot.eval_dfot_video_rollout",
        "egomimic.eval.dfot.eval_dfot_policy",
        "egomimic.eval.dfot.eval_dfot_bundle_anchored",
    ]
    for m in mods:
        importlib.import_module(m)
    # the collapsed families resolve as aliases of the unified eval
    uni = importlib.import_module("egomimic.eval.dfot.eval_dfot_video_rollout")
    assert uni.DFoTPixelVideoRolloutEval is uni.DFoTVideoRolloutEval
    assert uni.DFoTSpatialVideoRolloutEval is uni.DFoTVideoRolloutEval
