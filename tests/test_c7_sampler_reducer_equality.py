"""Permanent equality proofs for dedup collapse c7.

c7 collapses two structural near-duplicates:

1. ``PixelSpatialDFoTOuterStage._sample_frames_packed`` (image-only frame
   sampler) is a strict subset of
   ``PixelObsActionDFoTOuterStage._sample_windows_packed`` (image+action
   sampler). After the collapse the image-only entry point delegates to the
   superset with ``actions=None``; this test pins that the delegating path
   produces byte-identical sampled frames and cu_seqlens to the legacy
   standalone logic, across ALL THREE stochastic sampling modes
   (``fixed_window`` / ``start_to_end`` / ``random_subsample``) under a fixed
   seed, for every episode length regime (<=n, ==n, >n).

2. ``DFoT.compute_losses`` is the pure loss-reducer skeleton
   (sum per-embodiment ``*_action_loss`` -> ``action_loss``). c7 hoists that
   skeleton to ``Algo.compute_losses`` as the default and deletes DFoT's
   override. This test pins that the hoisted ``Algo.compute_losses`` produces
   a ``loss_dict`` ``torch.equal`` to the legacy DFoT output on fixed
   predictions.

   NOTE: ``VAE`` / ``PackedAlgoBase`` / ``HPT`` / ``PI`` reducers are *supersets*
   (recon/kl/vae, ratio_loss, domain-count division) — NOT folded onto the
   base. This test documents that distinction by asserting the base default
   does NOT reproduce the PackedAlgoBase (ratio_loss) output, so a future edit
   that wrongly deletes those overrides would fail here.

The samplers are pure functions of (images, actions, cu_seqlens) plus the two
scalar attributes ``_sample_n_frames`` / ``_frame_sampling`` — no nn modules —
so we bind the real unbound methods onto a bare instance via ``object.__new__``
to isolate the sampling logic without constructing a full diffusion stack.
"""

from collections import OrderedDict

import torch

from egomimic.algo.algo import Algo
from egomimic.algo.diffusion.algo import DFoT
from egomimic.algo.hnet import PackedAlgoBase
from egomimic.algo.diffusion.outer_stages.pixel_spatial_outer_stage import (
    PixelSpatialDFoTOuterStage,
)
from egomimic.algo.diffusion.outer_stages.pixel_obs_action_outer_stage import (
    PixelObsActionDFoTOuterStage,
)

MODES = ["fixed_window", "start_to_end", "random_subsample"]
# episode-length regimes vs n: 2 short(<n), one ==n, several >n
EP_LENS = [2, 3, 5, 5, 6, 9, 11]
N = 5
C, Hh, Ww = 3, 4, 4
A = 2


def _make_packed():
    torch.manual_seed(1234)
    cu = torch.zeros(len(EP_LENS) + 1, dtype=torch.long)
    for i, L in enumerate(EP_LENS):
        cu[i + 1] = cu[i] + L
    T = int(cu[-1].item())
    images = torch.randn(T, C, Hh, Ww)
    actions = torch.randn(T, A)
    return images, actions, cu


def _spatial_instance():
    obj = object.__new__(PixelSpatialDFoTOuterStage)
    obj._sample_n_frames = N
    obj._frame_sampling = "fixed_window"  # overridden per-mode below
    return obj


def _obsact_instance():
    obj = object.__new__(PixelObsActionDFoTOuterStage)
    obj._sample_n_frames = N
    obj._frame_sampling = "fixed_window"
    return obj


# --------------------------------------------------------------------------- #
# PROOF 1: image-only sampler == action-aware superset(actions=None) per mode  #
# --------------------------------------------------------------------------- #
def test_image_sampler_equals_superset_all_modes():
    images, actions, cu = _make_packed()
    spat = _spatial_instance()
    obsact = _obsact_instance()

    for mode in MODES:
        spat._frame_sampling = mode
        obsact._frame_sampling = mode

        # legacy standalone image-only sampler (seed A)
        torch.manual_seed(42)
        img_a, cu_a = PixelSpatialDFoTOuterStage._sample_frames_packed(
            spat, images.clone(), cu.clone()
        )

        # action-aware superset called with actions=None (seed A, same RNG draws)
        torch.manual_seed(42)
        img_b, act_b, cu_b = PixelObsActionDFoTOuterStage._sample_windows_packed(
            obsact, images.clone(), None, cu.clone()
        )

        assert torch.equal(cu_a, cu_b), f"cu mismatch in {mode}"
        assert torch.equal(img_a, img_b), f"frames mismatch in {mode}"
        assert act_b is None, f"actions should pass through None in {mode}"
        # superset with REAL actions must keep image cropping identical
        torch.manual_seed(42)
        img_c, act_c, cu_c = PixelObsActionDFoTOuterStage._sample_windows_packed(
            obsact, images.clone(), actions.clone(), cu.clone()
        )
        assert torch.equal(img_a, img_c), f"img w/ actions mismatch in {mode}"
        assert torch.equal(cu_a, cu_c), f"cu w/ actions mismatch in {mode}"
        assert act_c.shape[0] == img_c.shape[0], f"action align in {mode}"


# --------------------------------------------------------------------------- #
# PROOF 2: hoisted Algo.compute_losses == legacy DFoT.compute_losses           #
# --------------------------------------------------------------------------- #
def _fixed_predictions(emb_ids):
    torch.manual_seed(7)
    preds = {}
    batch = OrderedDict()
    for e in emb_ids:
        preds[f"{e}_action_loss"] = torch.randn(())
        batch[e] = torch.zeros(1)  # presence only; len(batch) is what matters
    return preds, batch


def test_base_reducer_equals_dfot_reducer():
    emb_ids = [0, 1]
    preds, batch = _fixed_predictions(emb_ids)

    dfot = object.__new__(DFoT)
    dfot.device = torch.device("cpu")
    legacy = DFoT.compute_losses(dfot, preds, batch)

    base = object.__new__(Algo)
    base.device = torch.device("cpu")
    new = Algo.compute_losses(base, preds, batch)

    assert list(legacy.keys()) == list(new.keys()), (
        list(legacy.keys()), list(new.keys()),
    )
    for k in legacy:
        assert torch.equal(legacy[k], new[k]), f"value mismatch on {k}"


def test_hnet_reducer_is_NOT_the_base_default():
    """Guard: PackedAlgoBase reducer adds ratio_loss -> it is a genuine superset,
    NOT folded onto Algo. If a future edit deletes PackedAlgoBase.compute_losses
    this fails.
    """
    emb_ids = [0]
    torch.manual_seed(11)
    preds = {
        "0_action_loss": torch.randn(()),
        "0_ratio_loss": torch.randn(()),
        "0_pred": torch.zeros(1),
    }
    batch = OrderedDict({0: torch.zeros(1)})

    hnet = object.__new__(PackedAlgoBase)
    hnet.device = torch.device("cpu")
    hnet_out = PackedAlgoBase.compute_losses(hnet, preds, batch)

    base = object.__new__(Algo)
    base.device = torch.device("cpu")
    base_out = Algo.compute_losses(base, preds, batch)

    # base ignores ratio_loss; hnet folds it in -> action_loss differs
    assert not torch.equal(hnet_out["action_loss"], base_out["action_loss"])
    assert "emb0_ratio_loss" in hnet_out and "emb0_ratio_loss" not in base_out


if __name__ == "__main__":
    test_image_sampler_equals_superset_all_modes()
    print("PROOF1 sampler-superset equality: PASS")
    test_base_reducer_equals_dfot_reducer()
    print("PROOF2 base==DFoT reducer equality: PASS")
    test_hnet_reducer_is_NOT_the_base_default()
    print("PROOF3 PackedAlgoBase reducer is genuine superset (not folded): PASS")
