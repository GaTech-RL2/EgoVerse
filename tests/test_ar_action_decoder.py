"""Contract tests for the causal / bidirectional arc-token decoders.

The experiment these nodes serve is only meaningful if the masks do what they
claim. A causal decoder that can see the row it is predicting will train to a
beautiful loss and roll out to nothing, and the failure is silent — free
running at rollout simply produces a different distribution than training did.
So the mask, not the loss, is what is asserted here.
"""

from __future__ import annotations

import pytest
import torch

from egomimic.pipeline.stages_ar import ARActionDecoder
from egomimic.pipeline.stages_sampler import NativeActionMSELoss

HORIZON, N_WAYPOINTS, ACTION_DIM, COND_DIM = 17, 16, 5, 67
DOMAIN = "pushshapes_sim_gripper"
CAUSAL_VARIANTS = ("state_action_ar", "state_idm")


def _decoder(variant: str, **kw) -> ARActionDecoder:
    params = dict(
        condition_input_dim=COND_DIM,
        action_horizon=HORIZON,
        action_dims={DOMAIN: ACTION_DIM},
        variant=variant,
        d_model=64,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        n_waypoints=N_WAYPOINTS,
        gradient_checkpointing=False,
    )
    params.update(kw)
    return ARActionDecoder(**params).eval()


def _batch(batch_size: int = 3, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    return {
        "condition": torch.randn(batch_size, COND_DIM),
        "embodiment": DOMAIN,
        "target": torch.randn(batch_size, HORIZON, ACTION_DIM),
    }


@pytest.mark.parametrize("variant", CAUSAL_VARIANTS)
def test_causal_rows_do_not_see_their_own_or_later_targets(variant):
    """Perturbing target row k must leave the PATH at rows <= k untouched.

    Row k's pose is allowed to depend on rows < k only. Because position k
    consumes row k-1, changing row k may move rows k+1 onward — but never row
    k itself, which is what "causal" buys.

    Pose channels are the strict test for both arms. `state_idm`'s action
    channels are deliberately excluded: it defines a_m as a readout of the
    p_m -> p_{m+1} transition, so a_m depends on p_{m+1} and therefore on the
    row fed in at position m. That is the arm's premise, not a leak — at
    rollout p_{m+1} is the model's own output and no ground truth is consulted.
    Asserting otherwise here would be asserting the arm out of existence.
    """
    decoder = _decoder(variant)
    batch = _batch()
    n_pose = decoder.pose_channels if decoder.uses_idm else ACTION_DIM
    with torch.no_grad():
        base = decoder(dict(batch))["pred_action"]

    for k in range(HORIZON):
        perturbed = batch["target"].clone()
        perturbed[:, k] += 10.0
        with torch.no_grad():
            other = decoder(dict(batch, target=perturbed))["pred_action"]
        torch.testing.assert_close(
            other[:, : k + 1, :n_pose],
            base[:, : k + 1, :n_pose],
            msg=f"{variant}: target row {k} leaked into predictions <= {k}",
        )


def test_state_idm_is_never_shown_action_channels():
    """The arm's claim is that the path is the hard part.

    If action history reached the backbone, `state_idm` could lean on it as a
    shortcut and would no longer be testing inverse dynamics at all.
    """
    decoder = _decoder("state_idm")
    assert decoder.feeds_back_actions is False
    batch = _batch()
    scrambled = batch["target"].clone()
    scrambled[..., decoder.pose_channels :] += 100.0
    with torch.no_grad():
        base = decoder(dict(batch))["pred_action"]
        other = decoder(dict(batch, target=scrambled))["pred_action"]
    torch.testing.assert_close(base, other)


@pytest.mark.parametrize("variant", CAUSAL_VARIANTS)
def test_causal_later_rows_actually_use_the_fed_back_row(variant):
    """The mask must not be so tight that feedback is severed entirely.

    Guards the opposite failure from the leak test: a decoder that ignores its
    inputs would pass the causality assertion trivially while being no longer
    autoregressive.
    """
    decoder = _decoder(variant)
    batch = _batch()
    with torch.no_grad():
        base = decoder(dict(batch))["pred_action"]
    perturbed = batch["target"].clone()
    perturbed[:, 0] += 10.0
    with torch.no_grad():
        other = decoder(dict(batch, target=perturbed))["pred_action"]
    assert not torch.allclose(other[:, 1:], base[:, 1:]), (
        f"{variant}: perturbing row 0 changed nothing downstream — the decoder "
        "is ignoring its fed-back input"
    )


def test_bidirectional_never_reads_the_target():
    """Arm 2 is the control, so it must be provably target-independent.

    If the bidirectional arm could see the target it would beat every causal
    arm for a reason that has nothing to do with attention, and the ablation
    would invert.
    """
    decoder = _decoder("causal_bidir")
    batch = _batch()
    with torch.no_grad():
        with_target = decoder(dict(batch))["pred_action"]
        without = decoder(
            {"condition": batch["condition"], "embodiment": DOMAIN}
        )["pred_action"]
        scrambled = decoder(
            dict(batch, target=torch.randn_like(batch["target"]) * 100.0)
        )["pred_action"]
    torch.testing.assert_close(with_target, without)
    torch.testing.assert_close(with_target, scrambled)


@pytest.mark.parametrize("variant", ["causal_bidir", *CAUSAL_VARIANTS])
def test_train_and_rollout_agree_when_the_model_is_fed_its_own_output(variant):
    """Teacher forcing on the model's OWN rollout must reproduce the rollout.

    This is the property that makes the training objective and the deployed
    behaviour the same function. If it fails, the arm is training against a
    different model than the one the simulator runs.
    """
    decoder = _decoder(variant)
    seed = {"condition": torch.randn(2, COND_DIM), "embodiment": DOMAIN}
    with torch.no_grad():
        rolled = decoder(dict(seed))["pred_action"]
        replayed = decoder(dict(seed, target=rolled))["pred_action"]
    torch.testing.assert_close(rolled, replayed, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("variant", ["causal_bidir", *CAUSAL_VARIANTS])
def test_shapes_and_shared_loss_node(variant):
    """Every arm emits the sampler's exact output contract.

    Shared scoring is what keeps the arms comparable, so the shared loss node
    is exercised rather than a local MSE.
    """
    decoder = _decoder(variant)
    batch = _batch()
    out = decoder(dict(batch))
    assert out["pred_action"].shape == (3, HORIZON, ACTION_DIM)
    scored = NativeActionMSELoss()(dict(out, target=batch["target"]))
    assert torch.isfinite(scored["loss/native_action"])


def test_idm_leaves_the_velocity_row_to_the_head():
    """`append` layout's trailing row is a summary, not a point on the path.

    Differencing into it would invent a waypoint that does not exist and put a
    bogus target under the IDM.
    """
    decoder = _decoder("state_idm")
    assert decoder.n_waypoints == N_WAYPOINTS < HORIZON
    idm = decoder.idm[DOMAIN]
    assert idm.net[0].in_features == 2 * decoder.pose_channels
    assert idm.net[-1].out_features == ACTION_DIM - decoder.pose_channels


def test_idm_rejects_a_pose_width_that_leaves_it_nothing_to_predict():
    with pytest.raises(ValueError, match="non-pose channel"):
        _decoder("state_idm", pose_channels=ACTION_DIM)


def test_unknown_variant_and_embodiment_are_rejected_loudly():
    with pytest.raises(ValueError, match="variant must be one of"):
        _decoder("definitely_not_a_variant")
    decoder = _decoder("state_action_ar")
    with pytest.raises(KeyError, match="Unknown embodiment"):
        decoder({"condition": torch.randn(1, COND_DIM), "embodiment": "nope"})


def test_horizon_mismatch_is_rejected_rather_than_broadcast():
    decoder = _decoder("state_action_ar")
    bad = torch.randn(2, HORIZON - 1, ACTION_DIM)
    with pytest.raises(ValueError, match="expects target horizon"):
        decoder({"condition": torch.randn(2, COND_DIM), "embodiment": DOMAIN,
                 "target": bad})


@pytest.mark.parametrize("variant", ["causal_bidir", *CAUSAL_VARIANTS])
def test_rollout_contract_excludes_target(variant):
    reads, writes = _decoder(variant).contract("rollout")
    assert "target" not in reads
    assert "pred_action" in writes
