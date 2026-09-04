"""Test the keypoint validity mask and the loss and viz paths that read it."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from egomimic.models.action_mask import ActionMaskMixin, masked_loss
from egomimic.rldb.embodiment.action_layout import (
    action_masks,
    keypoint_action_mask,
    keypoint_layout,
)
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.registry import KeypointSpec
from egomimic.utils.pose_utils import _split_keypoints

JAW = {"platform": "eva_x5", "end_effector": "eva_parallel_jaw"}
HAND = {"platform": "human_body", "end_effector": "mano_hand"}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action_dim,expected",
    [(140, (70, 7)), (138, (69, 6)), (126, (63, 0)), (14, None), (137, None)],
)
def test_the_wrist_width_falls_out_of_the_action_width(action_dim, expected) -> None:
    assert keypoint_layout(action_dim, n_slots=21) == expected


def test_an_end_effector_that_owns_every_slot_masks_nothing() -> None:
    assert keypoint_action_mask(HAND, 138) is None
    assert keypoint_action_mask("human_bimanual", 140) is None


def test_a_cartesian_width_is_not_a_keypoint_layout() -> None:
    assert keypoint_action_mask(JAW, 14) is None


def test_a_jaw_in_the_keypoint_space_keeps_its_three_slots() -> None:
    mask = keypoint_action_mask(JAW, 138)

    # Per side: 6 wrist dims plus 3 owned slots of 3 coordinates each.
    assert mask.sum() == 2 * (6 + 3 * 3)
    left = mask[:69]
    assert left[:6].all(), "the wrist pose is never masked"
    for slot in range(21):
        block = left[6 + 3 * slot : 9 + 3 * slot]
        assert block.all() == (slot in (0, 4, 8)), slot


def test_the_mask_lines_up_with_the_split_the_viz_path_uses() -> None:
    mask = keypoint_action_mask(JAW, 138)
    _, _, left, _, _, right = _split_keypoints(mask, wrist_in_data=True, is_quat=False)

    for side in (left, right):
        assert side.reshape(21, 3)[:, 0].sum() == 3


def test_sides_may_carry_different_end_effectors() -> None:
    mask = keypoint_action_mask(
        {
            "platform": "eva_x5",
            "end_effector": {"left": "eva_parallel_jaw", "right": "mano_hand"},
        },
        138,
    )

    assert mask[:69].sum() == 6 + 9
    assert mask[69:].all()


def test_a_head_only_carries_masks_for_what_it_masks() -> None:
    assert action_masks({"human_bimanual": 138, "eva_bimanual": 14}) == {}
    assert action_masks({"unknown_embodiment": 138}) == {}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class _Head(ActionMaskMixin, torch.nn.Module):
    def __init__(self, infer_ac_dims):
        super().__init__()
        self.init_action_masks(infer_ac_dims)


def test_no_mask_takes_the_original_loss_call_bit_for_bit() -> None:
    generator = torch.Generator().manual_seed(0)
    pred = torch.randn(4, 3, 138, generator=generator)
    target = torch.randn(4, 3, 138, generator=generator)

    assert masked_loss(F.mse_loss, pred, target, None) is not None
    assert torch.equal(
        masked_loss(F.mse_loss, pred, target, None), F.mse_loss(pred, target)
    )


def test_an_all_true_mask_reproduces_the_unmasked_loss() -> None:
    generator = torch.Generator().manual_seed(1)
    pred = torch.randn(4, 3, 138, generator=generator)
    target = torch.randn(4, 3, 138, generator=generator)
    mask = torch.ones(138, dtype=torch.bool)

    assert torch.allclose(
        masked_loss(F.mse_loss, pred, target, mask),
        F.mse_loss(pred, target),
        atol=1e-6,
    )


def test_a_masked_dimension_cannot_change_the_loss() -> None:
    generator = torch.Generator().manual_seed(2)
    pred = torch.randn(2, 3, 138, generator=generator)
    target = torch.randn(2, 3, 138, generator=generator)
    mask = torch.from_numpy(keypoint_action_mask(JAW, 138))

    before = masked_loss(F.mse_loss, pred, target, mask)
    target[..., ~mask] += 1000.0
    after = masked_loss(F.mse_loss, pred, target, mask)

    assert torch.equal(before, after)


def test_a_masked_loss_still_sees_the_dimensions_it_keeps() -> None:
    mask = torch.from_numpy(keypoint_action_mask(JAW, 138))
    pred = torch.zeros(1, 1, 138)
    target = torch.zeros(1, 1, 138)
    target[..., 0] = 2.0

    assert masked_loss(F.mse_loss, pred, target, mask) > 0


def test_the_head_selects_the_mask_by_the_batch_embodiment() -> None:
    head = _Head({"eva_bimanual": 138})
    target = torch.zeros(2, 3, 138)
    data = {"embodiment": torch.tensor([get_embodiment_id("eva_bimanual")] * 2)}

    mask = head.action_mask(data, target)

    assert mask.shape == (1, 1, 138)
    assert int(mask.sum()) == 2 * (6 + 9)


def test_a_batch_with_no_masked_embodiment_gets_no_mask() -> None:
    head = _Head({"eva_bimanual": 138})
    target = torch.zeros(2, 3, 138)

    assert head.action_mask({}, target) is None
    assert (
        head.action_mask(
            {"embodiment": torch.tensor([get_embodiment_id("human_bimanual")])}, target
        )
        is None
    )


def test_a_mixed_batch_gets_one_mask_row_per_sample() -> None:
    head = _Head({"eva_bimanual": 138})
    target = torch.zeros(2, 3, 138)
    ids = [get_embodiment_id("eva_bimanual"), get_embodiment_id("human_bimanual")]

    mask = head.action_mask({"embodiment": torch.tensor(ids)}, target)

    assert mask.shape == (2, 1, 138)
    assert int(mask[0].sum()) == 2 * (6 + 9)
    assert bool(mask[1].all())


def test_a_head_with_no_masked_embodiment_registers_no_buffer() -> None:
    head = _Head({"human_bimanual": 138})

    assert head.action_mask(
        {"embodiment": torch.tensor([get_embodiment_id("human_bimanual")])},
        torch.zeros(1, 138),
    ) is None
    assert not [name for name, _ in head.named_buffers()]


def test_masks_stay_out_of_the_checkpoint() -> None:
    head = _Head({"eva_bimanual": 138})

    assert head.state_dict() == {}


def test_a_denoising_head_masks_the_loss_it_computes() -> None:
    from egomimic.models.denoising_policy import DenoisingPolicy

    class _Echo(DenoisingPolicy):
        """Return the actions themselves, so the loss reads the action dims."""

        def predict(self, actions, global_cond):
            return actions, torch.zeros_like(actions)

    head = _Echo(
        model=torch.nn.Linear(1, 1),
        action_horizon=3,
        infer_ac_dims={"eva_bimanual": 138},
    )
    actions = torch.zeros(2, 3, 138)
    data = {
        "action": actions,
        "embodiment": torch.tensor([get_embodiment_id("eva_bimanual")] * 2),
    }
    global_cond = torch.zeros(2, 4)

    baseline = head.compute_loss(global_cond, data)
    mask = torch.from_numpy(keypoint_action_mask(JAW, 138))
    data["action"] = actions.clone()
    data["action"][..., ~mask] = 50.0

    assert torch.equal(head.compute_loss(global_cond, data), baseline)
    data["action"][..., 0] = 50.0
    assert head.compute_loss(global_cond, data) > baseline


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _draw(actions, **kwargs):
    from egomimic.rldb.embodiment.human import Human

    image = np.zeros((80, 80, 3), dtype=np.uint8)
    intrinsics = np.array(
        [[40.0, 0, 40, 0], [0, 40.0, 40, 0], [0, 0, 1.0, 0]]
    )
    return Human.viz(image, actions, mode="keypoints", intrinsics=intrinsics, **kwargs)


def _spread_keypoints(n_kp: int) -> np.ndarray:
    """Return a 138-wide tensor whose slots project to distinct pixels."""
    actions = np.zeros(2 * (6 + 3 * n_kp))
    for side in range(2):
        base = side * (6 + 3 * n_kp)
        for slot in range(n_kp):
            actions[base + 6 + 3 * slot] = -0.4 + 0.04 * slot + 0.4 * side
            actions[base + 6 + 3 * slot + 1] = -0.2 + 0.02 * slot
            actions[base + 6 + 3 * slot + 2] = 1.0
    return actions


def test_masked_slots_are_not_drawn() -> None:
    actions = _spread_keypoints(21)
    everything = _draw(actions)

    three_slots = _draw(actions, keypoint_spec=KeypointSpec("mano21", (0, 4, 8)))

    assert everything.any()
    assert three_slots.any()
    assert int((three_slots > 0).sum()) < int((everything > 0).sum())


def test_a_complete_spec_draws_what_no_spec_draws() -> None:
    actions = _spread_keypoints(21)

    assert np.array_equal(
        _draw(actions),
        _draw(actions, keypoint_spec=KeypointSpec("mano21", tuple(range(21)))),
    )
