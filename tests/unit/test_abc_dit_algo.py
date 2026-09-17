"""ABC-DiT algo on stub encoders: prompts, batch conversion, train / eval."""

from __future__ import annotations

import functools

import pytest
import torch
from fixtures.stub_abc_encoders import StubTaskEncoder, StubVisionTower

from egomimic.algo.abc_dit import ABCDiT
from egomimic.models.abc_dit_nets import ABCDiTPolicy
from egomimic.rldb.embodiment.embodiment import get_embodiment_id

EMB = "human_bimanual"
EMB_ID = get_embodiment_id(EMB)
CAM = "observations.images.front_img_1"
PROPRIO = "observations.state.keypoints"
EXTRA_PROPRIO = "observations.state.ee_pose"
AC = "actions_keypoints"
B, T, D_ACT, D_PROP, D_EXTRA, WIDTH = 2, 4, 6, 3, 2, 8
IMAGE_SIZE = (16, 16)


class _StubNormStats:
    """The four norm-stat calls the algo makes, for one embodiment."""

    def keys_of_type(self, kind, embodiment_id):
        return {
            "camera_keys": [CAM],
            "proprio_keys": [PROPRIO, EXTRA_PROPRIO],
            "action_keys": [AC],
            "lang_keys": [],
        }[kind]

    def is_key_with_embodiment(self, key, embodiment_id):
        return key in (CAM, PROPRIO, EXTRA_PROPRIO, AC)

    def zarr_key_to_keyname(self, key, embodiment_id):
        return key

    def unnormalize(self, predictions, embodiment_id):
        return {k: v * 2.0 for k, v in predictions.items()}


def _algo(history_len: int = 1, **kwargs) -> ABCDiT:
    torch.manual_seed(0)
    policy = functools.partial(
        ABCDiTPolicy,
        camera_roles=[CAM],
        action_horizon=T,
        vision=StubVisionTower(hidden_size=32, num_patches=4),
        task_encoder=StubTaskEncoder(output_dim=16),
        size=None,
        hidden_size=32,
        depth=2,
        num_heads=2,
        vision_pool_num_queries=3,
        vision_pool_num_heads=2,
        num_inference_steps=2,
        mask_state_ratio=0.0,
    )
    defaults = dict(
        norm_stats=_StubNormStats(),
        policy=policy,
        domains=[EMB],
        dims={EMB: {"proprio": D_PROP, "action": D_ACT}},
        ac_keys={EMB: AC},
        action_width=WIDTH,
        annotation_key="annotations",
        default_prompt="",
        history_len=history_len,
        proprio_keys=[PROPRIO],
        device=torch.device("cpu"),
    )
    defaults.update(kwargs)
    return ABCDiT(**defaults)


def _raw_batch(K: int = 1, annotations=None) -> dict:
    g = torch.Generator().manual_seed(1)
    proprio = torch.randn(B, K, D_PROP, generator=g)
    if K == 1:
        proprio = proprio[:, 0]
    batch = {
        AC: torch.randn(B, T, D_ACT, generator=g),
        PROPRIO: proprio,
        EXTRA_PROPRIO: torch.randn(B, D_EXTRA, generator=g),
        CAM: torch.rand(B, 3, *IMAGE_SIZE, generator=g),
        "action_pad_mask": torch.tensor(
            [[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.float32
        ),
        "annotations": annotations
        if annotations is not None
        else [["fold the towel"], []],
    }
    return {EMB: batch}


def test_compose_prompt_is_the_bare_task_text_by_default():
    """ABC conditions CLIP on the task text; the embodiment block is opt-in."""
    algo = _algo()
    assert algo.compose_prompt(EMB, "fold the towel") == "fold the towel"
    assert algo.compose_prompt(EMB, "") == ""
    labelled = _algo(embodiment_label=True)
    assert (
        labelled.compose_prompt(EMB, "fold the towel")
        == "Embodiment: human bimanual. Task: fold the towel"
    )
    modes = _algo(control_mode={"human": "wrist frame"})
    assert modes.compose_prompt(EMB, "") == "Control mode: wrist frame."


def test_process_batch_builds_prompts_pad_mask_and_embodiment():
    algo = _algo()
    algo.nets.train()
    out = algo.process_batch_for_training(_raw_batch())[EMB_ID]
    assert out["sampled_prompt"] == ["fold the towel", ""]
    assert out["pad_mask"].shape == (B, T, 1) and torch.all(out["pad_mask"] == 1)
    assert out["embodiment"].tolist() == [EMB_ID]
    masked = _algo(use_pad_mask=True).process_batch_for_training(_raw_batch())[EMB_ID]
    assert masked["pad_mask"][0, -1, 0] == 0 and masked["pad_mask"][1, -1, 0] == 1


def test_model_data_pads_the_action_and_masks_the_padding():
    algo = _algo()
    batch = algo.process_batch_for_training(_raw_batch())[EMB_ID]
    data = algo._to_model_data(batch, EMB_ID)
    assert data["action"].shape == (B, T, WIDTH)
    assert torch.all(data["action"][..., D_ACT:] == 0)
    assert torch.all(data["loss_mask"][..., :D_ACT] == 1)
    assert torch.all(data["loss_mask"][..., D_ACT:] == 0)
    assert data["images"].shape == (B, 1, 3, *IMAGE_SIZE)
    assert data["state"].shape == (B, 1, D_PROP)
    assert data["prompts"] == ["fold the towel", ""]
    assert data["embodiment_name"] == EMB


def test_proprio_keys_select_the_state_vector():
    """ABC-DiT has ONE state vector, so an unwanted proprio key is named out
    rather than left stemless as HPT does."""
    algo = _algo()
    assert algo.proprio_keys[EMB_ID] == [PROPRIO]
    both = _algo(
        proprio_keys=None, dims={EMB: {"proprio": D_PROP + D_EXTRA, "action": D_ACT}}
    )
    batch = both.process_batch_for_training(_raw_batch())[EMB_ID]
    assert both._to_model_data(batch, EMB_ID)["state"].shape[-1] == D_PROP + D_EXTRA
    with pytest.raises(KeyError, match="are not proprio keys"):
        _algo(proprio_keys=["observations.state.nope"])


def test_forward_training_returns_a_finite_loss():
    algo = _algo()
    algo.nets.train()
    batch = algo.process_batch_for_training(_raw_batch())
    preds = algo.forward_training(batch)
    assert torch.isfinite(preds[f"{EMB}_loss"])
    losses = algo.compute_losses(preds, batch)
    assert losses["action_loss"] == preds[f"{EMB}_loss"]
    log = algo.log_info({"losses": losses})
    assert log["Loss"] == losses["action_loss"].item()


def test_forward_eval_is_deterministic_and_native_width():
    algo = _algo()
    algo.nets.eval()

    def one_pass():
        algo.reset_eval_pass_counter()
        batch = algo.process_batch_for_training(_raw_batch())
        return algo.forward_eval(batch)

    a, b = one_pass(), one_pass()
    assert a[f"{EMB}_{AC}"].shape == (B, T, D_ACT)  # unnormalised, native width
    assert torch.equal(a[f"{EMB}_{AC}"], b[f"{EMB}_{AC}"])
    assert torch.equal(a[f"{EMB}_loss"], b[f"{EMB}_loss"])
    algo.reset_eval_pass_counter()
    first = algo.forward_eval(algo.process_batch_for_training(_raw_batch()))
    second = algo.forward_eval(algo.process_batch_for_training(_raw_batch()))
    assert not torch.equal(first[f"{EMB}_{AC}"], second[f"{EMB}_{AC}"])


def test_proprio_history_is_flattened_into_the_state():
    algo = _algo(history_len=2)
    algo.nets.train()
    batch = algo.process_batch_for_training(_raw_batch(K=2))
    assert torch.isfinite(algo.forward_training(batch)[f"{EMB}_loss"])
    with pytest.raises(ValueError, match="history steps"):
        algo.forward_training(algo.process_batch_for_training(_raw_batch(K=1)))


def test_action_wider_than_action_width_is_rejected():
    with pytest.raises(ValueError, match="action_width"):
        _algo(action_width=D_ACT - 1)


def test_cameras_must_match_the_policys_roles():
    """The policy holds one query set and camera embedding per role, in order."""
    with pytest.raises(ValueError, match="do not match the policy's camera_roles"):
        _algo(
            policy=functools.partial(
                ABCDiTPolicy,
                camera_roles=["observations.images.right_wrist_img"],
                action_horizon=T,
                vision=StubVisionTower(),
                task_encoder=StubTaskEncoder(),
                size=None,
                hidden_size=32,
                depth=2,
                num_heads=2,
            )
        )


def test_dims_without_proprio_are_rejected():
    with pytest.raises(ValueError, match="conditions on proprio"):
        _algo(dims={EMB: {"action": D_ACT}})


def test_vision_parameters_go_to_the_backbone_lr_group():
    algo = _algo()
    vision = algo.nets["policy"].encoders["vision"]
    assert vision.backbone_parameters() and all(
        p.requires_grad for p in vision.backbone_parameters()
    )


def test_algo_is_built_on_the_cpu_and_the_device_setter_moves_the_nets():
    """Every DDP rank constructs the algo before Lightning assigns its GPU; the
    nets must not be parked on cuda:0 meanwhile."""
    algo = _algo()
    assert algo.device == torch.device("cpu")
    assert next(algo.nets.parameters()).device.type == "cpu"
    algo.device = "cpu"  # what ModelWrapper.on_fit_start writes
    assert algo.device == torch.device("cpu")
    assert next(algo.nets.parameters()).device.type == "cpu"
