"""QwenVLA algo on the fake VLM: prompts, batch conversion, train / eval paths."""

from __future__ import annotations

import functools

import pytest
import torch
from fixtures.fake_qwen35 import STUB_IMAGE_SIZE, install_fake_qwen35

from egomimic.algo.qwenvla import QwenVLA
from egomimic.models.layerwise_dit import LayerwiseFMHead
from egomimic.models.qwenvla_nets import QwenVLBackbone
from egomimic.rldb.embodiment.embodiment import get_embodiment_id

EMB = "human_bimanual"
EMB_ID = get_embodiment_id(EMB)
CAM = "observations.images.front_img_1"
PROPRIO = "observations.state.ee_pose"
AC = "actions_cartesian"
B, T, D_ACT, D_PROP, WIDTH = 2, 4, 6, 3, 8


class _StubNormStats:
    """The four norm-stat calls the algo makes, for one embodiment."""

    def keys_of_type(self, kind, embodiment_id):
        return {
            "camera_keys": [CAM],
            "proprio_keys": [PROPRIO],
            "action_keys": [AC],
            "lang_keys": [],
        }[kind]

    def is_key_with_embodiment(self, key, embodiment_id):
        return key in (CAM, PROPRIO, AC)

    def zarr_key_to_keyname(self, key, embodiment_id):
        return key

    def unnormalize(self, predictions, embodiment_id):
        return {k: v * 2.0 for k, v in predictions.items()}


def _algo(snapshot, history_len: int = 1, **kwargs) -> QwenVLA:
    torch.manual_seed(0)
    backbone = QwenVLBackbone(
        model_name=snapshot, dtype="float32", image_size=STUB_IMAGE_SIZE
    )
    head = functools.partial(
        LayerwiseFMHead,
        action_horizon=T,
        head_dim=8,
        num_register_tokens=2,
        dropout=0.0,
        num_inference_steps=2,
    )
    defaults = dict(
        norm_stats=_StubNormStats(),
        backbone=backbone,
        head=head,
        domains=[EMB],
        dims={EMB: {"proprio": D_PROP, "action": D_ACT}},
        ac_keys={EMB: AC},
        action_width=WIDTH,
        annotation_key="annotations",
        default_prompt="",
        history_len=history_len,
        device=torch.device("cpu"),
    )
    defaults.update(kwargs)
    return QwenVLA(**defaults)


def _raw_batch(K: int = 1, annotations=None) -> dict:
    g = torch.Generator().manual_seed(1)
    proprio = torch.randn(B, K, D_PROP, generator=g)
    if K == 1:
        proprio = proprio[:, 0]
    batch = {
        AC: torch.randn(B, T, D_ACT, generator=g),
        PROPRIO: proprio,
        CAM: torch.rand(B, 3, *STUB_IMAGE_SIZE, generator=g),
        "action_pad_mask": torch.tensor(
            [[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.float32
        ),
        "annotations": annotations
        if annotations is not None
        else [["fold the towel"], []],
    }
    return {EMB: batch}


@pytest.fixture()
def snapshot(tmp_path, monkeypatch):
    return install_fake_qwen35(tmp_path, monkeypatch)


def test_compose_prompt(snapshot):
    algo = _algo(snapshot)
    assert (
        algo.compose_prompt(EMB, "fold the towel")
        == "Embodiment: human bimanual. Task: fold the towel"
    )
    assert algo.compose_prompt(EMB, "") == "Embodiment: human bimanual."
    plain = _algo(snapshot, embodiment_label=False)
    assert plain.compose_prompt(EMB, "fold") == "Task: fold"
    modes = _algo(snapshot, control_mode={"human": "wrist frame"})
    assert (
        modes.compose_prompt(EMB, "")
        == "Embodiment: human bimanual. Control mode: wrist frame."
    )


def test_process_batch_builds_prompts_pad_mask_and_embodiment(snapshot):
    algo = _algo(snapshot)
    algo.nets.train()
    out = algo.process_batch_for_training(_raw_batch())[EMB_ID]
    assert out["sampled_prompt"] == ["fold the towel", ""]
    assert out["pad_mask"].shape == (B, T, 1) and torch.all(out["pad_mask"] == 1)
    assert out["embodiment"].tolist() == [EMB_ID]
    masked = _algo(snapshot, use_pad_mask=True).process_batch_for_training(
        _raw_batch()
    )[EMB_ID]
    assert masked["pad_mask"][0, -1, 0] == 0 and masked["pad_mask"][1, -1, 0] == 1


def test_model_data_pads_the_action_and_masks_the_padding(snapshot):
    algo = _algo(snapshot)
    batch = algo.process_batch_for_training(_raw_batch())[EMB_ID]
    data = algo._to_model_data(batch, EMB_ID)
    assert data["action"].shape == (B, T, WIDTH)
    assert torch.all(data["action"][..., D_ACT:] == 0)
    assert data["loss_mask"].shape == (B, T, WIDTH)
    assert torch.all(data["loss_mask"][..., :D_ACT] == 1)
    assert torch.all(data["loss_mask"][..., D_ACT:] == 0)
    assert data["images"].shape == (B, 1, 3, *STUB_IMAGE_SIZE)
    assert data["state"].shape == (B, 1, D_PROP)
    assert data["prompts"] == [
        "Embodiment: human bimanual. Task: fold the towel",
        "Embodiment: human bimanual.",
    ]
    assert data["embodiment_name"] == EMB


def test_forward_training_returns_a_finite_loss(snapshot):
    algo = _algo(snapshot)
    algo.nets.train()
    batch = algo.process_batch_for_training(_raw_batch())
    preds = algo.forward_training(batch)
    assert torch.isfinite(preds[f"{EMB}_loss"])
    losses = algo.compute_losses(preds, batch)
    assert losses["action_loss"] == preds[f"{EMB}_loss"]
    log = algo.log_info({"losses": losses})
    assert log["Loss"] == losses["action_loss"].item()


def test_forward_eval_is_deterministic_and_native_width(snapshot):
    algo = _algo(snapshot)
    algo.nets.eval()

    def one_pass():
        algo.reset_eval_pass_counter()
        batch = algo.process_batch_for_training(_raw_batch())
        return algo.forward_eval(batch)

    a, b = one_pass(), one_pass()
    assert a[f"{EMB}_{AC}"].shape == (B, T, D_ACT)
    assert torch.equal(a[f"{EMB}_{AC}"], b[f"{EMB}_{AC}"])
    assert torch.equal(a[f"{EMB}_loss"], b[f"{EMB}_loss"])
    # a second batch within one pass draws different noise
    algo.reset_eval_pass_counter()
    first = algo.forward_eval(algo.process_batch_for_training(_raw_batch()))
    second = algo.forward_eval(algo.process_batch_for_training(_raw_batch()))
    assert not torch.equal(first[f"{EMB}_{AC}"], second[f"{EMB}_{AC}"])


def test_proprio_history_feeds_k_state_tokens(snapshot):
    algo = _algo(snapshot, history_len=2)
    algo.nets.train()
    batch = algo.process_batch_for_training(_raw_batch(K=2))
    assert torch.isfinite(algo.forward_training(batch)[f"{EMB}_loss"])
    with pytest.raises(ValueError):
        algo.forward_training(algo.process_batch_for_training(_raw_batch(K=1)))


def test_action_wider_than_action_width_is_rejected(snapshot):
    with pytest.raises(ValueError):
        _algo(snapshot, action_width=D_ACT - 1)


def test_backbone_parameters_go_to_the_vlm_lr_group(snapshot):
    algo = _algo(snapshot)
    vlm = algo.nets["policy"].encoders["vlm"]
    assert vlm.backbone_parameters() and all(
        p.requires_grad for p in vlm.backbone_parameters()
    )


def test_algo_is_built_on_the_cpu_and_the_device_setter_moves_the_nets(snapshot):
    """Every DDP rank constructs the algo before Lightning assigns its GPU; the
    nets must not be parked on cuda:0 meanwhile (8 x 4.8 GB overflowed it on a
    checkpoint resume, 2026-09-16)."""
    algo = _algo(snapshot)
    assert algo.device == torch.device("cpu")
    assert next(algo.nets.parameters()).device.type == "cpu"
    algo.device = "cpu"  # what ModelWrapper.on_fit_start writes (a str or device)
    assert algo.device == torch.device("cpu")
    assert next(algo.nets.parameters()).device.type == "cpu"
