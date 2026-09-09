import pytest
import torch
import torch.nn as nn

from egomimic.pipeline.stages_io import ActionTargetBuilder
from egomimic.pipeline.stages_sampler import (
    DPStyleObsEncoder,
    FusedObsEncoder,
    GaussianLatentNoise,
    KeyedFeatureProjection,
)


class _PackedState(nn.Module):
    def __init__(self):
        super().__init__()
        self.crop_scope = "episode"
        self.seen = None

    def forward_packed(
        self,
        *,
        obs_packed,
        cu_seqlens,
        T_total,
        device,
        dtype,
        embodiment_id=None,
    ):
        self.seen = {
            "cu": cu_seqlens.tolist(),
            "T_total": T_total,
            "device": device,
            "dtype": dtype,
            "embodiment": embodiment_id,
        }
        return obs_packed["state"]


class _MalformedPackedState(_PackedState):
    def forward_packed(self, **kwargs):
        return super().forward_packed(**kwargs).unsqueeze(0)


def test_dp_style_encoder_concatenates_sliced_state_and_image_features():
    image_encoder = nn.Linear(3, 2, bias=False)
    encoder = DPStyleObsEncoder(
        obs_specs={"state": {"input_dim": 2, "input_slice": [1, 3]}},
        img_encoders={"image": image_encoder},
    )
    state = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    image = torch.ones(4, 3)

    output = encoder.forward_packed(
        obs_packed={"state": state, "image": image}, T_total=4
    )

    assert output.shape == (4, 4)
    assert torch.equal(output[:, :2], state[:, 1:3])
    assert torch.equal(output[:, 2:], image_encoder(image))


def test_fused_encoder_ignores_unrelated_metadata_and_action_target():
    encoder = _PackedState()
    stage = FusedObsEncoder(
        encoder=encoder,
        inputs={"state": "state"},
        n_obs_steps=2,
    )
    state = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    actions = torch.randn(3, 5, 6)

    output = stage({"state": state, "actions": actions, "selector": "arm_a"})

    assert output["condition"].shape == (3, 8)
    assert output["actions"] is actions
    assert "target" not in output
    assert encoder.seen == {
        "cu": [0, 2, 4, 6],
        "T_total": 6,
        "device": state.device,
        "dtype": state.dtype,
        "embodiment": None,
    }
    assert encoder._episode_cu.tolist() == [0, 2, 4, 6]


def test_fused_encoder_rejects_malformed_encoder_output():
    stage = FusedObsEncoder(
        encoder=_MalformedPackedState(),
        inputs={"state": "state"},
        n_obs_steps=2,
    )

    with pytest.raises(ValueError, match="output must have shape"):
        stage({"state": torch.randn(3, 2, 4)})


def test_action_target_builder_is_a_separate_train_only_node():
    actions = torch.randn(3, 5, 6)
    stage = ActionTargetBuilder()

    output = stage({"actions": actions})

    assert output == {"target": actions}
    assert stage.train_only is True
    assert stage.contract("train") == (("actions",), ("target",))


def test_fused_encoder_single_step_shape_does_not_depend_on_execution_mode():
    stage = FusedObsEncoder(
        encoder=_PackedState(),
        inputs={"state": "state"},
        n_obs_steps=1,
    )
    value = torch.randn(2, 4)
    output = stage({"state": value, "unrelated": "metadata"})
    assert output["condition"].shape == (2, 4)
    assert "target" not in output


def test_explicit_inputs_are_reflected_in_mode_contracts():
    stage = FusedObsEncoder(
        encoder=_PackedState(),
        inputs={"state": "state_tensor", "image": "camera_tensor"},
        n_obs_steps=1,
    )
    assert stage.contract("train")[0] == (
        "state_tensor",
        "camera_tensor",
    )
    assert stage.contract("inference")[0] == stage.contract("train")[0]


def test_fused_encoder_does_not_require_embodiment_metadata():
    encoder = _PackedState()
    stage = FusedObsEncoder(
        encoder=encoder,
        inputs={"state": "state"},
        n_obs_steps=1,
    )

    output = stage({"state": torch.randn(2, 4)})

    assert output["condition"].shape == (2, 4)
    assert encoder.seen["embodiment"] is None


def test_fused_encoder_declares_only_explicit_forward_context():
    encoder = _PackedState()
    stage = FusedObsEncoder(
        encoder=encoder,
        inputs={"state": "state"},
        n_obs_steps=1,
        forward_context={"route": "embodiment_id"},
    )

    output = stage({"state": torch.randn(2, 4), "route": "arm_a"})

    assert output["condition"].shape == (2, 4)
    assert encoder.seen["embodiment"] == "arm_a"
    assert stage.contract("train")[0] == ("state", "route")


@pytest.mark.parametrize(
    "forward_context,match",
    [
        ({"left": "route", "right": "route"}, "must be unique"),
        ({"route": "obs_packed"}, "cannot replace encoder inputs"),
    ],
)
def test_fused_encoder_rejects_ambiguous_forward_context(forward_context, match):
    with pytest.raises(ValueError, match=match):
        FusedObsEncoder(
            encoder=_PackedState(),
            inputs={"state": "state"},
            n_obs_steps=1,
            forward_context=forward_context,
        )


def test_gaussian_noise_uses_explicit_token_count():
    condition = torch.zeros(2, 7)
    assert GaussianLatentNoise(num_tokens=4, latent_dim=3)({"condition": condition})[
        "sampler/noise"
    ].shape == (2, 4, 3)


def test_keyed_projection_resolves_homogeneous_selector_alias():
    stage = KeyedFeatureProjection(
        selector_key="model_selector",
        input_key="features",
        output_key="projected",
        output_dim=3,
        projections={"image": {"input_dim": 2, "hidden_dim": 4}},
        selector_aliases={19: "image"},
    )
    batch = {
        "model_selector": torch.tensor([19, 19]),
        "features": torch.randn(2, 2),
    }

    output = stage(batch)

    assert output["projected"].shape == (2, 3)
    assert set(stage.state_dict()) == {
        "projections.image.0.weight",
        "projections.image.0.bias",
        "projections.image.2.weight",
        "projections.image.2.bias",
    }


def test_keyed_projection_rejects_mixed_selector_batch():
    stage = KeyedFeatureProjection(
        selector_key="model_selector",
        input_key="features",
        output_key="projected",
        projections={"image": {"input_dim": 2}},
    )

    with pytest.raises(ValueError, match="homogeneous"):
        stage(
            {
                "model_selector": torch.tensor([1, 2]),
                "features": torch.randn(2, 2),
            }
        )
