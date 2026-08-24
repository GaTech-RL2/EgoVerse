from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from egomimic.models.stems.cond_encoders import CondEncoderModule
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.arc_length import ArcLengthRolloutAdapter
from egomimic.pipeline.core import Stage
from egomimic.pipeline.stages_sampler import NativeActionMSELoss
from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.zarr.arc_length_tokenizer import (
    TokenizeBimanualArcLengthCartesian,
)
from egomimic.robot.rollout import PolicyRollout


def _arc_tokens(batch_size=1):
    token = torch.zeros(batch_size, 4, 8)
    token[:, :3, 0] = torch.tensor([0.0, 0.1, 0.2])
    token[:, :3, 5] = torch.tensor([0.0, 0.1, 0.2])
    token[:, :3, 3] = torch.tensor([0.0, 0.5, 1.0])
    token[:, :3, 7] = torch.tensor([1.0, 0.5, 0.0])
    token[:, 3, 0] = 1.0
    token[:, 3, 5] = 1.0
    return token


def test_model_facing_tokenizer_emits_waypoints_plus_velocity_token():
    transform = TokenizeBimanualArcLengthCartesian(
        min_distance_unit=0.4,
        resampled_vector_length=25,
        dt=1.0 / 30.0,
    )
    actions = np.zeros((100, 14), dtype=np.float64)
    actions[:, 0] = np.linspace(0.0, 0.5, 100)
    actions[:, 6] = np.linspace(0.0, 1.0, 100)
    actions[:, 8] = np.linspace(0.0, 0.5, 100)
    actions[:, 13] = np.linspace(1.0, 0.0, 100)

    tokenized = transform.transform({"actions_cartesian": actions})[
        "actions_cartesian"
    ]
    decoded = transform.detokenize(tokenized, action_horizon=100)

    assert tokenized.shape == (26, 8)
    assert decoded.shape == (100, 8)
    assert np.isfinite(tokenized).all()
    assert np.linalg.norm(tokenized[-1, :3]) > 0.0
    assert np.linalg.norm(tokenized[-1, 4:7]) > 0.0


def test_model_facing_tokenizer_rejects_invalid_pose_sentinel():
    transform = TokenizeBimanualArcLengthCartesian()
    actions = np.zeros((100, 14), dtype=np.float64)
    actions[5, 0] = 1e9

    with pytest.raises(ValueError, match="invalid-pose sentinel"):
        transform.transform({"actions_cartesian": actions})


class _TinyImageEncoder(nn.Module):
    embed_dim = 2

    def forward(self, value):
        return value.mean(dim=(-2, -1))[..., : self.embed_dim]


def test_condition_encoder_accepts_exact_dotted_arc_data_aliases():
    encoder = CondEncoderModule(
        d_cond=4,
        obs_specs={
            "observations.state.ee_pose": {
                "input_dim": 3,
                "embed_dim": 2,
            }
        },
        img_encoders={
            "observations.images.front_img_1": _TinyImageEncoder(),
        },
        cond_proj_widths=[],
    )

    encoded = encoder.encode(
        {
            "observations.state.ee_pose": torch.ones(2, 3),
            "observations.images.front_img_1": torch.ones(2, 3, 4, 4),
        },
        T_action=1,
    )

    assert encoded["fused_cond"].shape == (2, 1, 4)
    assert all("." not in key for key in encoder.obs_encoders)
    assert all("." not in key for key in encoder.img_encoders)


def test_embodiment_hooks_keep_source_arc_windows_and_stride_aware_dt():
    assert Eva._get_keymap("arc_tokenizer_cartesian")["left.cmd_ee_pose"][
        "horizon"
    ] == 200
    assert Human._get_keymap("arc_tokenizer_cartesian")["left.action_ee_pose"][
        "horizon"
    ] == 600
    eva_tokenizer = next(
        stage
        for stage in Eva.get_transform_list(
            "arc_tokenizer_cartesian",
            min_distance_unit=0.4,
            resampled_vector_length=25,
        )
        if isinstance(stage, TokenizeBimanualArcLengthCartesian)
    )
    human_tokenizer = next(
        stage
        for stage in Human.get_transform_list(
            "arc_tokenizer_cartesian",
            stride=1,
            min_distance_unit=0.4,
            resampled_vector_length=25,
        )
        if isinstance(stage, TokenizeBimanualArcLengthCartesian)
    )
    assert eva_tokenizer.M == 25
    assert human_tokenizer.M == 25
    assert eva_tokenizer.tokenizer.config.dt == pytest.approx(199.0 / (30.0 * 99.0))
    assert human_tokenizer.tokenizer.config.dt == pytest.approx(
        599.0 / (30.0 * 99.0)
    )


def test_corrected_token_dt_recovers_physical_constant_speed():
    dt = 199.0 / (30.0 * 99.0)
    transform = TokenizeBimanualArcLengthCartesian(
        min_distance_unit=0.4,
        resampled_vector_length=25,
        dt=dt,
    )
    actions = np.zeros((100, 14), dtype=np.float64)
    actions[:, 0] = np.linspace(0.0, 199.0 / 30.0, 100)

    tokens = transform.transform({"actions_cartesian": actions})[
        "actions_cartesian"
    ]

    assert tokens[-1, 0] == pytest.approx(1.0)


def test_arc_rollout_adapter_decodes_time_and_holds_current_orientation():
    adapter = ArcLengthRolloutAdapter(
        min_distance_unit=0.2,
        resampled_vector_length=3,
        action_horizon=4,
        dt=0.1,
    )
    state = torch.tensor(
        [[0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0,
          0.0, 0.0, 0.0, -0.1, -0.2, -0.3, 0.0]]
    )
    decoded = adapter.decode(_arc_tokens(), {"state_ee_pose": state})

    assert decoded.shape == (1, 4, 14)
    assert torch.allclose(decoded[0, :, 0], torch.tensor([0.0, 0.1, 0.2, 0.2]))
    assert torch.allclose(decoded[0, :, 8], torch.tensor([0.0, 0.1, 0.2, 0.2]))
    assert torch.allclose(decoded[0, :, 3:6], state[0, 3:6].expand(4, 3))
    assert torch.allclose(decoded[0, :, 10:13], state[0, 10:13].expand(4, 3))
    assert torch.isfinite(decoded).all()


def test_stationary_wrist_gripper_motion_survives_rollout_decode():
    transform = TokenizeBimanualArcLengthCartesian(
        min_distance_unit=0.4,
        resampled_vector_length=4,
        dt=0.1,
    )
    actions = np.zeros((4, 14), dtype=np.float64)
    actions[:, 6] = np.linspace(0.0, 1.0, 4)
    actions[:, 13] = np.linspace(1.0, 0.0, 4)

    tokens = transform.transform({"actions_cartesian": actions})[
        "actions_cartesian"
    ]
    decoded = transform.detokenize(tokens, action_horizon=4)

    np.testing.assert_allclose(decoded[:, 0:3], 0.0)
    np.testing.assert_allclose(decoded[:, 4:7], 0.0)
    np.testing.assert_allclose(decoded[:, 3], actions[:, 6])
    np.testing.assert_allclose(decoded[:, 7], actions[:, 13])


def test_moving_wrist_gripper_keeps_xyz_coupled_source_schedule():
    transform = TokenizeBimanualArcLengthCartesian(
        min_distance_unit=0.9,
        resampled_vector_length=4,
        dt=1.0,
    )
    tokens = np.zeros((5, 8), dtype=np.float64)
    tokens[:4, 0] = [0.0, 0.3, 0.4, 0.9]
    tokens[:4, 3] = [0.0, 0.3, 0.6, 1.0]
    tokens[4, 0] = 0.1
    tokens[4, 3] = 100.0

    decoded = transform.detokenize(tokens, action_horizon=4)

    np.testing.assert_allclose(decoded[:, 0], [0.0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(decoded[:, 3], [0.0, 0.1, 0.2, 0.3])


def test_curved_waypoint_decode_preserves_chord_velocity_duration():
    transform = TokenizeBimanualArcLengthCartesian(
        min_distance_unit=2.0,
        resampled_vector_length=3,
        dt=1.0,
    )
    tokens = np.zeros((4, 8), dtype=np.float64)
    tokens[:3, 0:2] = [[0.0, 0.0], [0.5, 0.5], [1.0, 0.0]]
    tokens[3, 0] = 0.5

    decoded = transform.detokenize(tokens, action_horizon=3)

    np.testing.assert_allclose(decoded[0, 0:3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(decoded[1, 0:3], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(decoded[2, 0:3], [1.0, 0.0, 0.0])


class _IdentityNormStats:
    def keys_of_type(self, key_type, emb_id):
        return {
            "proprio_keys": ["state_ee_pose"],
            "lang_keys": [],
            "camera_keys": [],
            "action_keys": ["actions_cartesian"],
        }[key_type]

    def is_key_with_embodiment(self, key, emb_id):
        return key in {"state_ee_pose", "actions_cartesian"}

    def zarr_key_to_keyname(self, key, emb_id):
        return key if self.is_key_with_embodiment(key, emb_id) else None

    def keyname_to_zarr_key(self, key, emb_id):
        return key if self.is_key_with_embodiment(key, emb_id) else None

    def normalize(self, data, emb_id):
        return dict(data)

    def unnormalize(self, data, emb_id):
        return dict(data)


class _AffineNormStats(_IdentityNormStats):
    def normalize(self, data, emb_id):
        out = dict(data)
        if "state_ee_pose" in out:
            out["state_ee_pose"] = out["state_ee_pose"] * 0.5
        return out

    def unnormalize(self, data, emb_id):
        out = dict(data)
        if "state_ee_pose" in out:
            out["state_ee_pose"] = out["state_ee_pose"] * 2.0
        if "actions_cartesian" in out:
            out["actions_cartesian"] = out["actions_cartesian"] * 2.0
        return out


class _RolloutObs(Stage):
    reads = ["obs/*", "embodiment"]
    writes = ["condition", "target"]
    rollout_obs_steps = 1

    def forward(self, batch):
        assert "actions" not in batch
        assert batch["obs/state_ee_pose"].shape == (1, 1, 14)
        batch["condition"] = torch.zeros(1, 1)
        return batch


class _FixedArcPrediction(Stage):
    reads = ["condition", "embodiment"]
    writes = ["pred_action"]

    def forward(self, batch):
        batch["pred_action"] = _arc_tokens().to(batch["condition"])
        return batch


def test_pipeline_algo_rollout_is_action_free_and_decodes_arc_tokens():
    adapter = ArcLengthRolloutAdapter(
        min_distance_unit=0.2,
        resampled_vector_length=3,
        action_horizon=4,
        dt=0.1,
    )
    algo = PipelineAlgo(
        stages=[_RolloutObs(), _FixedArcPrediction(), NativeActionMSELoss()],
        norm_stats=_IdentityNormStats(),
        domains=["eva_bimanual"],
        ac_keys={"eva_bimanual": "actions_cartesian"},
        action_horizon=4,
        rollout_adapter=adapter,
        device=torch.device("cpu"),
    )
    algo.nets.eval()
    raw = {
        "eva_bimanual": {
            "state_ee_pose": torch.tensor(
                [[0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0,
                  0.0, 0.0, 0.0, -0.1, -0.2, -0.3, 0.0]]
            )
        }
    }
    processed = algo.process_batch_for_rollout(raw)
    predictions = algo.forward_rollout(processed)

    assert predictions["eva_bimanual_actions_cartesian"].shape == (1, 4, 14)
    assert predictions["eva_bimanual_actions_cartesian_tokens"].shape == (1, 4, 8)
    assert np.isfinite(
        predictions["eva_bimanual_actions_cartesian"].detach().numpy()
    ).all()


def test_pipeline_rollout_normalizes_observation_and_unnormalizes_tokens_once():
    algo = PipelineAlgo(
        stages=[_RolloutObs(), _FixedArcPrediction(), NativeActionMSELoss()],
        norm_stats=_AffineNormStats(),
        domains=["eva_bimanual"],
        ac_keys={"eva_bimanual": "actions_cartesian"},
        action_horizon=4,
        rollout_adapter=ArcLengthRolloutAdapter(
            min_distance_unit=0.2,
            resampled_vector_length=3,
            action_horizon=4,
            dt=0.1,
        ),
        device=torch.device("cpu"),
    )
    algo.nets.eval()
    state = torch.tensor(
        [
            [
                0.0,
                0.0,
                0.0,
                0.1,
                0.2,
                0.3,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.1,
                -0.2,
                -0.3,
                0.0,
            ]
        ]
    )

    processed = algo.process_batch_for_rollout(
        {"eva_bimanual": {"state_ee_pose": state}}
    )
    processed_state = next(iter(processed.values()))["state_ee_pose"]
    assert processed_state[0, 3] == pytest.approx(0.05)
    decoded = algo.forward_rollout(processed)["eva_bimanual_actions_cartesian"]

    assert torch.allclose(decoded[0, :, 0], torch.tensor([0.0, 0.2, 0.4, 0.4]))
    assert torch.allclose(decoded[0, :, 3:6], state[0, 3:6].expand(4, 3))


def test_robot_rollout_enforces_arc_action_contract():
    model = SimpleNamespace(
        domains=["eva_bimanual"],
        rollout_adapter=ArcLengthRolloutAdapter(
            min_distance_unit=0.4,
            resampled_vector_length=25,
            action_horizon=100,
        ),
    )

    effective = PolicyRollout._resolve_action_chunk_contract(
        model=model,
        embodiment_name="eva_bimanual",
        arm="both",
        cartesian=True,
        query_frequency=30,
        requested_resampled_len=45,
    )

    assert effective == 100
    with pytest.raises(ValueError, match="--arms both"):
        PolicyRollout._resolve_action_chunk_contract(
            model=model,
            embodiment_name="eva_bimanual",
            arm="right",
            cartesian=True,
            query_frequency=30,
            requested_resampled_len=100,
        )
    with pytest.raises(ValueError, match="--cartesian"):
        PolicyRollout._resolve_action_chunk_contract(
            model=model,
            embodiment_name="eva_bimanual",
            arm="both",
            cartesian=False,
            query_frequency=30,
            requested_resampled_len=100,
        )
