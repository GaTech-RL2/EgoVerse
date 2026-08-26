from pathlib import Path

from omegaconf import OmegaConf
import numpy as np
import pytest
import torch

from egomimic.pipeline.stages_flow import SDPHead
from egomimic.rldb.embodiment.embodiment import (
    get_embodiment,
    get_embodiment_id,
)
from egomimic.rldb.embodiment.pushshapes import _draw_chunk
from egomimic.rldb.embodiment.pushshapes import get_keymap_with_pusher_cmd


ROOT = Path(__file__).parents[1]


def _small_head():
    return SDPHead(
        d_a=8,
        d_s=4,
        action_dim=2,
        action_dims={
            "pushshapes_sim": 2,
            "pushshapes_sim_u_socket": 3,
        },
        latent_dim=3,
        dual_stream=True,
        enc_hidden=8,
        chunk_len=2,
        embodiments=["pushshapes_sim", "pushshapes_sim_u_socket"],
        buffer_chunks=2,
        num_train_timesteps=8,
        num_inference_steps=4,
        end_mode="pusher_hold",
        denoiser_arch="adaln",
        d_model_a=8,
        d_model_s=8,
        n_layers=1,
        n_heads=2,
        ffn_mult=2,
    )


def test_usocket_embodiment_registration():
    assert get_embodiment_id("pushshapes_sim_u_socket") == 19
    assert get_embodiment(19) == "PUSHSHAPES_SIM_U_SOCKET"


def test_pusher_command_keymap_matches_action_horizon():
    key_map = get_keymap_with_pusher_cmd(action_horizon=37)
    assert key_map["pusher_cmd_pose"] == {
        "key_type": "proprio_keys",
        "zarr_key": "observations.pusher_cmd_pose",
        "horizon": 37,
    }


@pytest.mark.parametrize(
    ("embodiment", "action_dim"),
    (("pushshapes_sim", 2), ("pushshapes_sim_u_socket", 3)),
)
def test_heterogeneous_pusher_hold_targets_and_backward(embodiment, action_dim):
    torch.manual_seed(0)
    head = _small_head().train()
    T, C = 3, head.C
    target = torch.randn(T, C, action_dim)
    state = torch.randn(T, 6)
    final_pose = torch.arange(1, action_dim + 1, dtype=state.dtype)
    state[-1, :action_dim] = final_pose
    cu = torch.tensor([0, T])

    buffered, valid = head._buffer_targets(target, cu, state)
    assert buffered.shape == (T, head.K, C, action_dim)
    assert valid.all()
    torch.testing.assert_close(
        buffered[-1, 1],
        final_pose[None, :].expand(C, action_dim),
    )

    batch = {
        "a_top": torch.randn(T, 8),
        "s": torch.randn(T, 4),
        "embodiment": embodiment,
        "cu_seqlens": cu,
        "target": target,
        "obs/state_agent_obj": state,
    }
    loss = head(batch)["loss/sdp"]
    assert loss.isfinite()
    loss.backward()
    assert head.net.enc[embodiment][0].weight.grad is not None
    assert head.net.dec[embodiment][-1].weight.grad is not None


def test_production_configs_match_2d_3d_contract():
    model = OmegaConf.load(
        ROOT / "egomimic/hydra_configs/model/bf/bf_nopre_sdp_noattn_usocket.yaml"
    )
    data = OmegaConf.load(
        ROOT / "egomimic/hydra_configs/data/pusht/circle4500_usocket.yaml"
    )
    stages = model.robomimic_model.stages
    head = stages[-2]

    assert list(model.robomimic_model.domains) == [
        "pushshapes_sim",
        "pushshapes_sim_u_socket",
    ]
    assert dict(head.action_dims) == {
        "pushshapes_sim": 2,
        "pushshapes_sim_u_socket": 3,
    }
    assert head.latent_dim == 3
    assert head.end_mode == "pusher_hold"
    assert stages[2].levels[0].attn_window == 1
    assert stages[2].levels[2].attn_window == 1

    specific = stages[1].specific[0].obs_encoder.encoders
    assert specific.pushshapes_sim.obs_specs.state_agent_obj.input_dim == 2
    assert specific.pushshapes_sim_u_socket.obs_specs.state_agent_obj.input_dim == 3
    assert data.train_datasets.pushshapes_sim.resolver.folder_path.endswith(
        "/circle_4500"
    )
    assert data.train_datasets.pushshapes_sim_u_socket.resolver.folder_path.endswith(
        "/u_socket_1000"
    )
    assert (
        data.train_datasets.pushshapes_sim_u_socket.resolver.embodiment_override
        == "pushshapes_sim_u_socket"
    )


def test_xy_overlay_accepts_theta_actions():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    actions = np.array([[2.0, 3.0, -0.5], [4.0, 5.0, 0.5]], dtype=np.float32)
    rendered = _draw_chunk(frame, actions, scale=1.0, palette="Greens")
    assert rendered.shape == frame.shape
