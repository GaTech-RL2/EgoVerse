import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.stages_flow import DPUNetHead
from egomimic.pipeline.stages_io import TargetBuilder
from egomimic.pipeline.stages_seq import VisualEncode
from egomimic.pl_utils.ema_callback import EMACallback
from egomimic.rldb.zarr.zarr_dataset_packed import ZarrDPWindowDataset
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset


class _CountingDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, sample, timestep, global_cond):
        self.calls += 1
        return torch.zeros_like(sample)


def _small_dp_head(inference_steps=3):
    head = DPUNetHead(
        cond_keys=["a_top"],
        cond_dims=[4],
        action_dim=2,
        chunk_len=16,
        num_train_timesteps=10,
        num_inference_steps=inference_steps,
        down_dims=[8, 16],
        n_groups=8,
        diffusion_step_embed_dim=8,
    )
    head.net = _CountingDenoiser()
    return head


def test_dp_sampler_is_not_run_during_training():
    head = _small_dp_head(inference_steps=3).train()
    batch = {
        "a_top": torch.randn(2, 4),
        "target": torch.randn(2, 16, 2),
        "embodiment": "pushshapes_sim",
    }
    out = head(batch)
    assert head.net.calls == 1
    assert "loss/dp" in out
    assert "pred_action" not in out


def test_dp_sampler_still_runs_during_eval():
    head = _small_dp_head(inference_steps=3).eval()
    batch = {
        "a_top": torch.randn(2, 4),
        "target": torch.randn(2, 16, 2),
        "embodiment": "pushshapes_sim",
    }
    out = head(batch)
    assert head.net.calls == 4  # one loss call + three inference steps
    assert out["pred_action"].shape == (2, 16, 2)


def test_prewindowed_target_keeps_current_observation_and_full_action_window():
    first = torch.arange(16, dtype=torch.float32)
    second = torch.arange(100, 116, dtype=torch.float32)
    actions = torch.cat([first, second])[:, None].repeat(1, 2)
    obs_history = torch.arange(32, dtype=torch.float32)[:, None]
    batch = {
        "actions": actions,
        "a_top": obs_history,
        "cu_seqlens": torch.tensor([0, 16, 32]),
        "max_seq_len": 16,
    }
    out = TargetBuilder(
        chunk_len=16, prewindowed=True, current_index=1
    )(batch)
    assert torch.equal(out["target"][0, :, 0], first)
    assert torch.equal(out["target"][1, :, 0], second)
    assert torch.equal(out["actions"][:, 0], torch.tensor([1.0, 101.0]))
    assert torch.equal(out["a_top"][:, 0], torch.tensor([1.0, 17.0]))
    assert torch.equal(out["cu_seqlens"], torch.tensor([0, 1, 2]))
    assert out["max_seq_len"] == 1


def test_pipeline_executes_dp_positions_one_through_eight():
    algo = SimpleNamespace(action_start=1, replan_every=8)
    chunk = torch.arange(16, dtype=torch.float32)[:, None]
    queued = PipelineAlgo._actions_from_chunk(algo, chunk)
    assert [int(action.item()) for action in queued] == list(range(1, 9))


def test_cotrain_loss_uses_window_count_weighting():
    algo = SimpleNamespace(loss_weight_by_samples=True)
    predictions = {
        "15_action_loss": torch.tensor(2.0),
        "17_action_loss": torch.tensor(6.0),
    }
    batch = {15: {"batch_size": 38}, 17: {"batch_size": 26}}
    losses = PipelineAlgo.compute_losses(algo, predictions, batch)
    assert torch.allclose(
        losses["action_loss"], torch.tensor((2.0 * 38 + 6.0 * 26) / 64)
    )


class _FakeEpisode:
    def __init__(self, length):
        self.total_frames = length

    def _read_span(self, start, end, episode_idx=None):
        values = torch.arange(start, end, dtype=torch.float32)[:, None]
        return {
            "actions": values,
            "front_img_1": values[:, :, None, None],
            "seq_len": end - start,
            "embodiment": 15,
            "episode_idx": episode_idx,
        }


def test_dp_window_dataset_matches_sequence_sampler_padding():
    ds = ZarrDPWindowDataset(
        {"episode": _FakeEpisode(10)}, horizon=4, pad_before=1, pad_after=2
    )
    assert len(ds) == 10
    assert torch.equal(ds[0]["actions"][:, 0], torch.tensor([0.0, 0.0, 1.0, 2.0]))
    assert torch.equal(ds[-1]["actions"][:, 0], torch.tensor([8.0, 9.0, 9.0, 9.0]))
    assert ds[0]["seq_len"] == 4


def test_dp_window_episode_split_matches_dp_rounding_contract():
    episodes = {f"ep{i:03d}": _FakeEpisode(20) for i in range(100)}
    train = ZarrDPWindowDataset._split_datasets(episodes, "train", 0.02, 42)
    valid = ZarrDPWindowDataset._split_datasets(episodes, "valid", 0.02, 42)
    assert len(train) == 98
    assert len(valid) == 2
    assert set(train).isdisjoint(valid)
    assert set(train) | set(valid) == set(episodes)


def test_dp_window_split_can_share_one_global_mask_across_datasets():
    first = {f"a{i:03d}": _FakeEpisode(20) for i in range(60)}
    second = {f"b{i:03d}": _FakeEpisode(20) for i in range(40)}
    valid_first = ZarrDPWindowDataset._split_datasets(
        first, "valid", 0.02, 42, split_offset=0, split_total_episodes=100
    )
    valid_second = ZarrDPWindowDataset._split_datasets(
        second, "valid", 0.02, 42, split_offset=60, split_total_episodes=100
    )
    assert len(valid_first) + len(valid_second) == 2


class _CaptureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, x):
        self.seen = x.detach().clone()
        return x.mean(dim=(-1, -2))


def test_visual_encode_can_match_dp_minus_one_to_one_image_range():
    encoder = _CaptureEncoder()
    stage = VisualEncode(
        "obs/image", "feat", encoder, input_scale=2.0, input_bias=-1.0
    )
    batch = {
        "obs/image": torch.tensor([[[[0.0, 1.0]]]]),
        "embodiment": "pushshapes_sim",
    }
    stage(batch)
    assert torch.equal(encoder.seen, torch.tensor([[[[-1.0, 1.0]]]]))


def test_shipped_normalizer_is_shared_and_matches_original_dp_ranges():
    path = (
        Path(__file__).parents[1]
        / "egomimic/hydra_configs/norm_stats/dp_c3kgen_sc3k_v2_exact.json"
    )
    payload = json.loads(path.read_text())
    assert payload["stats"]["15"] == payload["stats"]["17"]

    norm = MultiDataset(state={}, norm_mode="minmax")
    norm.norm_stats[15] = payload["stats"]["15"]
    norm.key_types[15] = {
        "actions": "action_keys",
        "pusher_pose": "proprio_keys",
    }
    norm.zarr_keys[15] = {
        "actions": "actions",
        "pusher_pose": "pusher_pose",
    }
    endpoints = torch.tensor([[0.0, 0.0], [511.5, 511.5]])
    out = norm.normalize(
        {"actions": endpoints, "pusher_pose": endpoints}, 15
    )
    expected = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    assert torch.allclose(out["actions"], expected, atol=1e-6)
    assert torch.allclose(out["pusher_pose"], expected, atol=1e-6)


def test_ema_uses_diffusion_policy_inverse_power_schedule():
    ema = EMACallback(
        power=0.75,
        inv_gamma=1.0,
        min_value=0.0,
        max_value=0.9999,
        update_after_step=0,
    )
    assert ema._decay_at(1) == 0.0
    expected = 1.0 - (1.0 + 99.0) ** -0.75
    assert ema._decay_at(100) == pytest.approx(expected)
