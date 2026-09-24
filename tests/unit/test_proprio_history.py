"""Proprio observation history: the last K proprio vectors, one token per step.

K = 1 must stay bit-for-bit today's pipeline -- same sample keys, same shapes,
same stem tokens, same sinusoidal PE -- so every check below has a K = 1 twin.
For K > 1 the dataset reads a backward window (front-padded with the first real
frame so the current frame is always ``[-1]``), the cartesian transforms treat
the leading axis as batch, the baseline's per-channel proprio stats broadcast
over it, and the stem replaces the input-side sinusoid with a learned
time-step embedding added after the projection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import egomimic
from egomimic.algo.hpt import HPT, HPTModel
from egomimic.models.hpt_nets import MLPPolicyStem
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.zarr.zarr_dataset_multi import (
    LocalEpisodeResolver,
    MultiDataset,
)
from egomimic.rldb.zarr.zarr_writer import ZarrWriter

T_FRAMES = 24
HORIZON = Human.ACTION_HORIZON
CHUNK = 100
PROPRIO_KEYS = ("left.obs_ee_pose", "right.obs_ee_pose")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _pose_series(T: int, offset: float) -> np.ndarray:
    """(T, 7) xyz+quat(wxyz) whose translation is a distinct ramp per frame.

    The shared synthetic fixture holds every pose CONSTANT over time (so the
    per-cell quantile bounds stay usable); history needs frames that differ,
    so this writes its own episode with an identity rotation and a ramp.
    """
    t = np.arange(T, dtype=np.float64)[:, None]
    xyz = offset + t * np.array([[0.001, 0.002, 0.003]])
    quat = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (T, 1))
    return np.concatenate([xyz, quat], axis=1)


def _write_varying_episode(root, *, T: int = T_FRAMES, seed: int = 0):
    rng = np.random.default_rng(seed)
    H = W = 64
    K = np.array(
        [[100.0, 0.0, W / 2, 0.0], [0.0, 100.0, H / 2, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    numeric = {
        "left.obs_ee_pose": _pose_series(T, 0.10 + seed),
        "right.obs_ee_pose": _pose_series(T, 0.20 + seed),
        "obs_head_pose": _pose_series(T, 0.30 + seed),
    }
    images = {"images.front_1": rng.integers(0, 255, (T, H, W, 3), dtype=np.uint8)}
    return ZarrWriter.create_and_write(
        root / f"varying_{seed:02d}.zarr",
        numeric_data=numeric,
        image_data=images,
        embodiment="human_bimanual",
        fps=30,
        task_name="synthetic",
        intrinsics={"front_1": K},
    )


def _leaf(ds: MultiDataset):
    return next(iter(ds.datasets.values()))


def _dataset(
    tmp_path,
    *,
    proprio_history: int = 1,
    history_stride: int = 1,
    transforms=None,
    n: int = 1,
):
    for i in range(n):
        _write_varying_episode(tmp_path, seed=i)
    key_map = Human.get_keymap(
        keymap_mode="cartesian",
        proprio_history=proprio_history,
        history_stride=history_stride,
    )
    resolver = LocalEpisodeResolver(
        tmp_path, key_map=key_map, transform_list=transforms or []
    )
    return MultiDataset._from_resolver(resolver, mode="total")


def _raw_poses(tmp_path, key: str) -> np.ndarray:
    """The written (T, 7) array for ``key``, straight off disk."""
    import zarr

    store = zarr.open_group(str(next(tmp_path.glob("*.zarr"))), mode="r")
    return np.asarray(store[key][:])


# ---------------------------------------------------------------------------
# 1. dataset: backward window, front padding, mask
# ---------------------------------------------------------------------------


def test_history_window_is_the_last_k_frames(tmp_path):
    leaf = _leaf(_dataset(tmp_path, proprio_history=4))
    for key in PROPRIO_KEYS:
        raw = _raw_poses(tmp_path, key)
        got = np.asarray(leaf[10][key])
        assert got.shape == (4, 7)
        np.testing.assert_allclose(got, raw[7:11], rtol=0, atol=1e-6)
        # The current frame is always last.
        np.testing.assert_allclose(got[-1], raw[10], rtol=0, atol=1e-6)


def test_history_stride_spaces_the_window_out(tmp_path):
    """K = 3, s = 5 reads frames idx-10, idx-5, idx -- 0.33 s of past at 30 fps
    instead of 0.07 s, with the current frame still last."""
    leaf = _leaf(_dataset(tmp_path, proprio_history=3, history_stride=5))
    for key in PROPRIO_KEYS:
        raw = _raw_poses(tmp_path, key)
        got = np.asarray(leaf[12][key])
        assert got.shape == (3, 7)
        np.testing.assert_allclose(got, raw[[2, 7, 12]], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(
        np.asarray(leaf[12]["proprio_history_mask"]), np.ones(3, dtype=np.float32)
    )


def test_strided_history_front_pads_at_the_episode_start(tmp_path):
    leaf = _leaf(_dataset(tmp_path, proprio_history=3, history_stride=5))
    raw = _raw_poses(tmp_path, PROPRIO_KEYS[0])
    # idx 7: frames 7 and 2 are real, idx-10 would be negative
    got = np.asarray(leaf[7][PROPRIO_KEYS[0]])
    np.testing.assert_allclose(got, raw[[2, 2, 7]], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(
        np.asarray(leaf[7]["proprio_history_mask"]),
        np.array([0, 1, 1], dtype=np.float32),
    )
    # idx 3: only the current frame is real
    np.testing.assert_array_equal(
        np.asarray(leaf[3]["proprio_history_mask"]),
        np.array([0, 0, 1], dtype=np.float32),
    )


def test_stride_one_is_the_contiguous_window(tmp_path):
    a = _leaf(_dataset(tmp_path, proprio_history=4))
    b = _leaf(_dataset(tmp_path, proprio_history=4, history_stride=1))
    for key in PROPRIO_KEYS:
        np.testing.assert_array_equal(np.asarray(a[10][key]), np.asarray(b[10][key]))


def test_history_mask_is_all_ones_away_from_the_episode_start(tmp_path):
    leaf = _leaf(_dataset(tmp_path, proprio_history=4))
    mask = np.asarray(leaf[10]["proprio_history_mask"])
    assert mask.shape == (4,)
    np.testing.assert_array_equal(mask, np.ones(4, dtype=np.float32))


def test_history_front_pads_by_repeating_the_first_real_frame(tmp_path):
    leaf = _leaf(_dataset(tmp_path, proprio_history=4))
    for key in PROPRIO_KEYS:
        raw = _raw_poses(tmp_path, key)
        got = np.asarray(leaf[1][key])
        assert got.shape == (4, 7)
        expected = np.stack([raw[0], raw[0], raw[0], raw[1]])
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)

    # NOTE: the brief's example mask for idx=1 reads [0, 0, 0, 1]; the rule it
    # states ("1 = real, 0 = front padding") gives [0, 0, 1, 1] -- frames 0 and
    # 1 are both real reads, only the two leading copies of frame 0 are
    # padding. The rule wins; the mask is informational only (it is dropped in
    # process_batch_for_training and the padded steps already look exactly like
    # a history-dropout sample).
    mask = np.asarray(leaf[1]["proprio_history_mask"])
    np.testing.assert_array_equal(mask, np.array([0, 0, 1, 1], dtype=np.float32))

    np.testing.assert_array_equal(
        np.asarray(leaf[0]["proprio_history_mask"]),
        np.array([0, 0, 0, 1], dtype=np.float32),
    )


def test_history_mask_is_float32(tmp_path):
    value = _leaf(_dataset(tmp_path, proprio_history=4))[3]["proprio_history_mask"]
    assert isinstance(value, torch.Tensor)
    assert value.dtype == torch.float32


def test_k1_sample_is_bit_for_bit_todays_sample(tmp_path):
    """The K = 1 keymap must produce exactly what a keymap with no
    ``proprio_history`` argument at all produces."""
    _write_varying_episode(tmp_path, seed=0)
    baseline = Human.get_keymap(keymap_mode="cartesian")
    with_arg = Human.get_keymap(keymap_mode="cartesian", proprio_history=1)
    assert with_arg == baseline
    assert not any("history" in spec for spec in with_arg.values())

    def _sample(key_map):
        resolver = LocalEpisodeResolver(tmp_path, key_map=key_map, transform_list=[])
        ds = MultiDataset._from_resolver(resolver, mode="total")
        return _leaf(ds)[5]

    a, b = _sample(baseline), _sample(with_arg)
    assert set(a) == set(b)
    assert "proprio_history_mask" not in a
    for key in PROPRIO_KEYS:
        assert tuple(a[key].shape) == (7,)
    for key, value in a.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, b[key])


def test_keys_without_history_keep_the_single_frame_read(tmp_path):
    """Only the proprio keys the model consumes get a time axis: the head pose
    (the frame everything is expressed in) and the action chunks do not."""
    leaf = _leaf(_dataset(tmp_path, proprio_history=4))
    sample = leaf[10]
    assert tuple(sample["obs_head_pose"].shape) == (7,)
    assert tuple(sample["left.action_ee_pose"].shape) == (HORIZON, 7)


# ---------------------------------------------------------------------------
# 2. transforms: a leading time axis is a batch axis
# ---------------------------------------------------------------------------


def _wristframe_batch(poses: dict[str, np.ndarray]) -> dict:
    rng = np.random.default_rng(0)
    chunk = _pose_series(HORIZON, 0.4)
    batch = {
        "obs_head_pose": _pose_series(1, 0.3)[0],
        "left.action_ee_pose": chunk.copy(),
        "right.action_ee_pose": chunk.copy() + 0.01,
        "action_pad_mask": np.ones(HORIZON, dtype=np.float32),
    }
    del rng
    batch.update({k: v.copy() for k, v in poses.items()})
    return batch


def _run_wristframe(batch: dict) -> dict:
    for t in Human.get_transform_list(
        mode="cartesian_wristframe_6d",
        stride=1,
        fix_left_wrist_convention=True,
        pad_proprio_gripper=True,
    ):
        batch = t.transform(batch)
    return batch


def test_wristframe_chain_maps_k_by_7_proprio_to_k_by_20():
    K = 4
    hist = {key: _pose_series(K, 0.1 * (i + 1)) for i, key in enumerate(PROPRIO_KEYS)}
    out = _run_wristframe(_wristframe_batch(hist))
    state = np.asarray(out["observations.state.ee_pose"])
    assert state.shape == (K, 20)

    current = {key: value[-1] for key, value in hist.items()}
    out1 = _run_wristframe(_wristframe_batch(current))
    state1 = np.asarray(out1["observations.state.ee_pose"])
    assert state1.shape == (20,)
    np.testing.assert_allclose(state[-1], state1, rtol=1e-6, atol=1e-8)


def test_wristframe_actions_are_unchanged_by_the_proprio_time_axis():
    """The action chunk is expressed in the CURRENT step's wrist frame, so it
    must not move when history is added next to it."""
    K = 4
    hist = {key: _pose_series(K, 0.1 * (i + 1)) for i, key in enumerate(PROPRIO_KEYS)}
    current = {key: value[-1] for key, value in hist.items()}
    a = _run_wristframe(_wristframe_batch(hist))["actions_cartesian"]
    b = _run_wristframe(_wristframe_batch(current))["actions_cartesian"]
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-8)


def test_dataset_and_transforms_compose_to_a_k_by_20_state(tmp_path):
    transforms = Human.get_transform_list(
        mode="cartesian_wristframe_6d",
        stride=1,
        fix_left_wrist_convention=True,
        pad_proprio_gripper=True,
    )
    hist = _leaf(_dataset(tmp_path, proprio_history=4, transforms=transforms))
    base = _leaf(_dataset(tmp_path, proprio_history=1, transforms=transforms))
    for idx in (0, 3, 11):
        h = hist[idx]["observations.state.ee_pose"]
        b = base[idx]["observations.state.ee_pose"]
        assert tuple(h.shape) == (4, 20)
        assert tuple(b.shape) == (20,)
        torch.testing.assert_close(h[-1], b, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. normalization / bounds over the time axis
# ---------------------------------------------------------------------------


def _norm_shell(key: str, width: int, *, lo=-1.0, hi=1.0):
    md = MultiDataset.__new__(MultiDataset)
    md.norm_mode = "quantile"
    md.norm_stats = {
        0: {
            key: {
                "quantile_1": np.full(width, lo, dtype=np.float32),
                "quantile_99": np.full(width, hi, dtype=np.float32),
            }
        }
    }
    md.zarr_keys = {0: {key: key}}
    md.key_types = {0: {key: "proprio_keys"}}
    md._warned_violations = set()
    return md


def test_per_channel_stats_broadcast_over_the_time_axis():
    key = "observations.state.ee_pose"
    md = _norm_shell(key, 20, lo=-2.0, hi=4.0)
    stats = md.norm_stats[0][key]
    g = torch.Generator().manual_seed(0)
    arr = torch.randn(4, 20, generator=g)
    out = md._apply_norm_one(arr, stats)
    assert out.shape == (4, 20)
    for k in range(4):
        torch.testing.assert_close(out[k], md._apply_norm_one(arr[k], stats))


def test_bounds_check_catches_a_violation_in_a_past_step():
    key = "observations.state.ee_pose"
    md = _norm_shell(key, 20)
    arr = np.zeros((4, 20), dtype=np.float32)
    assert md._check_bounds({"embodiment": 0, key: arr.copy()}, None, 0, "ep") is None
    bad = arr.copy()
    bad[0, 0] = 50.0  # oldest history step, translation channel
    assert md._check_bounds({"embodiment": 0, key: bad}, None, 0, "ep") is not None


# ---------------------------------------------------------------------------
# 4. the stem
# ---------------------------------------------------------------------------


def _stems(D: int, out: int, **kwargs):
    torch.manual_seed(0)
    a = MLPPolicyStem(input_dim=D, output_dim=out, widths=[16], **kwargs)
    torch.manual_seed(0)
    b = MLPPolicyStem(input_dim=D, output_dim=out, widths=[16])
    return a, b


def test_history_stem_shape_and_zero_init_matches_the_baseline_stem():
    D, out, B, K = 5, 7, 3, 3
    hist, base = _stems(D, out, history_len=K)
    x = torch.randn(B, K, 1, D)
    y = hist(x)
    assert y.shape == (B, K, 1, out)
    # time_embed is zero at init, so every step is today's projection.
    torch.testing.assert_close(y, base(x))
    for k in range(K):
        torch.testing.assert_close(y[:, k], base(x[:, k]))


def test_time_embed_is_learned_zero_init_and_per_step():
    D, out, K = 5, 7, 3
    hist, _ = _stems(D, out, history_len=K)
    assert hist.time_embed.shape == (K, out)
    assert hist.time_embed.requires_grad
    torch.testing.assert_close(hist.time_embed, torch.zeros(K, out))

    with torch.no_grad():
        hist.time_embed.copy_(
            torch.arange(K, dtype=torch.float32)[:, None].expand(K, out)
        )
    x = torch.randn(2, K, 1, D)
    y = hist(x)
    assert not torch.allclose(y[:, 0], y[:, 1])
    # ... and the shift is exactly the per-step embedding.
    hist.time_embed.data.zero_()
    y0 = hist(x)
    hist.time_embed.data.copy_(
        torch.arange(K, dtype=torch.float32)[:, None].expand(K, out)
    )
    torch.testing.assert_close(
        hist(x) - y0,
        torch.arange(K, dtype=torch.float32)[None, :, None, None].expand(2, K, 1, out),
    )


def test_history_len_one_stem_has_no_time_embed_and_is_todays_stem():
    D, out = 5, 7
    one, base = _stems(D, out, history_len=1)
    assert not hasattr(one, "time_embed")
    assert dict(one.named_parameters()).keys() == dict(base.named_parameters()).keys()
    x = torch.randn(4, 1, 1, D)
    torch.testing.assert_close(one(x), base(x))


def test_history_dropout_replaces_the_past_with_the_current_step_in_train_mode():
    D, out, B, K = 5, 7, 6, 4
    hist, _ = _stems(D, out, history_len=K, history_dropout=1.0)
    x = torch.randn(B, K, 1, D)
    hist.train()
    y = hist(x)
    # Every step equals the current one: exactly an episode-start sample after
    # front padding.
    for k in range(K):
        torch.testing.assert_close(
            y[:, k], y[:, -1] - hist.time_embed[-1] + hist.time_embed[k]
        )
    hist.eval()
    torch.testing.assert_close(y, hist(_current_only(x)))


def _current_only(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    out[:, :-1] = out[:, -1:]
    return out


def test_history_dropout_is_a_noop_in_eval_mode():
    D, out, B, K = 5, 7, 6, 4
    drop, _ = _stems(D, out, history_len=K, history_dropout=1.0)
    torch.manual_seed(0)
    plain = MLPPolicyStem(input_dim=D, output_dim=out, widths=[16], history_len=K)
    drop.eval()
    plain.eval()
    x = torch.randn(B, K, 1, D)
    torch.testing.assert_close(drop(x), plain(x))
    assert not torch.allclose(drop(x), drop(_current_only(x)))


def test_history_dropout_zero_is_a_noop_in_train_mode():
    D, out, B, K = 5, 7, 6, 4
    a, _ = _stems(D, out, history_len=K, history_dropout=0.0)
    torch.manual_seed(0)
    b = MLPPolicyStem(input_dim=D, output_dim=out, widths=[16], history_len=K)
    a.train()
    b.eval()
    x = torch.randn(B, K, 1, D)
    torch.testing.assert_close(a(x), b(x))


def test_history_stem_rejects_a_wrong_length_history():
    hist, _ = _stems(5, 7, history_len=3)
    with pytest.raises(ValueError, match="3"):
        hist(torch.randn(2, 4, 1, 5))


# ---------------------------------------------------------------------------
# 5. stem_process: the input-side sinusoid goes once history_len > 1
# ---------------------------------------------------------------------------


class _EchoStem:
    """Stem stub whose ``compute_latent`` hands back what stem_process fed it."""

    def __init__(self, history_len: int = 1):
        self.history_len = history_len
        self.seen = None

    def compute_latent(self, x):
        self.seen = x.clone()
        return x


def _stem_process(history_len: int, data_shape):
    model = HPTModel.__new__(HPTModel)
    model.modalities = {"human_bimanual": ["state_ee_pose"]}
    model.shared_keys = []
    model.encoders = {}
    stem = _EchoStem(history_len)
    model.stems = {"human_bimanual_state_ee_pose": stem}
    model.stem_spec = {
        "human_bimanual": {
            "state_ee_pose": OmegaConf.create(
                {"specs": {"random_horizon_masking": False}}
            )
        }
    }
    model.train_mode = False
    torch.manual_seed(0)
    x = torch.randn(*data_shape)
    model.stem_process("human_bimanual", {"state_ee_pose": x.clone()})
    return x, stem.seen


def test_stem_process_skips_the_sinusoid_for_a_history_stem():
    x, seen = _stem_process(4, (2, 4, 20))
    torch.testing.assert_close(seen, x)


def test_stem_process_keeps_the_sinusoid_for_a_history_len_one_stem():
    x, seen = _stem_process(1, (2, 1, 20))
    from egomimic.utils.tensor_utils import get_sinusoid_encoding_table

    pe = get_sinusoid_encoding_table(0, 1, 20).to(x)
    torch.testing.assert_close(seen, x + pe.view(1, 1, 20))
    assert not torch.allclose(seen, x)


# ---------------------------------------------------------------------------
# 6. HPT batch plumbing
# ---------------------------------------------------------------------------


class _NoRenameNormStats:
    def zarr_key_to_keyname(self, zarr_key, embodiment_id):
        return None


def _hpt_with_stem(history_len: int):
    algo = HPT.__new__(HPT)
    policy = HPTModel.__new__(HPTModel)
    policy.stems = {"human_bimanual_state_ee_pose": _EchoStem(history_len)}
    algo.nets = {"policy": policy}
    algo.is_6dof = True
    algo.shared_ac_key = None
    algo.annotation_modality = "annotation"
    algo.eval_image_augs = None
    return algo


def _to_hpt(algo, proprio: torch.Tensor):
    batch = {
        "observations.state.ee_pose": proprio,
        "pad_mask": torch.ones(proprio.shape[0], 5, 1),
        "embodiment": torch.tensor([0]),
        "actions_cartesian": torch.zeros(proprio.shape[0], 5, 18),
    }
    return HPT._robomimic_to_hpt_data(
        algo,
        batch,
        cam_keys=[],
        proprio_keys=["observations.state.ee_pose"],
        lang_keys=[],
        ac_key="actions_cartesian",
        domain="human_bimanual",
    )


def test_two_dim_proprio_is_still_unsqueezed():
    algo = _hpt_with_stem(1)
    data = _to_hpt(algo, torch.randn(3, 20))
    assert data["state_ee_pose"].shape == (3, 1, 20)


def test_three_dim_proprio_passes_through_without_unsqueeze():
    algo = _hpt_with_stem(4)
    x = torch.randn(3, 4, 20)
    data = _to_hpt(algo, x)
    assert data["state_ee_pose"].shape == (3, 4, 20)
    torch.testing.assert_close(data["state_ee_pose"], x)


def test_history_len_is_read_from_the_domain_not_by_suffix():
    """A cotrain run has one state_ee_pose stem per domain; a suffix match
    returns whichever comes first in the dict, so the human side would be
    checked against eva's history_len."""
    algo = _hpt_with_stem(3)
    algo.nets["policy"].stems = {
        "eva_bimanual_state_ee_pose": _EchoStem(1),
        "human_bimanual_state_ee_pose": _EchoStem(3),
    }
    data = _to_hpt(algo, torch.randn(3, 3, 20))
    assert data["state_ee_pose"].shape == (3, 3, 20)


def test_history_length_mismatch_raises_naming_both():
    algo = _hpt_with_stem(2)
    with pytest.raises(ValueError, match="state_ee_pose"):
        _to_hpt(algo, torch.randn(3, 4, 20))


def _process(action_pad_mask=None, history_mask=None, S: int = 6):
    algo = HPT.__new__(HPT)
    algo.norm_stats = _NoRenameNormStats()
    algo.device = "cpu"
    algo.annotation_key = None
    algo.use_pad_mask = False
    emb = "human_bimanual"
    emb_id = get_embodiment_id(emb)
    algo.ac_keys = {emb_id: "actions_cartesian"}
    inner = {
        "actions_cartesian": torch.zeros(2, S, 18),
        "observations.state.ee_pose": torch.zeros(2, 4, 20),
    }
    if action_pad_mask is not None:
        inner["action_pad_mask"] = action_pad_mask
    if history_mask is not None:
        inner["proprio_history_mask"] = history_mask
    return HPT.process_batch_for_training(algo, {emb: inner})[emb_id]


def test_process_batch_drops_the_history_mask():
    out = _process(history_mask=torch.ones(2, 4))
    assert "proprio_history_mask" not in out
    assert out["observations.state.ee_pose"].shape == (2, 4, 20)


def test_process_batch_is_unchanged_without_the_history_mask():
    out = _process()
    assert "proprio_history_mask" not in out
    torch.testing.assert_close(out["pad_mask"], torch.ones(2, 6, 1))


# ---------------------------------------------------------------------------
# 7. configs
# ---------------------------------------------------------------------------

RECIPE = "train_zarr_mecka_flagship_6d_hpt"
DATA = "data=mecka_fold_flagship_opsplit_hpt_6d"
KEYMAP = "data.train_datasets.human_bimanual.resolver.key_map"
STEM = "model.robomimic_model.stem_specs.human_bimanual.state_ee_pose"


def test_flagship_ships_k_three_stride_five(compose_resolve):
    cfg = compose_resolve(RECIPE, [DATA])
    km = cfg.data.train_datasets.human_bimanual.resolver.key_map
    assert km.proprio_history == 3
    assert km.history_stride == 5
    stem = cfg.model.robomimic_model.stem_specs.human_bimanual.state_ee_pose
    assert stem.history_len == km.proprio_history, "data K and stem K must agree"
    assert stem.history_dropout == 0.2


def test_k_one_baseline_is_still_reachable(compose_resolve):
    cfg = compose_resolve(
        RECIPE,
        [
            DATA,
            f"{KEYMAP}.proprio_history=1",
            f"{KEYMAP}.history_stride=1",
            f"{STEM}.history_len=1",
        ],
    )
    km = cfg.data.train_datasets.human_bimanual.resolver.key_map
    assert km.proprio_history == 1 and km.history_stride == 1
    assert (
        cfg.model.robomimic_model.stem_specs.human_bimanual.state_ee_pose.history_len
        == 1
    )


def test_k_two_override_reaches_every_loader(compose_resolve):
    cfg = compose_resolve(
        RECIPE, [DATA, f"{KEYMAP}.proprio_history=2", f"{STEM}.history_len=2"]
    )
    data = cfg.data
    for group in ("train_datasets", "valid_datasets", "unseen_op_valid_datasets"):
        km = data[group]["human_bimanual"]["resolver"]["key_map"]
        assert km["proprio_history"] == 2, group
    assert (
        cfg.model.robomimic_model.stem_specs.human_bimanual.state_ee_pose.history_len
        == 2
    )


def test_k_two_keymap_marks_only_the_model_proprio_keys():
    km = Human.get_keymap(keymap_mode="cartesian", proprio_history=2)
    assert {k for k, v in km.items() if "history" in v} == set(PROPRIO_KEYS)
    assert all(km[k]["history"] == 2 for k in PROPRIO_KEYS)


# ---------------------------------------------------------------------------
# 8. the eval path: the revert transforms and the evaluator take the current
#    step, so val metrics and the overlay video work at K > 1
# ---------------------------------------------------------------------------

REVERT_BUILDERS = [
    "_build_human_cartesian_revert_6d_wristframe_transform_list",
    "_build_human_cartesian_revert_6d_transform_list",
]


def _revert_inputs(B: int = 2, K: int = 3):
    """``(actions, current_proprio, history_proprio)`` whose last history step
    is exactly the 2-D proprio."""
    g = torch.Generator().manual_seed(7)
    actions = torch.randn(B, 100, 18, generator=g)
    current = torch.randn(B, 20, generator=g)
    history = torch.randn(B, K, 20, generator=g)
    history[:, -1] = current
    return actions, current, history


@pytest.mark.parametrize("builder_name", REVERT_BUILDERS)
def test_revert_transform_list_is_blind_to_proprio_history(builder_name):
    from egomimic.rldb.embodiment import human as human_mod
    from egomimic.rldb.embodiment.embodiment import Embodiment

    transform_list = getattr(human_mod, builder_name)()
    actions, current, history = _revert_inputs()

    def _run(proprio):
        return Embodiment.apply_transform(
            {
                "actions_cartesian": actions.clone(),
                "observations.state.ee_pose": proprio.clone(),
            },
            transform_list,
        )

    flat, hist = _run(current), _run(history)
    assert flat["observations.state.ee_pose"].shape == (2, 12)
    assert hist["observations.state.ee_pose"].shape == (2, 12)
    for key in ("actions_cartesian", "observations.state.ee_pose"):
        torch.testing.assert_close(hist[key], flat[key], rtol=1e-5, atol=1e-6)


def test_select_current_step_leaves_a_flat_value_alone_and_is_idempotent():
    from egomimic.rldb.zarr.action_chunk_transforms import SelectCurrentStep

    t = SelectCurrentStep(keys=["observations.state.ee_pose", "missing"])
    flat = np.arange(20, dtype=np.float32)
    out = t.transform({"observations.state.ee_pose": flat})
    assert out["observations.state.ee_pose"] is flat  # untouched, not a copy

    hist = np.stack([flat - 1, flat, flat + 1])
    once = t.transform({"observations.state.ee_pose": hist.copy()})
    np.testing.assert_array_equal(once["observations.state.ee_pose"], flat + 1)
    twice = t.transform(dict(once))
    np.testing.assert_array_equal(twice["observations.state.ee_pose"], flat + 1)


def test_every_human_revert_list_selects_the_current_proprio_step():
    from egomimic.rldb.embodiment import human as human_mod
    from egomimic.rldb.zarr.action_chunk_transforms import SelectCurrentStep

    builders = {
        "_build_human_cartesian_revert_6d_transform_list": "observations.state.ee_pose",
        "_build_human_cartesian_revert_6d_wristframe_transform_list": "observations.state.ee_pose",
        "_build_human_cartesian_revert_eef_frame_transform_list": "observations.state.ee_pose",
        "_build_human_keypoints_revert_6d_transform_list": "observations.state.keypoints",
        "_build_human_keypoints_revert_6d_wristframe_transform_list": "observations.state.keypoints",
        "_build_human_keypoints_revert_eef_frame_transform_list": "observations.state.keypoints",
    }
    for name, obs_key in builders.items():
        tl = getattr(human_mod, name)()
        assert isinstance(tl[0], SelectCurrentStep), name
        assert tl[0].keys == [obs_key], name


# --- the evaluator itself -----------------------------------------------


class _IdentityNormStats:
    def unnormalize(self, batch, embodiment_id):
        return dict(batch)


class _StubAlgo:
    """Just enough of HPT for HPTEvalVideo.compute_metrics_and_viz."""

    def __init__(self, emb_id, preds):
        self.norm_stats = _IdentityNormStats()
        self.ac_keys = {emb_id: "actions_cartesian"}
        self.shared_ac_key = None
        self.auxiliary_ac_keys = {}
        self._preds = preds

    def forward_eval(self, batch):
        return self._preds


def _eval_once(proprio):
    """Run the real HPTEvalVideo over one batch, returning its metrics and the
    batch the viz function was handed."""
    from egomimic.eval.eval_hpt import HPTEvalVideo
    from egomimic.rldb.embodiment.human import (
        _build_human_cartesian_revert_6d_wristframe_transform_list,
    )

    emb_id = get_embodiment_id("human_bimanual")
    emb_name = "human_bimanual"
    # Action horizon 1: the evaluator's paired/final MSE does
    # ``pred[:, -1].cpu()``, which is a real (contiguous) copy for the CUDA
    # tensors a run produces but a non-contiguous view for the CPU tensors a
    # test hands it, and torchmetrics' ``view(-1)`` rejects that. S = 1 keeps
    # the slice contiguous; the proprio path under test is unaffected.
    g = torch.Generator().manual_seed(11)
    actions = torch.randn(2, 1, 18, generator=g)
    preds = {
        f"{emb_name}_actions_cartesian": torch.randn(2, 1, 18, generator=g),
        f"{emb_name}_loss": torch.tensor(0.5),
    }
    batch = {
        emb_id: {
            "actions_cartesian": actions.clone(),
            "observations.state.ee_pose": proprio.clone(),
            "embodiment": torch.tensor([emb_id, emb_id]),
        }
    }

    seen = {}

    def _viz(predictions, viz_batch):
        seen["batch"] = viz_batch
        seen["preds"] = predictions
        return np.zeros((2, 4, 4, 3), dtype=np.uint8)

    ev = HPTEvalVideo.__new__(HPTEvalVideo)
    ev._replay_now = False
    ev.model = _StubAlgo(emb_id, preds)
    ev.viz_func = {emb_name: _viz}
    ev.transform_lists = {
        emb_name: _build_human_cartesian_revert_6d_wristframe_transform_list()
    }
    metrics, images = ev.compute_metrics_and_viz(batch, do_viz=True)
    return metrics, images, seen


def test_evaluator_metrics_and_viz_are_blind_to_proprio_history():
    _, current, history = _revert_inputs()
    m_flat, im_flat, seen_flat = _eval_once(current)
    m_hist, im_hist, seen_hist = _eval_once(history)

    assert set(m_flat) == set(m_hist)
    cam_keys = [k for k in m_flat if "_cam_" in k]
    assert cam_keys, "the cam-frame metrics must be produced"
    for key in m_flat:
        torch.testing.assert_close(
            torch.as_tensor(m_flat[key]).float(),
            torch.as_tensor(m_hist[key]).float(),
            rtol=1e-5,
            atol=1e-6,
        )
    # The viz function sees the reverted, current-step proprio either way.
    assert seen_flat["batch"]["observations.state.ee_pose"].shape == (2, 12)
    torch.testing.assert_close(
        seen_hist["batch"]["observations.state.ee_pose"],
        seen_flat["batch"]["observations.state.ee_pose"],
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        seen_hist["preds"]["human_bimanual_actions_cartesian"],
        seen_flat["preds"]["human_bimanual_actions_cartesian"],
        rtol=1e-5,
        atol=1e-6,
    )
    assert set(im_flat) == set(im_hist)


TRAIN_CONFIGS = sorted(
    p.stem
    for p in (Path(egomimic.__file__).parent / "hydra_configs").glob("train_*.yaml")
)


@pytest.mark.parametrize("top", TRAIN_CONFIGS)
def test_every_recipe_agrees_with_its_stem_on_K(top, compose_resolve):
    """The data's `proprio_history` and the stem's `history_len` must match, or
    process_batch_for_training raises at the first training step. A recipe that
    inherits a data config from one model and overrides the model is exactly
    how they drift apart, and nothing else composes both halves."""
    cfg = compose_resolve(top, [])
    stem_specs = (cfg.model.robomimic_model.get("stem_specs") or {}).get(
        "human_bimanual"
    ) or {}
    for split in ("train_datasets", "valid_datasets"):
        node = (cfg.data.get(split) or {}).get("human_bimanual")
        if node is None or node.get("resolver") is None:
            continue
        k = int(node.resolver.key_map.get("proprio_history", 1) or 1)
        for name, stem in stem_specs.items():
            if stem is None or "history_len" not in stem:
                continue
            assert int(stem.history_len) == k, (top, split, name, k, stem.history_len)
