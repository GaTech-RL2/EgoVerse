"""Unit tests for whole-episode prompt sampling (prompt_dataset.py) and the
prompt collate. Builds small synthetic human_bimanual zarr episodes on disk
(no network), so it runs on CPU.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from egomimic.pl_utils.pl_data_utils import annotation_collate, prompt_collate
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.prompt_dataset import (
    EpisodePromptMultiDataset,
    build_episode_prompt,
    read_prompt_chunks,
)
from egomimic.rldb.zarr.zarr_dataset_multi import (
    LocalEpisodeResolver,
    MultiDataset,
)
from egomimic.rldb.zarr.zarr_writer import ZarrWriter

EMBODIMENT = "human_bimanual"
CHUNK = 30
MAX_LEN = 450
H, W = 48, 64
IMG_KEY = Human.VIZ_IMAGE_KEY  # observations.images.front_img_1
STATE_KEY = "observations.state.ee_pose"
ACTION_KEY = "actions_cartesian"

# (episode_hash, operator, n_frames)
EPISODES = [
    ("ep_a1", "opA", 100),
    ("ep_a2", "opA", 70),
    ("ep_a3", "opA", 45),
    ("ep_a4", "opA", 130),
    ("ep_b1", "opB", 61),
    ("ep_b2", "opB", 90),
    ("ep_b3", "opB", 55),
    ("ep_b4", "opB", 110),
    ("ep_c1", "opC", 80),  # single-episode operator: dropped everywhere
    ("ep_d1", "opD", 75),  # three episodes: enough for total, not for a split
    ("ep_d2", "opD", 66),
    ("ep_d3", "opD", 88),
]


def _unit_quats(rng, n):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _write_episode(root, name, operator, n_frames, seed):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_frames)[:, None]

    def pose(offset):
        return np.concatenate(
            [
                offset
                + 0.3 * t * rng.normal(size=(1, 3))
                + 0.01 * rng.normal(size=(n_frames, 3)),
                _unit_quats(rng, n_frames),
            ],
            axis=1,
        ).astype(np.float32)

    numeric = {
        "left.obs_ee_pose": pose(np.array([-0.2, 0.0, 0.3])),
        "right.obs_ee_pose": pose(np.array([0.2, 0.0, 0.3])),
        "obs_head_pose": pose(np.array([0.0, 0.0, 0.0])),
    }
    images = rng.integers(0, 255, size=(n_frames, H, W, 3), dtype=np.uint8)
    K = np.array([[300.0, 0, W / 2, 0], [0, 300.0, H / 2, 0], [0, 0, 1, 0]])
    ZarrWriter.create_and_write(
        root / name,
        numeric_data=numeric,
        image_data={"images.front_1": images},
        embodiment=EMBODIMENT,
        fps=30,
        task_name="cup_on_saucer",
        intrinsics={"front_1": K},
        metadata_override={"user_id": operator},
    )


@pytest.fixture(scope="module")
def episode_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("episodes")
    for i, (name, op, n) in enumerate(EPISODES):
        _write_episode(root, name, op, n, seed=i)
    return root


def _key_map():
    return Human.get_keymap("cartesian")


def _rollout_transforms():
    return Human.get_transform_list("cartesian", stride=1)


def _prompt_transforms(chunk_length=CHUNK):
    return Human.get_transform_list("cartesian", stride=1, chunk_length=chunk_length)


def _leaves(root):
    resolver = LocalEpisodeResolver(
        root, key_map=_key_map(), transform_list=_rollout_transforms()
    )
    return resolver.resolve(
        filters=DatasetFilter(["lambda row: row['task_name'] == 'cup_on_saucer'"])
    )


def _norm_stats():
    emb_id = get_embodiment_id(EMBODIMENT)
    keys = {
        IMG_KEY: "camera_keys",
        STATE_KEY: "proprio_keys",
        ACTION_KEY: "action_keys",
    }
    stats = {
        key: {
            # wide bounds: synthetic poses/angles must never trip outlier rejection
            "quantile_1": -10.0 * np.ones(12, dtype=np.float32),
            "quantile_99": 10.0 * np.ones(12, dtype=np.float32),
        }
        for key in (STATE_KEY, ACTION_KEY)
    }
    state = {
        "norm_mode": "quantile",
        "embodiments": [emb_id],
        "key_types": {emb_id: dict(keys)},
        "zarr_keys": {emb_id: {k: k for k in keys}},
        "shapes": {emb_id: {}},
        "norm_stats": {emb_id: stats},
    }
    return MultiDataset.from_state(state)


def _prompt_cfg(**over):
    cfg = dict(
        chunk_n_actions=CHUNK,
        max_sequence_length=MAX_LEN,
        prompt_stride=1,
        min_episodes_per_group=2,
        image_size=[32, 32],
        seed=0,
    )
    cfg.update(over)
    return cfg


def _dataset(root, mode="train", with_stats=True, **over):
    ds = EpisodePromptMultiDataset(
        datasets=_leaves(root),
        mode=mode,
        prompt=_prompt_cfg(**over),
        prompt_transform_list=_prompt_transforms(),
        valid_ratio=0.2,
    )
    if with_stats:
        ds.set_norm_stats_from(_norm_stats())
    return ds


# ---------------------------------------------------------------------------


def test_local_resolver_sets_metadata_row(episode_root):
    leaves = _leaves(episode_root)
    assert set(leaves) == {name for name, _, _ in EPISODES}
    row = leaves["ep_b1"].metadata_row
    assert row["task"] == "cup_on_saucer"
    assert row["operator"] == "opB"
    assert row["episode_hash"] == "ep_b1"


def test_groups_split_and_small_group_dropped(episode_root):
    train = _dataset(episode_root, mode="train")
    valid = _dataset(episode_root, mode="valid")
    # opC (1 episode) and opD (3: cannot give 2 train + 2 valid) are dropped
    # from the split modes; opA and opB (4 each) split 2 / 2.
    assert set(train.group_names) == {
        ("cup_on_saucer", "opA"),
        ("cup_on_saucer", "opB"),
    }
    assert set(valid.group_names) == set(train.group_names)
    for g in train.group_names:
        assert len(train._episodes_by_group[g]) >= 2
        assert len(valid._episodes_by_group[g]) >= 2
    assert set(train.datasets).isdisjoint(valid.datasets)
    assert set(train.datasets) | set(valid.datasets) == {
        n for n, op, _ in EPISODES if op in ("opA", "opB")
    }
    # total mode keeps opD (3 >= 2) and still drops opC
    total = _dataset(episode_root, mode="total", with_stats=False)
    assert ("cup_on_saucer", "opD") in total.group_names
    assert all(g[1] != "opC" for g in total.group_names)


def test_no_prompt_before_norm_stats(episode_root):
    ds = _dataset(episode_root, mode="total", with_stats=False)
    sample = ds[0]
    assert "prompt" not in sample
    assert "group_idx" in sample
    # default collate must still work (norm-stat inference path)
    batch = annotation_collate([ds[0], ds[1]])
    assert batch["group_idx"].shape == (2,)


def test_prompt_shapes_and_same_group(episode_root):
    ds = _dataset(episode_root, mode="total")
    lengths = {
        name: EPISODES[[e[0] for e in EPISODES].index(name)][2] for name in ds.datasets
    }
    for idx in range(0, len(ds), 37):
        sample = ds[idx]
        prompt = sample["prompt"]
        own_name, _ = ds.index_map[idx]
        prompt_name = ds.episode_names[sample["prompt_episode_idx"]]
        assert ds._group_of_episode[prompt_name] == ds._group_of_episode[own_name]
        assert sample["group_idx"] == ds._group_index[ds._group_of_episode[own_name]]
        P = math.ceil(lengths[prompt_name] / CHUNK)
        assert prompt["length"] == P
        img = prompt["obs"][IMG_KEY]
        assert img.shape == (P, 3, 32, 32) and img.dtype == torch.float32
        assert 0.0 <= img.min() and img.max() <= 1.0
        assert prompt["obs"][STATE_KEY].shape == (P, 12)
        assert prompt["action"].shape == (P, CHUNK, 12)
        assert torch.isfinite(prompt["action"]).all()


def test_never_prompts_with_own_episode(episode_root):
    ds = _dataset(episode_root, mode="total")
    for idx in range(0, len(ds), 11):
        sample = ds[idx]
        own_name, _ = ds.index_map[idx]
        assert ds.episode_names[sample["prompt_episode_idx"]] != own_name


def test_prompt_chunk_matches_rollout_frame_convention(episode_root):
    """A prompt chunk read at frame s with the rollout's window (30 raw
    frames -> 100 interpolated steps) equals the rollout sample at s."""
    leaves = _leaves(episode_root)
    leaf = leaves["ep_a1"]
    for s in (0, 17, 60):
        # Read the rollout's window (Human.ACTION_HORIZON = 30 raw frames) and
        # interpolate to the rollout's 100 steps: must equal leaf[s].
        chunks30 = read_prompt_chunks(
            leaf,
            chunk_n_actions=Human.ACTION_HORIZON,
            prompt_stride=1,
            transform_list=_prompt_transforms(chunk_length=100),
            image_size=None,
            action_key=ACTION_KEY,
            state_key=STATE_KEY,
            chunk_starts=np.array([s]),
            action_steps=100,
        )
        rollout = leaf[s]
        assert torch.allclose(chunks30["action"][0], rollout[ACTION_KEY], atol=1e-5)
        assert torch.allclose(
            chunks30["obs"][STATE_KEY][0], rollout[STATE_KEY], atol=1e-5
        )
        img = chunks30["obs"][IMG_KEY][0].float() / 255.0
        assert torch.allclose(img, rollout[IMG_KEY], atol=1.0 / 255.0)


def test_too_long_episode_raises(episode_root):
    with pytest.raises(ValueError, match="exceed max_sequence_length"):
        _dataset(episode_root, mode="total", max_sequence_length=100)


def test_collate_pads_to_batch_max_and_masks(episode_root):
    ds = _dataset(episode_root, mode="total")
    samples = [ds[i] for i in (0, 5, 200, 300)]
    lengths = [s["prompt"]["length"] for s in samples]
    batch = annotation_collate(samples)
    prompt = batch["prompt"]
    P_max = max(lengths)
    assert prompt["action"].shape == (4, P_max, CHUNK, 12)
    assert prompt["obs"][IMG_KEY].shape == (4, P_max, 3, 32, 32)
    assert prompt["obs"][STATE_KEY].shape == (4, P_max, 12)
    mask = prompt["metadata"]["mask"]
    assert mask.shape == (4, P_max) and mask.dtype == torch.bool
    for i, L in enumerate(lengths):
        assert int((~mask[i]).sum()) == L
        assert not mask[i, :L].any() and mask[i, L:].all()
        # padded chunks are zero
        assert (prompt["action"][i, L:] == 0).all()
    assert batch["group_idx"].shape == (4,)
    assert batch["prompt_episode_idx"].shape == (4,)
    assert batch[ACTION_KEY].shape[0] == 4


def test_sample_weights_balance_groups(episode_root):
    ds = _dataset(episode_root, mode="total", balance_by="group")
    w = ds.sample_weights()
    assert w.shape == (len(ds),)
    per_group = {}
    for i, (name, _) in enumerate(ds.index_map):
        g = ds._group_of_episode[name]
        per_group[g] = per_group.get(g, 0.0) + float(w[i])
    vals = list(per_group.values())
    # total mode keeps opA, opB and opD; every group gets the same total weight
    assert len(vals) == 3 and max(vals) - min(vals) < 1e-9
    assert (
        _dataset(episode_root, mode="total", balance_by="none").sample_weights() is None
    )


def test_build_episode_prompt_matches_dataset(episode_root):
    ds = _dataset(episode_root, mode="total")
    name = "ep_b2"
    from_ds = ds.build_prompt_for_episode(name)
    deploy = build_episode_prompt(
        episode_root / name,
        key_map=_key_map(),
        prompt_transform_list=_prompt_transforms(),
        norm_stats=_norm_stats(),
        chunk_n_actions=CHUNK,
        image_size=(32, 32),
    )
    assert deploy["action"].shape == (1, from_ds["length"], CHUNK, 12)
    assert torch.allclose(deploy["action"][0], from_ds["action"])
    assert torch.allclose(deploy["obs"][STATE_KEY][0], from_ds["obs"][STATE_KEY])
    assert torch.allclose(deploy["obs"][IMG_KEY][0], from_ds["obs"][IMG_KEY])
    assert deploy["metadata"]["mask"].shape == (1, from_ds["length"])
    assert not deploy["metadata"]["mask"].any()


def test_prompt_collate_direct():
    def p(L):
        return {
            "obs": {"img": torch.rand(L, 3, 4, 4), "st": torch.rand(L, 12)},
            "action": torch.rand(L, CHUNK, 12),
            "length": L,
        }

    out = prompt_collate([p(2), p(5), p(3)])
    assert out["action"].shape == (3, 5, CHUNK, 12)
    assert out["metadata"]["mask"].tolist() == [
        [False, False, True, True, True],
        [False] * 5,
        [False, False, False, True, True],
    ]
    assert out["metadata"]["length"].tolist() == [2, 5, 3]


def test_reduce_stats_to_last_dim():
    from egomimic.rldb.zarr.prompt_dataset import reduce_stats_to_last_dim

    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 100, 12)) * np.arange(1, 101)[None, :, None]
    per_step = MultiDataset._compute_stats_for_array(X)
    assert per_step["mean"].shape == (100, 12)
    red = reduce_stats_to_last_dim(per_step)
    for name, arr in red.items():
        assert arr.shape == (12,), name
    flat = MultiDataset._compute_stats_for_array(X.reshape(-1, 12))
    assert np.allclose(red["mean"], flat["mean"], atol=1e-6)
    assert np.allclose(red["std"], flat["std"], rtol=1e-4)
    assert np.allclose(red["min"], flat["min"]) and np.allclose(red["max"], flat["max"])
    # collapsed quantiles bound the pooled quantiles
    assert (red["quantile_1"] <= flat["quantile_1"] + 1e-6).all()
    assert (red["quantile_99"] >= flat["quantile_99"] - 1e-6).all()
    # per-dim stats pass through unchanged
    assert np.allclose(reduce_stats_to_last_dim(flat)["std"], flat["std"])


def test_heldout_operator_goes_entirely_to_valid(episode_root):
    train = _dataset(episode_root, mode="train", heldout_groups=["opB"])
    valid = _dataset(episode_root, mode="valid", heldout_groups=["opB"])
    held = ("cup_on_saucer", "opB")
    assert train.heldout_groups_resolved == [held]
    assert valid.heldout_groups_resolved == [held]
    assert held not in train.group_names
    assert held in valid.group_names
    # every episode of the held-out operator is in valid, none in train
    held_eps = {n for n, op, _ in EPISODES if op == "opB"}
    assert held_eps <= set(valid.datasets)
    assert held_eps.isdisjoint(train.datasets)
    # the seen operator still contributes >= 2 held-out episodes to valid
    assert len(valid._episodes_by_group[("cup_on_saucer", "opA")]) >= 2
    # per-sample tags; prompts stay within the operator, held out or not
    for idx in range(0, len(valid), 23):
        sample = valid[idx]
        name, _ = valid.index_map[idx]
        assert sample["operator_seen"] == int(name not in held_eps)
        prompt_name = valid.episode_names[sample["prompt_episode_idx"]]
        assert valid._group_of_episode[prompt_name] == valid._group_of_episode[name]
    for idx in range(0, len(train), 41):
        assert train[idx]["operator_seen"] == 1


def test_total_mode_tags_heldout_without_moving(episode_root):
    """total mode keeps every episode (nothing moved or dropped) but still
    resolves heldout_groups for the operator_seen tag."""
    held = ("cup_on_saucer", "opB")
    plain = _dataset(episode_root, mode="total", with_stats=False)
    total = _dataset(episode_root, mode="total", heldout_groups=["opB"])
    assert total.heldout_groups_resolved == [held]
    assert held in total.group_names
    assert set(total.datasets) == set(plain.datasets)
    assert total.group_names == plain.group_names
    held_eps = {n for n, op, _ in EPISODES if op == "opB"}
    seen_tags = {}
    for idx in range(0, len(total), 17):
        sample = total[idx]
        name, _ = total.index_map[idx]
        assert sample["operator_seen"] == int(name not in held_eps)
        seen_tags.setdefault(sample["operator_seen"], set()).add(name)
        # prompts still come from the sample's own operator
        prompt_name = total.episode_names[sample["prompt_episode_idx"]]
        assert total._group_of_episode[prompt_name] == total._group_of_episode[name]
    assert seen_tags[0] == held_eps
    assert seen_tags[1] == set(total.datasets) - held_eps
    # holding out every group is allowed in total mode (nothing to train on
    # is not a concern there): everything is tagged unseen
    all_held = _dataset(
        episode_root, mode="total", heldout_groups=["opA", "opB", "opD"]
    )
    assert all(all_held[i]["operator_seen"] == 0 for i in range(0, len(all_held), 50))
    # hydra passes lists as omegaconf ListConfig
    from omegaconf import OmegaConf

    via_cfg = _dataset(
        episode_root, mode="total", heldout_groups=OmegaConf.create(["opB"])
    )
    assert via_cfg.heldout_groups_resolved == [held]


def test_ignore_knob_skips_prompt_but_keeps_tags(episode_root):
    ds = _dataset(episode_root, mode="total", heldout_groups=["opB"], ignore=True)
    assert ds.get_ignore_prompt()
    tagged = _dataset(episode_root, mode="total", heldout_groups=["opB"])
    assert ds.group_names == tagged.group_names
    assert set(ds.datasets) == set(tagged.datasets)
    for idx in range(0, len(ds), 29):
        sample = ds[idx]
        assert "prompt" not in sample and "prompt_episode_idx" not in sample
        assert sample["operator_seen"] == tagged[idx]["operator_seen"]
        assert sample["group_idx"] == tagged[idx]["group_idx"]
    batch = annotation_collate([ds[0], ds[1]])
    assert "prompt" not in batch and batch["operator_seen"].shape == (2,)


def test_heldout_all_groups_raises(episode_root):
    with pytest.raises(ValueError, match="every group"):
        _dataset(episode_root, mode="train", heldout_groups=["opA", "opB"])


def test_exclude_self_false_rejected(episode_root):
    with pytest.raises(ValueError, match="exclude_self=False"):
        _dataset(episode_root, mode="total", with_stats=False, exclude_self=False)
