"""End-to-end test of the RoboTwin Zarr reusable-corpus path.

Converts a synthetic RoboTwin task to Zarr, then loads it through the *standard*
EgoVerse data pipeline — `LocalEpisodeResolver` + `MultiDataset._from_resolver` +
`Eva` keymap/transform (no SQL/S3) — and replicates `trainHydra.py`'s
norm-stats wiring (populate -> infer -> cache -> set_norm_stats_from). This proves the
converted stores are a loadable `eva_bimanual` cartesian corpus and that the
`data/robotwin_local.yaml` recipe works, a stronger check than the single-episode
`ZarrDataset` test in `robotwin_to_zarr_test.py`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from egomimic.ricl.scripts.download_robotwin import make_synthetic_robotwin_task
from egomimic.ricl.scripts.robotwin_to_zarr import convert_task
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset

CHUNK = 15
N_EPS = 3
N_FRAMES = 60


@pytest.fixture(scope="module")
def zarr_dir(tmp_path_factory):
    src = tmp_path_factory.mktemp("rt_md_src")
    out = tmp_path_factory.mktemp("rt_md_zarr")
    make_synthetic_robotwin_task(
        str(src), task="pick_cube", n_episodes=N_EPS, n_frames=N_FRAMES, seed=3
    )
    convert_task(str(src), str(out))
    return str(out)


def _make_dataset(zarr_dir: str, *, norm_mode_keymap: bool):
    """Build a MultiDataset over the local zarr dir (mirrors data/robotwin_local.yaml).

    ``norm_mode_keymap`` drops camera/annotation keys (what trainHydra feeds to
    norm-stats inference)."""
    resolver = LocalEpisodeResolver(
        folder_path=Path(zarr_dir),
        key_map=Eva.get_keymap("cartesian_pi", norm_mode=norm_mode_keymap),
        transform_list=Eva.get_transform_list("cartesian", chunk_length=CHUNK),
    )
    return MultiDataset._from_resolver(
        resolver,
        filters=DatasetFilter(filter_lambdas=[]),
        mode="total",
        norm_mode="quantile",
    )


def test_multidataset_loads_offline(zarr_dir):
    md = _make_dataset(zarr_dir, norm_mode_keymap=False)
    assert len(md) == N_EPS * N_FRAMES  # one sample per frame, all episodes kept
    s = md[0]  # no norm stats set yet -> post-transform sample passes through
    act = np.asarray(s["actions_cartesian"])
    state = np.asarray(s["observations.state.ee_pose"])
    assert act.shape == (CHUNK, 14)
    assert state.shape[-1] == 14
    img = np.asarray(s["base_0_rgb"])
    assert img.ndim == 3 and 3 in img.shape
    assert s.get("embodiment") == get_embodiment_id("eva_bimanual")


def test_norm_stats_inferred_and_cached_offline(zarr_dir, tmp_path):
    md = _make_dataset(zarr_dir, norm_mode_keymap=False)
    train_datasets = {"eva_bimanual": md}

    # Replicate trainHydra.py's norm-stats wiring (stats-only MultiDataset).
    norm_stats = MultiDataset(state={}, norm_mode="quantile")
    norm_stats.populate_from_datasets(train_datasets)
    norm_stats.infer_shapes_from_batch(md[0])
    norm_dataset = _make_dataset(zarr_dir, norm_mode_keymap=True)
    norm_stats.infer_norm_from_dataset(
        norm_dataset,
        "eva_bimanual",
        sample_frac=1.0,
        num_workers=0,  # single-process for a deterministic test
    )
    save_dir = str(tmp_path / "stats")
    norm_stats.cache_stats(save_cache_dir=save_dir)

    # norm_stats.json written with stats for the eva_bimanual embodiment id
    out_path = os.path.join(save_dir, "norm_stats", "norm_stats.json")
    assert os.path.isfile(out_path)
    with open(out_path) as f:
        payload = json.load(f)
    emb_id = get_embodiment_id("eva_bimanual")
    assert str(emb_id) in payload["stats"]
    assert payload["stats"][str(emb_id)], "no per-key stats computed"

    # After wiring stats by reference, samples are normalized but keep their shape.
    md.set_norm_stats_from(norm_stats)
    s = md[0]
    assert np.asarray(s["actions_cartesian"]).shape == (CHUNK, 14)
    assert np.isfinite(np.asarray(s["observations.state.ee_pose"])).all()


def test_hydra_config_instantiates(zarr_dir):
    """`data/robotwin_local.yaml`'s dataset node instantiates over the local zarr
    (guards the config's _target_s / arg names against drift)."""
    import hydra
    from omegaconf import OmegaConf

    import egomimic

    cfg_path = os.path.join(
        os.path.dirname(egomimic.__file__),
        "hydra_configs",
        "data",
        "robotwin_local.yaml",
    )
    cfg = OmegaConf.load(cfg_path)
    node = cfg.train_datasets.eva_bimanual
    node.resolver.folder_path = zarr_dir  # override ${paths.dataset_dir}
    md = hydra.utils.instantiate(node)
    assert isinstance(md, MultiDataset)
    assert len(md) == N_EPS * N_FRAMES
    s = md[0]
    assert all(
        k in s
        for k in ("actions_cartesian", "observations.state.ee_pose", "base_0_rgb")
    )
