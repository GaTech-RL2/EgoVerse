import importlib.util
import json
import os
import sys
import time

import numpy as np
import zarr
from fixtures.recipes import (
    RECIPES,
    common_overrides,
    cpu_trainer_overrides,
    hpt_small_overrides,
)
from fixtures.synthetic_episodes import write_episode
from fixtures.train_harness import compose_recipe, hermetic_env, write_fixtures
from omegaconf import OmegaConf

import egomimic.trainHydra as train_hydra
from egomimic.rldb.zarr import episode_norm_samples, norm_cache
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.rldb.zarr.zarr_writer import ZarrWriter

CFG = OmegaConf.create(
    {
        "_target_": "egomimic.rldb.zarr.zarr_dataset_multi.MultiDataset._from_resolver",
        "resolver": {
            "_target_": "egomimic.rldb.zarr.zarr_dataset_multi.S3EpisodeResolver",
            "folder_path": "/data/a",
            "bucket_name": "rldb",
            "main_prefix": "processed_v3",
            "key_map": {
                "_target_": "egomimic.rldb.embodiment.human.Human.get_keymap",
                "keymap_mode": "cartesian",
            },
            "transform_list": {
                "_target_": "egomimic.rldb.embodiment.human.Human.get_transform_list",
                "mode": "cartesian",
                "stride": 3,
            },
        },
        "filters": {
            "_target_": "egomimic.rldb.filters.DatasetFilter",
            "episode_hashes": ["h1"],
        },
        "mode": "total",
    }
)


def _key(**kw):
    args = dict(
        dataset_name="human_bimanual",
        episodes={"h2": "fp2", "h1": "fp1"},
        dataset_cfg=CFG,
        sample_frac=1.0,
    )
    args.update(kw)
    return norm_cache.norm_cache_key(norm_cache.cache_inputs(**args))


def test_key_ignores_where_data_lives_but_not_what_it_is() -> None:
    base = _key()
    assert len(base) == 64
    assert _key(episodes={"h1": "fp1", "h2": "fp2"}) == base
    moved = OmegaConf.merge(
        CFG,
        {"resolver": {"folder_path": "/elsewhere", "bucket_name": "other", "debug": 2}},
    )
    assert _key(dataset_cfg=moved) == base
    assert _key(episodes={"h1": "fp1"}) != base
    assert _key(episodes={"h2": "fp2", "h1": "re-exported"}) != base
    assert _key(sample_frac=0.5) != base
    strided = OmegaConf.merge(CFG, {"resolver": {"transform_list": {"stride": 1}}})
    assert _key(dataset_cfg=strided) != base
    assert _key(dataset_cfg=OmegaConf.merge(CFG, {"mode": "train"})) != base


def _resolver(name):
    return OmegaConf.merge(
        CFG,
        {"resolver": {"_target_": f"egomimic.rldb.zarr.zarr_dataset_multi.{name}"}},
    )


def test_key_depends_on_the_dataset_class_the_resolver_loads() -> None:
    """S3/Local variants load the same dataset class and share entries; the
    annotation-cutoff resolvers clamp action chunks, so they must not."""
    base = _key()
    assert _key(dataset_cfg=_resolver("LocalEpisodeResolver")) == base
    cutoff = _key(dataset_cfg=_resolver("S3AnnotationCutoffEpisodeResolver"))
    assert cutoff != base
    assert _key(dataset_cfg=_resolver("LocalAnnotationCutoffEpisodeResolver")) == cutoff


def test_key_changes_with_code(monkeypatch) -> None:
    base = _key()
    monkeypatch.setattr(norm_cache, "code_hash", lambda recipe: "edited")
    assert _key() != base


def test_code_modules_cover_reader_targets_and_their_imports() -> None:
    mods = norm_cache.code_modules(OmegaConf.to_container(CFG))
    for m in (
        "egomimic.rldb.zarr.zarr_dataset_multi",  # reader, resolvers, stats methods
        "egomimic.rldb.embodiment.human",  # keymap + transform factory (targets)
        "egomimic.rldb.filters",
        "egomimic.rldb.zarr.action_chunk_transforms",  # imported by human
        "egomimic.utils.pose_utils",  # imported by action_chunk_transforms
    ):
        assert m in mods
    assert all(m.startswith("egomimic.") for m in mods)


def _probe_module(monkeypatch, directory, source):
    """Import ``source`` as egomimic._norm_cache_probe from ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "probe.py"
    path.write_text(source)
    name = "egomimic._norm_cache_probe"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setitem(sys.modules, name, mod)
    norm_cache._module_source_hash.cache_clear()
    return {"_target_": f"{name}.f"}


def test_code_hash_changes_when_target_source_changes(tmp_path, monkeypatch) -> None:
    recipe = _probe_module(monkeypatch, tmp_path / "a", "def f():\n    return 1\n")
    assert "egomimic._norm_cache_probe" in norm_cache.code_modules(recipe)
    base = norm_cache.code_hash(recipe)
    assert base != norm_cache.code_hash({})
    # Same source from another directory: no paths in the hash.
    _probe_module(monkeypatch, tmp_path / "b", "def f():\n    return 1\n")
    assert norm_cache.code_hash(recipe) == base
    _probe_module(monkeypatch, tmp_path / "a", "def f():\n    return 2\n")
    assert norm_cache.code_hash(recipe) != base


def test_episode_fingerprint_changes_when_episode_is_rewritten(tmp_path) -> None:
    write_episode(tmp_path, "aria", seed=0)
    ep = tmp_path / "aria_00.zarr"
    before = norm_cache.episode_fingerprint(ep)
    assert before is not None
    assert norm_cache.episode_fingerprint(ep) == before
    zarr.open_group(str(ep), mode="r+").attrs["re_exported"] = True
    assert norm_cache.episode_fingerprint(ep) != before
    assert norm_cache.episode_fingerprint(tmp_path / "missing.zarr") is None
    assert norm_cache.episode_fingerprint(None) is None


def test_episode_fingerprint_changes_when_annotations_are_rewritten(tmp_path) -> None:
    """Re-annotating in place (scale_to_zarr_annotation.py) and adding a key
    (zarr_key_transform.py) touch only a child array, not the root zarr.json."""
    write_episode(tmp_path, "aria", seed=0)
    ep = tmp_path / "aria_00.zarr"
    writer = ZarrWriter(episode_path=ep)
    seen = [norm_cache.episode_fingerprint(ep)]
    for mode, annotations in [
        ("w", [("pick", 0, 5)]),
        ("w", [("place", 2, 9)]),  # same shape: only the rewrite itself shows
        ("a", [("x", 0, 1)]),
    ]:
        time.sleep(0.02)  # past the filesystem's mtime granularity
        key = "annotations" if mode == "w" else "annotations_v2"
        writer.append_annotations(key, annotations, mode=mode)
        seen.append(norm_cache.episode_fingerprint(ep))
    assert len(set(seen)) == len(seen)


def test_write_then_find_roundtrip(tmp_path) -> None:
    key = _key()
    stats = {"actions": {"mean": np.zeros(3), "std": np.ones(3)}}
    assert norm_cache.find_cached(tmp_path, "human_bimanual", key, 3) is None
    p = norm_cache.write_cached(
        tmp_path,
        "human_bimanual",
        key,
        {"dataset": "human_bimanual"},
        3,
        stats,
        {"frames": 10},
    )
    assert p == tmp_path / "human_bimanual" / f"{key}.json"
    assert norm_cache.find_cached(tmp_path, "human_bimanual", key, 3) == p
    assert (
        norm_cache.find_cached(tmp_path, "human_bimanual", key, 6) is None
    )  # wrong embodiment id
    payload = json.loads(p.read_text())
    assert payload["stats"]["3"]["actions"]["std"] == [1.0, 1.0, 1.0]
    assert payload["norm_run_metadata"] == {"frames": 10}
    assert payload["cache"]["key"] == key and payload["cache"]["inputs"] == {
        "dataset": "human_bimanual"
    }
    assert not list(tmp_path.rglob("*.tmp"))


def test_find_cached_rejects_non_dict_payload(tmp_path) -> None:
    """A JSON payload that is not an object (list/string) must read as
    unreadable -- warn and recompute -- not blow up the run."""
    key = _key()
    p = norm_cache.cache_path(tmp_path, "human_bimanual", key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("[]")
    assert norm_cache.find_cached(tmp_path, "human_bimanual", key, 3) is None
    p.write_text('"nope"')
    assert norm_cache.find_cached(tmp_path, "human_bimanual", key, 3) is None


def test_write_cached_tmp_name_is_per_writer(tmp_path, monkeypatch) -> None:
    """Concurrent writers (every DDP rank, or two jobs sharing the cache dir,
    possibly on nodes that reuse PIDs) must not share one tmp file."""
    seen = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append(str(src))
        return real_replace(src, dst)

    monkeypatch.setattr(norm_cache.os, "replace", spy)
    for _ in range(2):
        norm_cache.write_cached(
            tmp_path,
            "human_bimanual",
            _key(),
            {},
            3,
            {"actions": {"mean": np.zeros(3)}},
            None,
        )
    assert len(set(seen)) == 2
    assert not list(tmp_path.rglob("*.tmp"))


def test_write_cached_never_raises_on_io_error(tmp_path, monkeypatch) -> None:
    """A cache write failure is logged and returns None; it must not fail a run."""
    blocked = tmp_path / "blocked"
    blocked.write_text("i am a file, not a directory")
    assert (
        norm_cache.write_cached(
            blocked,
            "human_bimanual",
            _key(),
            {},
            3,
            {"actions": {"mean": np.zeros(3)}},
            None,
        )
        is None
    )

    def fail(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(norm_cache.os, "replace", fail)
    args = ("human_bimanual", _key(), {}, 3, {"actions": {"mean": np.zeros(3)}}, None)
    assert norm_cache.write_cached(tmp_path, *args) is None
    assert not list(tmp_path.rglob("*.tmp"))  # no orphaned tmp file


def _cfg(tmp_path, cache_dir, data, out, hashes, extra=()):
    recipe = RECIPES[("aria", "hpt")]
    return compose_recipe(
        recipe,
        common_overrides(
            recipe.embodiment,
            data,
            out,
            batch_size=2,
            num_workers=0,
            episode_hashes=hashes,
        )
        + cpu_trainer_overrides(2)
        + hpt_small_overrides(recipe.embodiment)
        + [f"paths.cache_dir={cache_dir}", *extra],
        out,
    )


def test_train_writes_then_reuses_norm_cache(tmp_path, monkeypatch) -> None:
    hermetic_env(monkeypatch)
    cache_dir = tmp_path / "cache"
    data, out, hashes = write_fixtures(tmp_path, "aria")
    seen = []
    orig = MultiDataset.infer_norm_from_dataset
    monkeypatch.setattr(
        MultiDataset,
        "infer_norm_from_dataset",
        lambda self, *a, **k: seen.append(k.get("precomputed_norm_path"))
        or orig(self, *a, **k),
    )
    sampled = []
    orig_sample = episode_norm_samples._sample_episodes
    monkeypatch.setattr(
        episode_norm_samples,
        "_sample_episodes",
        lambda root, leaves, *a, **k: sampled.append(sorted(leaves))
        or orig_sample(root, leaves, *a, **k),
    )

    train_hydra.train(_cfg(tmp_path, cache_dir, data, out, hashes))
    files = list((cache_dir / "norm_stats" / "human_bimanual").glob("*.json"))
    assert len(files) == 1 and seen == [None]
    first = files[0].read_text()

    train_hydra.train(_cfg(tmp_path, cache_dir, data, out, hashes))
    assert str(seen[1]) == str(files[0])  # hit: loaded through the precomputed path
    assert files[0].read_text() == first  # not rewritten

    # Re-exported under the same hash: the episode fingerprint changes the key.
    zarr.open_group(str(data / f"{hashes[0]}.zarr"), mode="r+").attrs["note"] = "v2"
    train_hydra.train(_cfg(tmp_path, cache_dir, data, out, hashes))
    assert seen[2] is None
    assert len(list((cache_dir / "norm_stats" / "human_bimanual").glob("*.json"))) == 2
    assert sampled == [sorted(hashes), [hashes[0]]]  # only the re-export resampled

    train_hydra.train(_cfg(tmp_path, cache_dir, data, out, hashes[:2]))
    assert (
        len(list((cache_dir / "norm_stats" / "human_bimanual").glob("*.json"))) == 3
    )  # new episode set, new key
    assert len(sampled) == 2  # ...served from the per-episode samples


def test_common_overrides_keep_the_cache_out_of_the_checkout(tmp_path) -> None:
    data, out, hashes = write_fixtures(tmp_path, "aria")
    recipe = RECIPES[("aria", "hpt")]
    cfg = compose_recipe(
        recipe,
        common_overrides(recipe.embodiment, data, out, batch_size=2, num_workers=0),
        out,
    )
    assert cfg.norm_stats.cache_dir == f"{out}/cache/norm_stats"


def test_use_cache_false_writes_nothing(tmp_path, monkeypatch) -> None:
    hermetic_env(monkeypatch)
    cache_dir = tmp_path / "cache"
    data, out, hashes = write_fixtures(tmp_path, "aria")
    train_hydra.train(
        _cfg(tmp_path, cache_dir, data, out, hashes, ["norm_stats.use_cache=false"])
    )
    assert not cache_dir.exists()
