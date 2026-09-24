"""populate_from_datasets probes one episode per key_map/transform group."""

from fixtures.synthetic_episodes import write_episode

from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.zarr.zarr_dataset_multi import (
    LocalEpisodeResolver,
    MultiDataset,
    ZarrDataset,
)


def _dataset(tmp_path, n=4):
    for i in range(n):
        write_episode(tmp_path, "eva", seed=i)
    resolver = LocalEpisodeResolver(
        tmp_path,
        key_map=Eva.get_keymap(keymap_mode="cartesian"),
        transform_list=Eva.get_transform_list(mode="cartesian_wristframe_6d"),
    )
    return MultiDataset._from_resolver(resolver, mode="total")


def _count_probes(monkeypatch, fail_first=0):
    calls = []
    orig = ZarrDataset.__getitem__

    def spy(self, idx, *a, **k):
        calls.append(self.episode_path)
        if len(calls) <= fail_first:
            raise RuntimeError("bad episode")
        return orig(self, idx, *a, **k)

    monkeypatch.setattr(ZarrDataset, "__getitem__", spy)
    return calls


def test_one_probe_per_group_gives_the_per_leaf_inventory(tmp_path, monkeypatch):
    ds = _dataset(tmp_path)
    expected = {
        k for k in ds[0] if k in ds.datasets[next(iter(ds.datasets))].key_map
    } | {"actions_cartesian", "observations.state.ee_pose"}
    calls = _count_probes(monkeypatch)
    stats = MultiDataset(state={})
    stats.populate_from_datasets({"eva_bimanual": ds})
    assert len(calls) == 1
    (emb,) = stats.key_types
    assert expected <= set(stats.key_types[emb])
    assert stats.key_types[emb]["actions_cartesian"] == "action_keys"
    assert stats.zarr_keys[emb]["actions_cartesian"] == "actions_cartesian"


def test_a_failed_probe_tries_the_next_episode(tmp_path, monkeypatch):
    ds = _dataset(tmp_path)
    calls = _count_probes(monkeypatch, fail_first=2)
    stats = MultiDataset(state={})
    stats.populate_from_datasets({"eva_bimanual": ds})
    assert len(calls) == 3 and len(set(calls)) == 3
    (emb,) = stats.key_types
    assert stats.zarr_keys[emb]["actions_cartesian"] == "actions_cartesian"
