import pytest

from astra_reversal.libero_runner import shard_entries


def test_full_benchmark_shards_cover_each_frozen_episode_once():
    manifest = {
        "episodes": [
            f"task{task}:state{state}" for task in range(10) for state in range(50)
        ]
    }
    shards = [shard_entries(manifest, i, 10) for i in range(10)]
    flattened = [entry for shard in shards for entry in shard]
    assert len(flattened) == len(set(flattened)) == 500
    assert set(flattened) == set(manifest["episodes"])
    assert all(len(shard) == 50 for shard in shards)
    assert shard_entries(manifest) == manifest["episodes"]


@pytest.mark.parametrize("index,count", [(-1, 2), (2, 2), (0, 0), (0, 11), (True, 2)])
def test_invalid_or_empty_shards_are_rejected(index, count):
    with pytest.raises(ValueError, match="shard"):
        shard_entries({"episodes": list(range(10))}, index, count)
