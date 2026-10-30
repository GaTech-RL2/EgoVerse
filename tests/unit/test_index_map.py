import pytest

from egomimic.rldb.zarr.zarr_dataset_multi import _IndexMap


def test_index_map_matches_per_sample_list():
    datasets = {"a": [0] * 3, "empty": [], "b": [0] * 2}
    im = _IndexMap(datasets)
    expected = [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1)]
    assert len(im) == 5
    assert [im[i] for i in range(5)] == expected
    assert list(im) == expected
    assert im[-1] == ("b", 1)
    assert im.ranges() == {"a": range(0, 3), "empty": range(3, 3), "b": range(3, 5)}
    with pytest.raises(IndexError):
        im[5]
