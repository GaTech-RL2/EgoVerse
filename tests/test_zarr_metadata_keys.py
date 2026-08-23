import numpy as np
import pytest

from egomimic.rldb.zarr.zarr_dataset_multi import resolve_metadata_value


def test_resolve_metadata_value_supports_dotted_paths_and_copies_values():
    metadata = {"intrinsics": {"front_1": [[1.0, 0.0], [0.0, 1.0]]}}

    value = resolve_metadata_value(metadata, "intrinsics.front_1")

    np.testing.assert_array_equal(value, metadata["intrinsics"]["front_1"])
    value[0][0] = 9.0
    assert metadata["intrinsics"]["front_1"][0][0] == 1.0


def test_resolve_metadata_value_reports_missing_component():
    with pytest.raises(KeyError, match="front_1"):
        resolve_metadata_value({"intrinsics": {}}, "intrinsics.front_1")
