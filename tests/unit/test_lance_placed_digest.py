import hashlib

import pyarrow as pa
import pytest

pytest.importorskip("lance")
from egomimic.rldb.lance import commit  # noqa: E402


def test_placed_digest_is_content_digest_and_hashes_each_file(tmp_path):
    steps = pa.table({"recording_id": ["r"] * 3, "timestamp_ns": [1, 2, 3]})
    files = {"front_1": b"a" * 3_000_000, "left_wrist": b"bcd"}
    paths = {}
    for cam, data in files.items():
        paths[cam] = tmp_path / f"{cam}.mp4"
        paths[cam].write_bytes(data)
    digest, shas = commit.placed_digest(steps, {c: str(p) for c, p in paths.items()})
    assert digest == commit.content_digest(steps, paths)
    assert shas == {c: hashlib.sha256(d).hexdigest() for c, d in files.items()}
    # Any changed byte changes the digest.
    paths["left_wrist"].write_bytes(b"bce")
    assert (
        commit.placed_digest(steps, {c: str(p) for c, p in paths.items()})[0] != digest
    )
