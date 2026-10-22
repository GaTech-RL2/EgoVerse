"""The ingestion store against real R2: what the in-memory client in
tests/unit/test_ingest_store.py cannot prove.

    EGOVERSE_TEST_R2=1 pytest --integration tests/integration/test_ingest_r2.py

Everything lives under one disposable prefix, _test/ingest-<time>-<random>/ in the rldb
bucket (R2_* credentials from ~/.egoverse_env), and is deleted afterwards, pass or fail:
every object under it and every unfinished multipart upload. About 80 MB of traffic.
"""

import hashlib
import os
import uuid
from pathlib import Path

import pytest

from egomimic.ingest.store import R2Store, Ref, now

MiB = 1024 * 1024
BUCKET = "rldb"
REF = Ref("test-ci", "episode_0001")


def _sweep(s3, root: str) -> int:
    """Delete everything under `root`; returns how much was left to delete."""
    left = 0
    for page in s3.get_paginator("list_multipart_uploads").paginate(
        Bucket=BUCKET, Prefix=root
    ):
        for up in page.get("Uploads", []):
            s3.abort_multipart_upload(
                Bucket=BUCKET, Key=up["Key"], UploadId=up["UploadId"]
            )
            left += 1
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=root
    ):
        for obj in page.get("Contents", []):
            s3.delete_object(Bucket=BUCKET, Key=obj["Key"])
            left += 1
    return left


@pytest.fixture(scope="module")
def r2():
    if os.environ.get("EGOVERSE_TEST_R2") != "1":
        pytest.skip("set EGOVERSE_TEST_R2=1 to run against the real rldb bucket")
    from egomimic.utils.aws.aws_data_utils import get_boto3_s3_client

    s3 = get_boto3_s3_client()
    root = f"_test/ingest-{now()[:15]}-{uuid.uuid4().hex[:8]}/"
    try:
        yield R2Store(BUCKET, client=s3, root=root), s3, root
    finally:
        removed = _sweep(s3, root)
        objs = s3.list_objects_v2(Bucket=BUCKET, Prefix=root).get("KeyCount", 0)
        ups = s3.list_multipart_uploads(Bucket=BUCKET, Prefix=root).get("Uploads", [])
        print(
            f"\ncleanup: removed {removed} items under {root}; left {objs} + {len(ups)}"
        )
        assert objs == 0 and not ups, f"{root} not empty after cleanup"


def _put_part(store, name, upload_id, n, data):
    """PUT one part to its presigned URL, as a browser would; returns the ETag header."""
    import requests

    r = requests.put(store.part_url(REF, name, upload_id, n, expires_s=600), data=data)
    r.raise_for_status()
    return r.headers["ETag"]


def test_presigned_multipart_round_trip_and_correction(r2, tmp_path):
    from botocore.exceptions import ClientError

    store, s3, root = r2
    episode = os.urandom(5 * MiB) + os.urandom(1000)  # a >=5 MiB part, a small last one
    labels = b"labels " * 100

    # 1-4. episode.mcap in two presigned parts, completed with the ETags as returned.
    uid = store.start_upload(REF, "episode.mcap")
    parts = [
        (1, _put_part(store, "episode.mcap", uid, 1, episode[: 5 * MiB])),
        (2, _put_part(store, "episode.mcap", uid, 2, episode[5 * MiB :])),
    ]
    assert all(e.startswith('"') and e.endswith('"') for _, e in parts), parts
    store.finish_upload(REF, "episode.mcap", uid, parts)

    # 5-6. annotation.mcap (one presigned part), then finalize with both names.
    uid = store.start_upload(REF, "annotation.mcap")
    store.finish_upload(
        REF,
        "annotation.mcap",
        uid,
        [(1, _put_part(store, "annotation.mcap", uid, 1, labels))],
    )
    manifest = store.mark_complete(
        REF, {"by": "test"}, ["episode.mcap", "annotation.mcap"]
    )
    assert set(manifest) == {"episode.mcap", "annotation.mcap"}
    assert manifest["episode.mcap"][1].endswith(
        "-2"
    )  # R2's multipart ETag: <md5>-<parts>

    # 7. listed complete, downloads byte-exact.
    (d,) = store.list_deliveries("test-ci")
    assert d.complete and d.files == manifest
    out = store.download(d, tmp_path / "dl")
    assert (out / "episode.mcap").read_bytes() == episode
    assert (out / "annotation.mcap").read_bytes() == labels

    # 8-9. a correction with only episode.mcap: annotation.mcap is really gone.
    uid = store.start_upload(REF, "episode.mcap")
    fixed = b"corrected episode"
    store.finish_upload(
        REF, "episode.mcap", uid, [(1, _put_part(store, "episode.mcap", uid, 1, fixed))]
    )
    store.mark_complete(REF, {}, ["episode.mcap"])
    (d,) = store.list_deliveries("test-ci")
    assert d.complete and set(d.files) == {"episode.mcap"}
    with pytest.raises(ClientError) as e:
        s3.head_object(Bucket=BUCKET, Key=root + REF.file_key("annotation.mcap"))
    assert e.value.response["Error"]["Code"] in ("404", "NoSuchKey")
    assert store.download(d, tmp_path / "dl")  # replaced dest: exactly the new files
    assert sorted(p.name for p in (tmp_path / "dl").iterdir()) == ["episode.mcap"]

    # 10. start and abort another upload.
    uid = store.start_upload(REF, "extra.bin")
    store.abort_upload(REF, "extra.bin", uid)
    assert store.uploads_in_progress(REF) == []

    # 11. R2's error code for a gone upload, and stale cleanup treating it as done.
    with pytest.raises(ClientError) as e:
        s3.abort_multipart_upload(
            Bucket=BUCKET, Key=root + REF.file_key("extra.bin"), UploadId=uid
        )
    assert e.value.response["Error"]["Code"] == "NoSuchUpload"
    uid = store.start_upload(REF, "stale.bin")
    s3.abort_multipart_upload(  # gone behind the record's back (e.g. lifecycle rule)
        Bucket=BUCKET, Key=root + REF.file_key("stale.bin"), UploadId=uid
    )
    (rec,) = store.uploads_in_progress(REF)
    store.clear_upload(REF, rec["record"])
    assert store.uploads_in_progress(REF) == []


def test_r2_enforces_the_minimum_part_size(r2):
    from botocore.exceptions import ClientError

    store, _, _ = r2
    ref = Ref("test-ci", "too_small")
    uid = store.start_upload(ref, "x.bin")
    parts = [
        (1, store.upload_part(ref, "x.bin", uid, 1, os.urandom(1 * MiB))),  # < 5 MiB
        (2, store.upload_part(ref, "x.bin", uid, 2, os.urandom(1 * MiB))),
    ]
    with pytest.raises(ClientError) as e:
        store.finish_upload(ref, "x.bin", uid, parts)
    assert e.value.response["Error"]["Code"] == "EntityTooSmall"
    store.abort_upload(ref, "x.bin", uid)


def test_internal_upload_delivery_takes_the_managed_multipart_path(r2, tmp_path):
    store, _, _ = r2
    folder = tmp_path / "episode_0002"
    folder.mkdir()
    big = os.urandom(70 * MiB)  # over boto3's 64 MiB threshold: multipart
    (folder / "episode.mcap").write_bytes(big)
    (folder / "annotation.mcap").write_bytes(b"labels")
    ref = Ref("test-ci", "episode_0002")
    manifest = store.upload_delivery(ref, folder, {"by": "lab"})
    assert manifest["episode.mcap"][1].endswith("-2")  # two 64 MiB-or-less parts
    (d,) = [x for x in store.list_deliveries("test-ci") if x.ref == ref]
    assert d.complete and d.files == manifest
    out = store.download(d, tmp_path / "dl2")
    assert (
        hashlib.sha256((out / "episode.mcap").read_bytes()).digest()
        == hashlib.sha256(big).digest()
    )
    assert Path(out / "annotation.mcap").read_bytes() == b"labels"
