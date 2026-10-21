"""List episode hashes from a stub S3 client and read the listing back."""

from __future__ import annotations

import json

import pytest

from egomimic.scripts.sharpa_process import existing_hashes_s3 as lister

FIRST = "2026-05-06-02-44-02-000000"
SECOND = "2026-05-06-02-44-03-000000"
NESTED = "2026-05-07-10-00-00-000000"
#: Folder -> pages of (sub-prefixes, object keys) the stub serves.
TREE = {
    "processed_v3/": [
        (["processed_v3/sharpa_dex/", "processed_v3/other_lab/"], []),
        (["processed_v3/logs/"], ["processed_v3/README.md"]),
    ],
    "processed_v3/sharpa_dex/": [
        (
            [f"processed_v3/sharpa_dex/{SECOND}.zarr/"],
            [f"processed_v3/sharpa_dex/{SECOND}.mp4"],
        ),
        ([f"processed_v3/sharpa_dex/{FIRST}.zarr/"], []),
    ],
    "processed_v3/other_lab/": [(["processed_v3/other_lab/kitchen/"], [])],
    "processed_v3/other_lab/kitchen/": [
        ([f"processed_v3/other_lab/kitchen/{NESTED}.zarr/"], []),
    ],
    "processed_v3/logs/": [([], ["processed_v3/logs/run.txt"])],
}


class _Paginator:
    def __init__(self, calls):
        self.calls = calls

    def paginate(self, **kwargs):
        self.calls.append(kwargs)
        prefix = kwargs["Prefix"]
        assert not prefix.endswith(".zarr/"), f"descended into {prefix}"
        for prefixes, keys in TREE.get(prefix, [([], [])]):
            page = {"KeyCount": len(prefixes) + len(keys)}
            if prefixes:
                page["CommonPrefixes"] = [{"Prefix": p} for p in prefixes]
            if keys:
                page["Contents"] = [{"Key": k, "Size": 1} for k in keys]
            yield page


class StubClient:
    def __init__(self):
        self.calls = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _Paginator(self.calls)


def test_listing_records_zarr_stores_without_descending():
    client = StubClient()
    found = lister.list_episodes(client, "rldb", "processed_v3/")
    assert found == [
        (NESTED, "processed_v3/other_lab/kitchen/"),
        (FIRST, "processed_v3/sharpa_dex/"),
        (SECOND, "processed_v3/sharpa_dex/"),
    ]
    listed = sorted(call["Prefix"] for call in client.calls)
    assert listed == sorted(TREE)
    assert all(
        call["Delimiter"] == "/" and call["Bucket"] == "rldb" for call in client.calls
    )


def test_the_cli_writes_a_listing_that_reads_back(tmp_path, monkeypatch):
    from egomimic.utils.aws import aws_data_utils

    client = StubClient()
    monkeypatch.setattr(aws_data_utils, "get_boto3_s3_client", lambda: client)
    out = tmp_path / "s3_hashes.tsv"
    assert lister.main(["--prefix", "processed_v3", "--output", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert lines[0].startswith("# ")
    header = json.loads(lines[0][2:])
    assert header["bucket"] == "rldb" and header["prefix"] == "processed_v3/"
    assert header["count"] == 3 and header["listed_at"].endswith("+00:00")
    assert lines[1:] == [
        f"{NESTED}\tprocessed_v3/other_lab/kitchen/",
        f"{FIRST}\tprocessed_v3/sharpa_dex/",
        f"{SECOND}\tprocessed_v3/sharpa_dex/",
    ]
    assert not list(tmp_path.glob(".*partial"))

    read_header, folders = lister.read_s3_hashes(out)
    assert read_header == header
    assert folders == {
        NESTED: "processed_v3/other_lab/kitchen/",
        FIRST: "processed_v3/sharpa_dex/",
        SECOND: "processed_v3/sharpa_dex/",
    }


def test_a_hash_under_two_folders_and_a_cut_off_file(tmp_path):
    path = tmp_path / "s3.tsv"
    pairs = [(FIRST, "processed_v3/other_lab/"), (FIRST, "processed_v3/sharpa_dex/")]
    lister.write_s3_hashes(
        path, "rldb", "processed_v3/", "2026-10-21T00:00:00+00:00", pairs
    )
    _, episodes = lister.read_s3_listing(path)
    assert episodes == pairs
    assert lister.read_s3_hashes(path)[1] == {FIRST: "processed_v3/other_lab/"}

    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="list again"):
        lister.read_s3_hashes(path)
    path.write_text(f"{FIRST}\tprocessed_v3/\n")
    with pytest.raises(ValueError, match="not an existing_hashes_s3 listing"):
        lister.read_s3_hashes(path)
