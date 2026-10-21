"""The committer on R2: concurrent Lance commits there are safe, and staged deliveries go
through commit_staged into an s3:// root that LanceStreamingResolver then reads.

Needs EGOVERSE_TEST_R2_PREFIX (an s3:// scratch prefix; each run writes under a fresh
subprefix and deletes it) and R2_* credentials. CPU only.
"""

import concurrent.futures as cf
import os
import uuid

import pyarrow as pa
import pytest

lance = pytest.importorskip("lance")
pytest.importorskip("av")

from egomimic.rldb.lance import commit  # noqa: E402
from egomimic.rldb.lance.dataset import LanceStreamingResolver, open_table  # noqa: E402


@pytest.fixture
def prefix():
    base = os.environ.get("EGOVERSE_TEST_R2_PREFIX")
    if not base:
        pytest.skip("set EGOVERSE_TEST_R2_PREFIX")
    p = commit.join(base, f"test-{uuid.uuid4().hex[:12]}")
    yield p
    commit.remove([commit.join(p, f) for f in commit.list_files(p)])


def test_concurrent_appends_all_land(prefix):
    uri = commit.join(prefix, "t.lance")
    commit._append(uri, pa.table({"w": pa.array([], pa.int64())}))
    writers = 6

    def write(w):
        commit._append(uri, pa.table({"w": [w] * 100}))

    with cf.ThreadPoolExecutor(writers) as ex:
        list(ex.map(write, range(writers)))
    ds = open_table(uri)
    assert ds.count_rows() == writers * 100
    assert sorted(set(ds.to_table().column("w").to_pylist())) == list(range(writers))
    assert ds.version == writers + 1


def test_staged_deliveries_commit_to_r2_and_stream(prefix, tmp_path):
    from fixtures.lance_tables import eva_delivery

    from egomimic.scripts.lance_commit import commit_staged

    staging, root = commit.join(prefix, "staging"), commit.join(prefix, "lance")
    for vid in ("100", "200"):
        d = eva_delivery(tmp_path / vid, vid)
        commit.stage(d, commit.join(staging, "eva", vid))
    report = commit_staged(staging, "eva", root, tmp_path / "work")
    (batch,) = report["batches"]
    assert len(batch["added"]) == 2 and report["rejected"] == {}
    assert commit.list_files(commit.join(staging, "eva")) == {}

    key_map = {
        "img": {"key_type": "camera_keys", "zarr_key": "images.front_1"},
        "left.cmd_ee_pose": {
            "key_type": "action_keys",
            "zarr_key": "left.cmd_ee_pose",
            "horizon": 2,
        },
    }
    leaves = LanceStreamingResolver(
        root,
        lab="eva",
        key_map=key_map,
        cameras={"images.front_1": "front_1"},
        tag=batch["tag"],
    ).resolve()
    leaf = next(iter(leaves.values()))
    (sample,) = leaf.fetch_planned([(leaf, leaf.plan_sample(3))])
    assert sample[1]["img"].shape == (1, 48, 64, 3)
