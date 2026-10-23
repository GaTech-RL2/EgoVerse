"""Each edit of the update bench changes its target in every format and nothing else."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from egomimic.rldb.lance.schema import HUMAN_STEPS, steps_path
from egomimic.scripts.format_bench import bench_update as bu

ROWS = 1300


@pytest.fixture
def lay(tmp_path):
    """Three source recordings; episode lengths that are not multiples of the zarr chunk, and
    a 150-frame episode with no annotation span."""
    rng = np.random.default_rng(0)
    batches = []
    for i, n in enumerate((300, 450, 700)):
        cols = {
            k: rng.standard_normal((n, d), dtype=np.float32)
            for k, d in bu.VECTORS.items()
        }
        cols.update({k: rng.random(n, dtype=np.float32) for k in bu.SCALARS})
        cols.update({k: rng.random(n) > 0.1 for k in bu.BOOLS})
        cols["g"] = np.arange(n)
        batches.append(bu.steps_batch(cols, f"src-{i}"))
    pilot = tmp_path / "pilot"
    key = {"lab": "rl2", "embodiment": "human_bimanual", "end_effector": "human_hand"}
    lance.write_dataset(
        pa.Table.from_pylist([key]), str(pilot / "lance/recordings.lance")
    )
    lance.write_dataset(
        batches, str(pilot / "lance" / steps_path(**key)), schema=HUMAN_STEPS
    )
    with open(pilot / "manifest.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["episode_hash", "num_frames"])
        w.writerows([["a", 150], ["b", 330], ["c", 640]])
    lay = bu.pilot_layout(pilot, ROWS)
    assert lay.lens.sum() == ROWS and lay.n_eps > 3
    assert min(len(lay.spans(ep, 0)) for ep in range(lay.n_eps)) == 0
    assert max(len(lay.spans(ep, 0)) for ep in range(lay.n_eps)) > 1
    return lay


def _digests(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_zarr_shards_match_zarr(tmp_path, lay):
    """Hand-built objects read back through zarr exactly as ZarrWriter would write them."""
    pytest.importorskip("obstore")  # the optional "bench" extra
    root = tmp_path / "zarr"
    bu.make_zarr(str(root), lay, workers=4)
    for ep in range(lay.n_eps):
        g = zarr.open_group(str(root / bu.ep_name(ep)), mode="r")
        hp = lay.pool["head_pose"][lay.src[lay.g(ep)]]
        np.testing.assert_array_equal(g["obs_head_pose"][: len(hp)], hp)
        assert not g["obs_head_pose"][len(hp) :].any()
        want = [json.loads(b) for b in bu.annotation_items(lay.spans(ep, 0))]
        assert [json.loads(b) for b in g["annotations"][:]] == want
        assert sorted(g.keys()) == ["annotations", "obs_head_pose"]


@pytest.mark.parametrize("op", bu.OPS)
def test_zarr(tmp_path, lay, op):
    pytest.importorskip("obstore")  # the optional "bench" extra
    root = tmp_path / "zarr"
    bu.make_zarr(str(root), lay, workers=4)
    target = {"replace": "obs_head_pose/c", "annotations": "annotations/"}.get(op)
    before = _digests(root)
    bu.update_zarr(str(root), lay, op, k=3, workers=4)
    bu.verify_zarr(str(root), lay, op, k=3)
    after = _digests(root)
    for path, digest in before.items():
        if op == "append" and path.count("/") == 1 and path.endswith("zarr.json"):
            continue
        if target is None or target not in path:
            assert after.get(path) == digest, path
    if op == "append":
        g = zarr.open_group(str(root / bu.ep_name(0)), mode="r")
        assert set(g.attrs["features"]) == {"obs_head_pose", "annotations", "feat_3"}


@pytest.mark.parametrize("fmt", ["lerobot", "lerobot_zstd"])
@pytest.mark.parametrize("op", bu.OPS)
def test_lerobot(tmp_path, lay, monkeypatch, op, fmt):
    monkeypatch.setattr(bu, "PARQUET_MB", 0.2)
    root = tmp_path / fmt
    bu.make_lerobot(str(root), lay, workers=4, compression=bu.COMPRESSION[fmt])
    files = sorted((root / "data").rglob("*.parquet"))
    assert len(files) > 1
    assert pq.ParquetFile(files[0]).metadata.row_group(0).column(0).compression == (
        bu.COMPRESSION[fmt].upper()
    )
    target = {"replace": ["head_pose"], "annotations": bu.LANGUAGE_COLS}.get(op, [])
    before = pq.read_table(files).drop_columns(target)
    groups = [pq.ParquetFile(f).num_row_groups for f in files]
    meta_before = pq.read_table(root / "meta/episodes")
    bu.update_lerobot(str(root), lay, op, k=3, workers=4)
    bu.verify_lerobot(str(root), lay, op, k=3)
    after = pq.read_table(files)
    if op == "append":
        after = after.drop_columns(["feat_3"])
    assert after.drop_columns(target).equals(before)
    assert [pq.ParquetFile(f).num_row_groups for f in files] == groups
    meta = pq.read_table(root / "meta/episodes")
    for c in meta_before.column_names:
        changed = (
            op == "replace" and c.startswith("stats/head_pose/") and "count" not in c
        )
        assert meta.column(c).equals(meta_before.column(c)) != changed, c
    if op == "append":
        assert "stats/feat_3/mean" in meta.column_names
        info = json.loads((root / "meta/info.json").read_text())
        assert "feat_3" in info["features"]


@pytest.mark.parametrize("op", bu.OPS)
def test_lance(tmp_path, lay, op):
    root = str(tmp_path / "lance")
    bu.make_lance(root, lay, rows_per_frag=500)
    steps = lance.dataset(bu.steps_uri(root))
    assert len(steps.get_fragments()) > 1
    steps_before = steps.to_table()
    tasks_before = lance.dataset(bu.tasks_uri(root)).to_table()
    files_before = {d.path for f in steps.get_fragments() for d in f.data_files()}
    bu.update_lance(root, lay, op, k=3, workers=4)
    bu.verify_lance(root, lay, op, k=3)
    steps = lance.dataset(bu.steps_uri(root))
    steps_after = steps.to_table()
    tasks_after = lance.dataset(bu.tasks_uri(root)).to_table()
    files_after = {d.path for f in steps.get_fragments() for d in f.data_files()}
    if op == "annotations":
        assert steps_after.equals(steps_before)
        assert files_after == files_before
        assert tasks_after.drop_columns(["annotations"]).equals(
            tasks_before.drop_columns(["annotations"])
        )
        return
    assert tasks_after.equals(tasks_before)
    # One new file per fragment, and every existing data file is left in place.
    assert files_before < files_after
    assert len(files_after) == 2 * len(files_before)
    if op == "replace":
        assert steps_after.drop_columns(["head_pose"]).equals(
            steps_before.drop_columns(["head_pose"])
        )
    else:
        assert steps_after.drop_columns(["feat_3"]).equals(steps_before)
