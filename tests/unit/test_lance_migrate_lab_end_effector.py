import datetime

import pyarrow as pa
import pytest

lance = pytest.importorskip("lance")

from fixtures.lance_tables import (  # noqa: E402
    alias,
    annotation,
    eva_steps_table,
    steps_table,
    task,
)

from egomimic.rldb.lance.schema import (  # noqa: E402
    ALIASES,
    ANNOTATIONS,
    TASKS,
    recording_id,
)
from egomimic.scripts.lance_pilot import migrate_lab_end_effector as m  # noqa: E402

# Frozen here: schema.RECORDINGS / CHANGES move to the new layout.
OLD_RECORDINGS = pa.schema(
    [
        ("recording_id", pa.string()),
        ("vendor_recording_id", pa.string()),
        ("lab", pa.string()),
        ("site", pa.string()),
        ("embodiment", pa.string()),
        ("rig_name", pa.string()),
        ("operator", pa.string()),
        ("scene", pa.string()),
        ("captured_at", pa.timestamp("ms", tz="UTC")),
        ("fps", pa.float32()),
        ("num_frames", pa.int64()),
        ("video_path", pa.map_(pa.string(), pa.string())),
        ("intrinsics", pa.map_(pa.string(), pa.list_(pa.float32()))),
        ("extrinsics", pa.map_(pa.string(), pa.list_(pa.float32()))),
        ("license", pa.string()),
        ("content_digest", pa.string()),
    ]
)
OLD_CHANGES = pa.schema(
    [
        ("tag", pa.string()),
        ("lab", pa.string()),
        ("committed_at", pa.timestamp("ms", tz="UTC")),
        ("operator", pa.string()),
        ("added", pa.list_(pa.string())),
        ("replaced", pa.list_(pa.string())),
        ("provenance", pa.string()),
    ]
)

# (source, vendor id, site, embodiment, frames)
RECS = [
    ("aria", "a1", "rl2", "human_bimanual", 7),
    ("aria", "a2", "eth", "human_bimanual", 5),
    ("aria", "a3", "rl2", "human_right_arm", 6),
    ("aria", "a4", "rl2", "human_bimanual", 5),
    ("eva", "e1", "rl2", "eva_bimanual", 6),
    ("eva", "e2", "rl2", "eva_left_arm", 5),
]


def _rid(src, vid):
    return recording_id("aria", f"{src}-{vid}")


def _old_root(
    root,
    sources=("aria", "eva"),
    site=lambda s: s,
    episode_hash=lambda rid: f"h{rid}",
    task_annotations=False,
):
    """An old-schema root holding the RECS of `sources`; returns their recording_ids."""
    steps = {s: [] for s in sources}
    rows = []
    for i, (src, vid, s, emb, n) in enumerate(RECS):
        if src not in sources:
            continue
        rid = _rid(src, vid)
        steps[src].append(
            steps_table(rid, n, 100 * i)
            if src == "aria"
            else eva_steps_table(rid, n, 100 * i)
        )
        path = f"videos/{rid}/front_1.mp4"
        (root / path).parent.mkdir(parents=True)
        (root / path).write_bytes(rid.encode())
        rows.append(
            dict(
                recording_id=rid,
                vendor_recording_id=vid,
                lab=src,
                site=site(s),
                embodiment=emb,
                operator="op",
                fps=30.0,
                num_frames=n,
                video_path=[("front_1", path)],
                intrinsics=[],
                extrinsics=[],
            )
        )
    for src, parts in steps.items():
        lance.write_dataset(pa.concat_tables(parts), root / "steps" / f"{src}.lance")
    lance.write_dataset(
        pa.Table.from_pylist(rows, schema=OLD_RECORDINGS), root / "recordings.lance"
    )
    rids = [r["recording_id"] for r in rows]
    tasks = pa.Table.from_pylist([task(r, 0, 2) for r in rids], schema=TASKS)
    if task_annotations:
        # Tables written while annotations were nested in tasks.
        legacy = [["wipe"]] + [[]] * (len(rids) - 2) + [None]
        tasks = tasks.append_column(
            "annotations", pa.array(legacy, pa.list_(pa.string()))
        )
    lance.write_dataset(tasks, root / "tasks.lance")
    lance.write_dataset(
        pa.Table.from_pylist(
            [alias(r, 0, 2, episode_hash(r)) for r in rids], schema=ALIASES
        ),
        root / "aliases.lance",
    )
    lance.write_dataset(
        pa.Table.from_pylist([annotation(rids[0], 0, 1, "x")], schema=ANNOTATIONS),
        root / "annotations.lance",
    )
    now = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    change = dict(committed_at=now, operator="op", provenance=None)
    by_src = {s: [r["recording_id"] for r in rows if r["lab"] == s] for s in sources}
    lance.write_dataset(
        pa.Table.from_pylist(
            [
                *(
                    dict(tag=f"{s}-1", lab=s, added=ids, replaced=[], **change)
                    for s, ids in by_src.items()
                ),
                dict(
                    tag=f"annotations-{sources[0]}",
                    lab="annotations",
                    added=rids[:1],
                    replaced=[],
                    **change,
                ),
            ],
            schema=OLD_CHANGES,
        ),
        root / "changes.lance",
    )
    (root / f"README-{'-'.join(sources)}.txt").write_text("pilot")
    return rids


def _two_roots(tmp_path, **kw):
    aria, eva = tmp_path / "aria", tmp_path / "eva"
    rids = _old_root(aria, ("aria",), task_annotations=True, **kw)
    rids += _old_root(eva, ("eva",), **kw)
    return aria, eva, rids


def test_merges_roots_and_splits_steps_by_lab_embodiment_end_effector(tmp_path):
    aria, eva, rids = _two_roots(tmp_path)
    dst = tmp_path / "new"
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    m.main(["--src", str(aria), "--src", str(eva), "--dst", str(dst)])
    assert (
        sorted(
            p.relative_to(tmp_path)
            for p in tmp_path.rglob("*")
            if dst not in p.parents and p != dst
        )
        == before
    )

    rec = lance.dataset(dst / "recordings.lance").to_table()
    assert rec.column_names == m.NEW_RECORDINGS_COLUMNS
    assert rec["recording_id"].to_pylist() == rids
    assert rec["lab"].to_pylist() == ["rl2", "eth", "rl2", "rl2", "rl2", "rl2"]
    assert (
        rec["end_effector"].to_pylist()
        == ["human_hand"] * 4 + ["parallel_jaw_gripper"] * 2
    )

    tables = m.new_steps_uris(dst)
    assert sorted(tables) == [
        "eth/human_bimanual/hand",
        "rl2/eva_bimanual/gripper",
        "rl2/eva_left_arm/gripper",
        "rl2/human_bimanual/hand",
        "rl2/human_right_arm/hand",
    ]
    bimanual = lance.dataset(tables["rl2/human_bimanual/hand"]).to_table()
    assert bimanual["recording_id"].to_pylist() == [rids[0]] * 7 + [rids[3]] * 5
    assert bimanual.schema == lance.dataset(aria / "steps" / "aria.lance").schema
    total = sum(lance.dataset(u).count_rows() for u in tables.values())
    assert total == sum(r[-1] for r in RECS)

    changes = lance.dataset(dst / "changes.lance").to_table().to_pylist()
    assert [(c["tag"], c["lab"], c["added"]) for c in changes] == [
        ("aria-1", "eth/human_bimanual/hand", [rids[1]]),
        ("aria-1", "rl2/human_bimanual/hand", [rids[0], rids[3]]),
        ("aria-1", "rl2/human_right_arm/hand", [rids[2]]),
        ("annotations-aria", "annotations", [rids[0]]),
        ("eva-1", "rl2/eva_bimanual/gripper", [rids[4]]),
        ("eva-1", "rl2/eva_left_arm/gripper", [rids[5]]),
        ("annotations-eva", "annotations", [rids[4]]),
    ]
    tasks = lance.dataset(dst / "tasks.lance").to_table()
    assert tasks.schema.names == TASKS.names
    assert tasks["recording_id"].to_pylist() == rids
    for name in ("aliases.lance", "annotations.lance"):
        old = [lance.dataset(r / name).to_table() for r in (aria, eva)]
        assert lance.dataset(dst / name).to_table() == pa.concat_tables(old)
    for r in rec.to_pylist():
        path = dict(r["video_path"])["front_1"]
        assert (dst / path).read_bytes() == r["recording_id"].encode()
    assert (dst / "README-aria.txt").exists() and (dst / "README-eva.txt").exists()
    assert not (dst / "steps" / "aria.lance").exists()
    assert "migrate-lab-end-effector" in lance.dataset(dst / "tasks.lance").tags.list()


def test_dry_run_writes_nothing_and_counts_task_annotations(tmp_path, capsys):
    aria, eva, _ = _two_roots(tmp_path)
    dst = tmp_path / "new"
    m.main(["--src", str(aria), "--src", str(eva), "--dst", str(dst), "--dry-run"])
    assert not dst.exists()
    out = capsys.readouterr().out
    assert "12  steps/rl2/human_bimanual/human_hand.lance" in out
    assert "7  changes.lance" in out
    assert "tasks.annotations: 1 rows" in out


def test_recording_in_two_sources_fails(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    rids = _old_root(a, ("aria",))
    _old_root(b, ("aria",), episode_hash=lambda rid: f"other{rid}")
    with pytest.raises(ValueError, match=rids[0]):
        m.main(["--src", str(a), "--src", str(b), "--dst", str(tmp_path / "new")])
    assert not (tmp_path / "new").exists()


def test_alias_episode_hash_in_two_sources_fails(tmp_path):
    aria, eva, _ = _two_roots(tmp_path, episode_hash=lambda rid: "same")
    with pytest.raises(ValueError, match="episode_hash"):
        m.main(["--src", str(aria), "--src", str(eva), "--dst", str(tmp_path / "n")])


def test_null_site_fails_unless_overridden(tmp_path):
    src = tmp_path / "old"
    rids = _old_root(src, site=lambda s: None if s == "eth" else s)
    with pytest.raises(ValueError, match=rids[1]):
        m.main(["--src", str(src), "--dst", str(tmp_path / "new")])
    assert not (tmp_path / "new").exists()
    m.main(
        ["--src", str(src), "--dst", str(tmp_path / "new"), "--lab-for-null-site", "x"]
    )
    rec = lance.dataset(tmp_path / "new" / "recordings.lance").to_table()
    assert rec["lab"].to_pylist()[1] == "x"
    assert "x/human_bimanual/hand" in m.new_steps_uris(tmp_path / "new")


def test_orphan_steps_rows_need_a_flag(tmp_path):
    src = tmp_path / "old"
    _old_root(src)
    orphan = steps_table(recording_id("aria", "gone"), 5, 0)
    lance.write_dataset(orphan, src / "steps" / "aria.lance", mode="append")
    with pytest.raises(ValueError, match="drop-orphan-steps"):
        m.main(["--src", str(src), "--dst", str(tmp_path / "new"), "--dry-run"])
    m.main(["--src", str(src), "--dst", str(tmp_path / "new"), "--drop-orphan-steps"])


def test_refuses_to_write_into_a_source(tmp_path):
    aria, eva, _ = _two_roots(tmp_path)
    with pytest.raises(SystemExit):
        m.main(["--src", str(aria), "--src", str(eva), "--dst", str(eva / "new")])
