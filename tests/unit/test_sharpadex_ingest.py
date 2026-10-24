"""Chunked SharpaDex ingestion on a tiny synthetic release, with fakes only.

The Hub downloader is a fake that writes files of the listed size, and the
converter is a stub script that writes the converter's sidecars; nothing
touches the network, S3 or the real converter.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest

from egomimic.scripts.sharpa_process import sharpadex_ingest as ingest

#: (task, season, v2.1 folder spelling) -> episode lengths.
SEASONS = {
    ("task_a", "season_POC1_2026_01_01_00_00_00_train", "lerobot_v2.1"): [100] * 3,
    ("task_a", "season_POC1_2026_01_02_00_00_00_train", "lerobot_v2.1"): [100] * 2,
    ("task_b", "season_POC2_2026_01_03_00_00_00_train", "lerobot_v2_1"): [100] * 6,
}
PARQUET, VIDEO = 1000, 2000
#: Output estimate: 100 frames x 10 + 1000 = 2000 bytes per episode.
X, Y = 10, 1000
#: 70 kB: one whole season of 3 x 13 kB plus one of 13 + 11 kB, or 5 episodes.
BUDGET_GB = 70_000 / 1e9
#: This stream's video is absent from the listing (a planning warning).
MISSING = ("task_a/season_POC1_2026_01_02_00_00_00_train/1", "tactile_deform")
#: Array folders (and chunk bytes) of a stub episode, one per report group.
ARRAYS = {
    "images.front_1": 300,
    "images.left_wrist": 200,
    "left.obs_tactile_image": 50,
    "left.obs_joints": 40,
}
Usage = namedtuple("Usage", "total used free")
#: The CPUs this process may use, read before any test patches
#: ``os.sched_getaffinity``; workers are pinned to real CPUs.
REAL_CPUS = sorted(os.sched_getaffinity(0))


def write_episode(
    out,
    rid,
    name,
    frames,
    status="complete",
    reasons=(),
    tcp_passed=True,
    errors=(),
    mp4=True,
    lab="sharpa_dex",
    store_rid=None,
):
    """Write what the converter leaves for one episode, ``conversion.json`` last.

    The report is in ``sharpa_to_zarr``'s format; ``errors`` are validator
    errors, and a failed TCP check or a non-complete status also appear
    among its errors, as the converter writes them. Self-contained: the stub
    converter runs this source in a subprocess.
    """
    import json
    from pathlib import Path

    out = Path(out)
    task, season, index = rid.split("/")
    vid = f"{task}__{season}__{int(index):06d}"
    stored = store_rid or rid
    s_task, s_season, s_index = stored.split("/")
    attrs = {
        "vendor_recording_id": f"{s_task}__{s_season}__{int(s_index):06d}",
        "source_episode_id": stored,
        "lab": lab,
        "license": "cc-by-nc-4.0",
    }
    store = out / f"{name}.zarr"
    for array, nbytes in ARRAYS.items():
        (store / array / "c").mkdir(parents=True, exist_ok=True)
        (store / array / "zarr.json").write_text("{}")
        (store / array / "c" / "0").write_bytes(b"\0" * nbytes)
    (store / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "attributes": attrs})
    )
    if mp4:
        (out / f"{name}.mp4").write_bytes(b"\0" * 500)
    else:
        (out / f"{name}.mp4").unlink(missing_ok=True)
    side = {
        "rotation_error_deg_median": 0.001,
        "swapped_rotation_error_deg_median": 115.0,
        "position_error_m_p95": 0.006,
    }
    tcp = {
        "passed": tcp_passed,
        "sides": {"left": side, "right": side},
        "problems": [] if tcp_passed else ["left TCP rotation 40 deg off"],
    }
    all_errors = [f"validator: {e}: bad" for e in errors]
    if not tcp_passed:
        all_errors += [f"tcp_check: {p}" for p in tcp["problems"]]
    if status != "complete":
        all_errors.append(f"conversion: data_status is {status!r}")
    warnings = [
        "validator: intrinsics: marked as not available (MissingMetadata)",
        "validator: left.cmd_hand_joints: 1 joint(s) more than 0.035 rad past",
    ]
    (out / f"{name}.validation.json").write_text(
        json.dumps(
            {
                "format": "sharpadex_validation_v1",
                "episode": f"{name}.zarr",
                "vendor_recording_id": vid,
                "source_episode_id": rid,
                "allow_missing": True,
                "waived": ["intrinsics", "objects", "scene"],
                "ok": not all_errors,
                "errors": all_errors,
                "warnings": warnings,
                "checks": {
                    "validator": {
                        "ok": not errors,
                        "errors": len(errors),
                        "warnings": len(warnings),
                    },
                    "fk_residual": {
                        "max_m": {
                            k: 1e-16
                            for k in ("left.obs", "right.obs", "left.cmd", "right.cmd")
                        },
                        "frames": 64,
                    },
                    "tcp_check": tcp,
                    "conversion": {
                        "data_status": status,
                        "data_status_reasons": list(reasons),
                        "source_frames": frames,
                        "retained_frames": frames,
                        "stored_frames": frames,
                    },
                },
            }
        )
    )
    report = {
        "data_status": status,
        "data_status_reasons": list(reasons),
        "data_limitations": ["no measured front_1 calibration"],
        "tcp_check": tcp,
        "annotations": {"subtasks": 2, "skills": 2, "notes": []},
    }
    sidecar = {
        "episode": str(store),
        "episode_hash": name,
        "vendor_recording_id": vid,
        "source_episode_id": rid,
        "total_frames": frames,
        "data_status": status,
        "seconds": frames * 0.03,
        "validation": {"ok": not all_errors, "errors": all_errors, "warnings": 2},
        "attributes": {**attrs, "conversion_report": report},
    }
    (out / f"{name}.conversion.json").write_text(json.dumps(sidecar, indent=2))


#: A stand-in for ``sharpa_to_zarr``: reads the flags the driver passes,
#: honours --skip-done, fails the ids in STUB_FAIL_IDS (leaving a partial),
#: first sleeps STUB_SLEEP seconds, and logs what it saw to
#: STUB_LOG_DIR/<pid>.json.
STUB_MAIN = """
import argparse, json, os, sys, time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--recording-ids", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--skip-done", action="store_true")
args, _ = parser.parse_known_args()
Path(os.environ["STUB_LOG_DIR"], f"{os.getpid()}.started").write_text("")
time.sleep(float(os.environ.get("STUB_SLEEP", "0")))
lines = Path(args.manifest).read_text().splitlines()
records = {r["recording_id"]: r for r in (json.loads(x) for x in lines[1:] if x)}
ids = [x.strip() for x in Path(args.recording_ids).read_text().splitlines()]
ids = [x for x in ids if x]
out = Path(args.output_dir)
fail = set(filter(None, os.environ.get("STUB_FAIL_IDS", "").split(",")))
converted = []
for rid in ids:
    name = records[rid]["episode_hash"]
    conv = out / f"{name}.conversion.json"
    if args.skip_done and conv.exists():
        task, season, index = rid.split("/")
        canonical = f"{task}__{season}__{int(index):06d}"
        if json.loads(conv.read_text())["vendor_recording_id"] == canonical:
            continue
    if rid in fail:
        (out / f".{name}.zarr.partial").mkdir(parents=True, exist_ok=True)
        continue
    write_episode(out, rid, name, records[rid]["length"])
    converted.append(rid)
names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
         "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "PYTHONUNBUFFERED", "PYTHONPATH")
log = {"ids": ids, "converted": converted, "argv": sys.argv[1:],
       "cpus": sorted(os.sched_getaffinity(0)),
       "env": {k: os.environ.get(k) for k in names}}
Path(os.environ["STUB_LOG_DIR"], f"{os.getpid()}.json").write_text(json.dumps(log))
sys.exit(1 if fail & set(ids) else 0)
"""


class Http429(Exception):
    """What the Hub client raises when rate limited."""

    response = SimpleNamespace(status_code=429)


class FakeHub:
    """Writes each requested file with its listed size; can rate-limit once."""

    def __init__(self, sizes: dict[str, int], fail_once=(), broken=()):
        self.sizes = sizes
        self.fail_once = set(fail_once)
        self.broken = set(broken)
        self.calls: list[str] = []

    def __call__(self, repo_id, filename, revision, local_dir):
        assert (repo_id, revision) == (ingest.REPO_ID, ingest.REVISION)
        self.calls.append(filename)
        if filename in self.fail_once:
            self.fail_once.discard(filename)
            raise Http429("429 Too Many Requests")
        if filename in self.broken:
            raise ValueError("repository not found")
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0" * self.sizes[filename])


class Release:
    """A synthetic SharpaDex release: local meta, a manifest and a size listing."""

    def __init__(self, tmp_path: Path, checked: bool = True):
        self.tmp = tmp_path
        self.root = tmp_path / "release"
        self.state = tmp_path / "state"
        self.work = tmp_path / "work"
        self.manifest = tmp_path / "manifest.jsonl"
        self.sizes = tmp_path / "sizes.csv"
        self.stub = tmp_path / "stub_converter.py"
        self.stub_logs = tmp_path / "stub_logs"
        self.stub_logs.mkdir()
        self.stub.write_text(
            f"import json\nfrom pathlib import Path\n\nARRAYS = {ARRAYS!r}\n\n"
            + inspect.getsource(write_episode)
            + STUB_MAIN
        )
        self.ids: dict[str, list[str]] = {}
        self.root.mkdir()
        (self.root / "README.md").write_text("license: cc-by-nc-4.0\n")
        rows = [("README.md", 23), (".gitattributes", 10)]
        records, n = [], 0
        for (task, season, v21), lengths in SEASONS.items():
            meta = self.root / task / season / v21 / "meta"
            meta.mkdir(parents=True)
            (meta / "info.json").write_text('{"fps": 30}')
            rows.append((f"{task}/{season}/{v21}/meta/info.json", 11))
            rows.append(
                (f"{task}/{season}/lerobot_v3.0/data/chunk-000/file-000.parquet", 9)
            )
            # Index len(lengths) is a dropped duplicate: listed, never planned.
            for index in range(len(lengths) + 1):
                rid = f"{task}/{season}/{index}"
                base = f"{task}/{season}/{v21}"
                rows.append(
                    (f"{base}/data/chunk-000/episode_{index:06d}.parquet", PARQUET)
                )
                for stream in (*ingest.STREAMS, "tactile_raw"):
                    if (rid, stream) == MISSING:
                        continue
                    rows.append(
                        (
                            f"{base}/videos/chunk-000/observation.images.{stream}/"
                            f"episode_{index:06d}.mp4",
                            VIDEO,
                        )
                    )
                keep = index < len(lengths)
                if keep:
                    self.ids.setdefault(season, []).append(rid)
                records.append(
                    {
                        "recording_id": rid,
                        "task": task,
                        "season": season,
                        "episode_index": index,
                        "rig": season.split("_")[1],
                        "length": 100,
                        "fps": 30,
                        "source_uid": f"uid{n}",
                        "fingerprint": f"fp{n}",
                        "tags": {},
                        "start_local": None,
                        "start_source": None,
                        "clock": "synthesized_from_season_name_and_frame_index",
                        "start_ns": 1_767_225_600_000_000_000 + n * 10**9,
                        "episode_hash": (
                            f"2026-02-01-00-00-{n:02d}-000000" if keep else None
                        ),
                        "hash_offset_ms": 0,
                        "decision": "keep" if keep else "drop_duplicate",
                        "reasons": [] if keep else ["identical copy"],
                        "duplicate_of": None,
                        "flags": [],
                    }
                )
                n += 1
        self.records = {r["recording_id"]: r for r in records}
        self.size_of = dict(rows)
        self.sizes.write_text("path,size\n" + "".join(f"{p},{s}\n" for p, s in rows))
        header = {
            "release_root": str(self.root),
            "source_revision": ingest.REVISION,
            "sql_checked": checked,
            "s3_checked": checked,
            "collision_check_complete": checked,
            "summary": {},
        }
        with self.manifest.open("w") as f:
            f.write(json.dumps({"manifest": header}, sort_keys=True) + "\n")
            for record in records:
                f.write(json.dumps(record, sort_keys=True) + "\n")

    # -- CLI ---------------------------------------------------------------

    def plan(self, *extra, state: Path | None = None) -> int:
        return ingest.main(
            [
                "plan",
                "--state-dir",
                str(state or self.state),
                "--manifest",
                str(self.manifest),
                "--sizes",
                str(self.sizes),
                "--budget-gb",
                str(BUDGET_GB),
                "--out-bytes-per-frame",
                str(X),
                "--mp4-bytes-per-episode",
                str(Y),
                *extra,
            ]
        )

    def run(self, command: str, *extra, chunk: str = "chunk_0000") -> int:
        argv = [command, "--state-dir", str(self.state)]
        if command != "status":
            argv += ["--chunk", chunk]
        if command in ("fetch", "convert", "clean"):
            argv += ["--dataset-root", str(self.root)]
        if command != "status":
            argv += ["--work-root", str(self.work)]
        if command == "fetch":
            argv += ["--backoff-seconds", "0"]
        if command == "convert":
            argv += [
                "--converter-cmd",
                shlex.join([sys.executable, str(self.stub)]),
                "--monitor-seconds",
                "0.02",
            ]
        return ingest.main([*argv, *extra])

    # -- helpers -----------------------------------------------------------

    def chunk(self, name: str = "chunk_0000") -> dict:
        plan = json.loads((self.state / "plan.json").read_text())
        return next(c for c in plan["chunks"] if c["chunk"] == name)

    def out(self, name: str = "chunk_0000") -> Path:
        return self.work / "out" / name

    def hub(self, monkeypatch, **kwargs) -> FakeHub:
        hub = FakeHub(self.size_of, **kwargs)
        monkeypatch.setattr(ingest, "download_file", hub)
        return hub

    def convert_all(self, name: str = "chunk_0000", **changes) -> None:
        """Write converter outputs for a chunk; ``changes[i]`` alters episode i."""
        out = self.out(name)
        out.mkdir(parents=True, exist_ok=True)
        for i, rid in enumerate(self.chunk(name)["ids"]):
            record = self.records[rid]
            write_episode(
                out,
                rid,
                record["episode_hash"],
                record["length"],
                **changes.get(f"e{i}", {}),
            )

    def ledger(self, stage: str | None = None) -> list[dict]:
        entries = ingest.read_ledger(self.state)
        return [e for e in entries if stage is None or e["stage"] == stage]

    def stub_runs(self) -> list[dict]:
        return [
            json.loads(p.read_text()) for p in sorted(self.stub_logs.glob("*.json"))
        ]


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    """A quiet node with ample disk; no git, squeue or Slurm job of our own."""
    monkeypatch.setattr(os, "getloadavg", lambda: (0.5, 0.5, 0.5))
    monkeypatch.setattr(
        ingest.shutil, "disk_usage", lambda path: Usage(10**15, 0, 10**15)
    )
    monkeypatch.setattr(ingest, "git_sha", lambda: "0" * 40)

    def no_squeue(job):
        raise AssertionError(f"squeue called for {job}")

    monkeypatch.setattr(ingest, "squeue_lists", no_squeue)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    for name in ("STUB_FAIL_IDS", "STUB_SLEEP", "R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)


@pytest.fixture
def release(tmp_path, monkeypatch) -> Release:
    data = Release(tmp_path)
    monkeypatch.setenv("STUB_LOG_DIR", str(data.stub_logs))
    return data


def season_ids(release: Release, n: int) -> list[str]:
    return release.ids[list(SEASONS)[n][1]]


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def test_plan_packs_whole_seasons_within_budget_and_splits_a_large_one(
    release, tmp_path
):
    assert release.plan() == 0
    plan = json.loads((release.state / "plan.json").read_text())
    chunks = plan["chunks"]
    a1, a2, b = (season_ids(release, n) for n in range(3))
    assert [c["chunk"] for c in chunks] == ["chunk_0000", "chunk_0001", "chunk_0002"]
    assert chunks[0]["ids"] == a1 + a2
    assert chunks[1]["ids"] == b[:5]
    assert chunks[2]["ids"] == b[5:]
    assert plan["header"]["budget_bytes"] == 70_000
    for chunk in chunks:
        assert chunk["est_in"] + chunk["est_out"] <= 70_000
        assert chunk["est_in"] == sum(size for _, size in chunk["files"])
        assert chunk["frames"] == 100 * chunk["episodes"]
        assert chunk["est_out"] == chunk["episodes"] * (100 * X + Y)
    # Exact inputs: parquet + the five streams, never tactile_raw or v3.0.
    assert chunks[0]["est_in"] == 4 * (PARQUET + 5 * VIDEO) + PARQUET + 4 * VIDEO
    assert chunks[0]["missing_videos"] == [f"{MISSING[0]}: {MISSING[1]}"]
    paths = [p for c in chunks for p, _ in c["files"]]
    assert len(paths) == len(set(paths)) == 11 * 6 - 1
    assert not [p for p in paths if "tactile_raw" in p or "lerobot_v3.0" in p]
    assert all("/lerobot_v2_1/" in p for p, _ in chunks[1]["files"])
    planned = {i for c in chunks for i in c["ids"]}
    assert planned == {r for r, v in release.records.items() if v["decision"] == "keep"}
    header = plan["header"]
    assert (
        header["manifest_sha256"]
        == hashlib.sha256(release.manifest.read_bytes()).hexdigest()
    )
    assert header["streams"] == list(ingest.STREAMS)
    assert header["revision"] == ingest.REVISION
    assert header["totals"] == {
        "chunks": 3,
        "episodes": 11,
        "frames": 1100,
        "est_in": sum(c["est_in"] for c in chunks),
        "est_out": 11 * 2000,
        "missing_videos": 1,
    }
    # The same inputs give the same chunks.
    assert release.plan(state=tmp_path / "again") == 0
    again = json.loads((tmp_path / "again" / "plan.json").read_text())
    assert again["chunks"] == chunks


def test_plan_filters_and_rejects_bad_inputs(release, tmp_path):
    assert release.plan("--tasks", "task_b", "--max-episodes", "2") == 0
    plan = json.loads((release.state / "plan.json").read_text())
    assert [c["ids"] for c in plan["chunks"]] == [season_ids(release, 2)[:2]]
    assert plan["header"]["filters"]["tasks"] == "task_b"

    only = tmp_path / "ids.txt"
    dropped = "task_a/season_POC1_2026_01_01_00_00_00_train/3"
    only.write_text(f"# pilot\n{season_ids(release, 0)[0]}\n{dropped}\n")
    with pytest.raises(SystemExit, match="does not keep"):
        release.plan("--only-ids", str(only), state=tmp_path / "s1")
    only.write_text(f"{season_ids(release, 0)[0]}\n{season_ids(release, 2)[3]}\n")
    assert release.plan("--only-ids", str(only), state=tmp_path / "s2") == 0

    with pytest.raises(SystemExit, match="revision"):
        release.plan("--revision", "f" * 40, state=tmp_path / "s3")
    lines = release.sizes.read_text().splitlines()
    release.sizes.write_text(
        "\n".join(x for x in lines if "episode_000002.parquet" not in x) + "\n"
    )
    with pytest.raises(SystemExit, match="no parquet"):
        release.plan(state=tmp_path / "s4")


def test_plan_refuses_overwrite_and_stages_refuse_a_changed_manifest(release):
    assert release.plan() == 0
    with pytest.raises(SystemExit, match="exists"):
        release.plan()
    assert release.plan("--force") == 0
    text = release.manifest.read_text()
    release.manifest.write_text(text.replace('"length": 100', '"length": 101', 1))
    for command in ("fetch", "convert", "verify", "upload"):
        with pytest.raises(SystemExit, match="not the planned manifest"):
            release.run(command)


# ---------------------------------------------------------------------------
# hash gate, fetch
# ---------------------------------------------------------------------------


def test_fetch_and_convert_refuse_provisional_hashes(tmp_path, monkeypatch):
    release = Release(tmp_path, checked=False)
    monkeypatch.setenv("STUB_LOG_DIR", str(release.stub_logs))
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(REAL_CPUS[:1]))
    release.hub(monkeypatch)
    assert release.plan() == 0
    for command in ("fetch", "convert"):
        with pytest.raises(SystemExit, match="provisional"):
            release.run(command)
    assert release.run("fetch", "--allow-provisional-hashes") == 0
    assert (
        release.run("convert", "--allow-provisional-hashes", "--cpus-per-worker", "1")
        == 0
    )
    for stage in ("fetch", "convert"):
        start = [e for e in release.ledger(stage) if e["state"] == "start"]
        assert [e["details"]["provisional_hashes"] for e in start] == [True]


def test_disk_guard_refuses_when_the_floor_would_be_crossed(
    release, monkeypatch, capsys
):
    assert release.plan() == 0
    hub = release.hub(monkeypatch)
    # 53 kB to download plus 10 kB of outputs on the same filesystem.
    need = release.chunk()["est_in"] + release.chunk()["est_out"]
    floor = 10**9
    free = {"bytes": floor + need - 1}
    monkeypatch.setattr(
        ingest.shutil, "disk_usage", lambda path: Usage(0, 0, free["bytes"])
    )
    with pytest.raises(SystemExit) as exc:
        release.run("fetch", "--floor-gb", "1")
    assert exc.value.code == ingest.EXIT_DISK
    assert "disk guard" in capsys.readouterr().err
    assert hub.calls == []
    (entry,) = release.ledger("fetch")
    assert entry["state"] == "fail" and entry["details"]["reason"] == "disk_guard"
    assert entry["details"]["margin"] == floor - 1
    # Outputs on another filesystem do not count against the dataset disk.
    monkeypatch.setattr(ingest, "same_filesystem", lambda a, b: False)
    free["bytes"] = floor + release.chunk()["est_in"]
    assert release.run("fetch", "--floor-gb", "1") == 0
    assert hub.calls


def test_fetch_downloads_only_missing_files_records_them_and_retries_429(
    release, monkeypatch
):
    assert release.plan() == 0
    files = release.chunk()["files"]
    (present, size), (wrong, wrong_size), (flaky, _) = files[0], files[1], files[2]
    for rel, nbytes in ((present, size), (wrong, wrong_size - 1)):
        (release.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (release.root / rel).write_bytes(b"\0" * nbytes)
    hub = release.hub(monkeypatch, fail_once={flaky})
    assert release.run("fetch") == 0
    assert present not in hub.calls
    assert sorted(set(hub.calls)) == sorted(r for r, _ in files if r != present)
    assert hub.calls.count(flaky) == 2
    for rel, nbytes in files:
        assert (release.root / rel).stat().st_size == nbytes
    record = release.state / "chunk_0000.fetched.txt"
    fetched = record.read_text().split()
    assert sorted(fetched) == sorted(
        str(release.root / r) for r, _ in files if r != present
    )
    start, ok = release.ledger("fetch")
    assert start["details"]["present"] == 1 and start["details"]["files"] == 28
    details = ok["details"]
    assert ok["state"] == "ok"
    assert details["files"] == len(files) - 1 and details["failures"] == 0
    assert details["retries"] == 1 and details["retry_reasons"] == {"HTTP 429": 1}
    assert {"seconds", "mb_per_s", "files_per_s", "bytes"} <= set(details)

    # A rerun downloads and records nothing.
    hub.calls.clear()
    assert release.run("fetch") == 0
    assert hub.calls == [] and record.read_text().split() == fetched

    # A non-retryable error fails the stage after one attempt.
    (release.root / files[3][0]).unlink()
    hub = release.hub(monkeypatch, broken={files[3][0]})
    assert release.run("fetch") == 1
    assert hub.calls == [files[3][0]]
    assert release.ledger("fetch")[-1]["details"]["failures"] == 1


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------


def test_clean_deletes_only_fetched_episode_files(release, monkeypatch, tmp_path):
    assert release.plan() == 0
    files = release.chunk()["files"]
    present = release.root / files[0][0]
    present.parent.mkdir(parents=True, exist_ok=True)
    present.write_bytes(b"\0" * files[0][1])
    release.hub(monkeypatch)
    assert release.run("fetch") == 0
    meta = next(release.root.glob("task_a/*/lerobot_v2.1/meta/info.json"))
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"keep")
    record = release.state / "chunk_0000.fetched.txt"
    with record.open("a") as f:  # entries clean must refuse even if listed
        f.write(f"{meta}\n{release.root / 'README.md'}\n{outside}\n")
    out = release.out()
    (out / ".h.zarr.partial").mkdir(parents=True)
    (out / ".h.zarr.partial" / "chunk").write_bytes(b"12")
    (out / ".h.mp4.partial").write_bytes(b"1")
    (out / "h.conversion.json").write_text("{}")

    # A live lock blocks clean.
    lock = release.state / "chunk_0000.lock"
    lock.write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid()}))
    with pytest.raises(SystemExit, match="held by"):
        release.run("clean")
    lock.unlink()

    assert release.run("clean") == 0
    assert present.is_file()
    assert meta.is_file() and (release.root / "README.md").is_file()
    assert outside.read_bytes() == b"keep"
    assert not [r for r, _ in files[1:] if (release.root / r).exists()]
    season2 = release.root / "task_a" / list(SEASONS)[1][1] / "lerobot_v2.1"
    assert sorted(p.name for p in season2.iterdir()) == ["meta"]
    assert not (present.parents[2] / "videos").exists()
    assert (out / "h.conversion.json").is_file()
    assert not list(out.glob(".*.partial"))
    entry = release.ledger("clean")[-1]["details"]
    assert entry["deleted"] == len(files) - 1 and entry["outputs"] == "kept"
    assert entry["protected"] == {
        "metadata": 2,
        "outside --dataset-root": 1,
    }
    assert entry["freed"] == sum(s for _, s in files[1:]) + 3

    assert release.run("clean", "--discard-outputs") == 0
    assert not out.exists() and present.is_file()


def test_clean_deletes_outputs_after_a_confirmed_upload(release):
    assert release.plan() == 0
    release.convert_all()
    ledger = ingest.Ledger(release.state)
    ledger.write("chunk_0000", "upload", "ok", confirmed=False)
    assert release.run("clean") == 0
    assert release.out().exists()
    ledger.write("chunk_0000", "upload", "ok", confirmed=True)
    assert release.run("clean") == 0
    assert not release.out().exists()
    assert release.ledger("clean")[-1]["details"]["outputs"] == "deleted (uploaded)"


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


def test_cpu_sets_and_shards_cover_every_id_once(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(8, 18)))
    cpus = sorted(os.sched_getaffinity(0))
    assert ingest.cpu_sets(cpus, None, 4) == [[8, 9, 10, 11], [12, 13, 14, 15]]
    assert ingest.cpu_sets(cpus, 1, 3) == [[8, 9, 10]]
    with pytest.raises(ingest.IngestError, match="needs 12 CPUs"):
        ingest.cpu_sets(cpus, 3, 4)
    with pytest.raises(ingest.IngestError, match="at least 4"):
        ingest.cpu_sets(cpus[:2], None, 4)

    ids = [f"t/s/{i}" for i in range(23)]
    lengths = {rid: 100 + (i * 37) % 250 for i, rid in enumerate(ids)}
    for n in (1, 2, 3, 5):
        for workers in (1, 2, 4):
            seen = []
            for i in range(n):
                bins = ingest.balance(ingest.shard_ids(ids, (i, n)), lengths, workers)
                assert len(bins) == workers
                seen += [rid for b in bins for rid in b]
            assert len(seen) == len(set(seen)) == len(ids)
            assert set(seen) == set(ids)
    bins = ingest.balance(ids, lengths, 3)
    assert ingest.balance(ids[::-1], lengths, 3) == bins
    loads = [sum(lengths[r] for r in b) for b in bins]
    assert max(loads) - min(loads) <= max(lengths.values())


def test_convert_pins_workers_and_a_rerun_resumes(release, monkeypatch):
    assert release.plan() == 0
    with pytest.raises(SystemExit, match="run fetch first"):
        release.run("convert", "--cpus-per-worker", "1")
    release.hub(monkeypatch)
    assert release.run("fetch") == 0
    real = sorted(os.sched_getaffinity(0))
    cpus = real[:2]
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(cpus))
    ids = release.chunk()["ids"]
    monkeypatch.setenv("STUB_FAIL_IDS", ids[1])

    assert release.run("convert", "--cpus-per-worker", "1") == 1
    runs = release.stub_runs()
    assert len(runs) == len(cpus)
    assert sorted(i for r in runs for i in r["ids"]) == sorted(ids)
    assert sorted(c for r in runs for c in r["cpus"]) == cpus
    for run in runs:
        assert len(run["cpus"]) == 1
        env = run["env"]
        assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "1"
        assert env["OPENBLAS_NUM_THREADS"] == "1"
        assert env["CUDA_VISIBLE_DEVICES"] == "" and env["HF_HUB_OFFLINE"] == "1"
        assert env["PYTHONPATH"].split(os.pathsep)[0] == str(ingest.REPO_ROOT)
        argv = run["argv"]
        for flag in ("--skip-done", "--overwrite", "--include-head-right"):
            assert flag in argv
        # The real converter accepts exactly what the driver passes.
        from egomimic.scripts.sharpa_process.sharpa_to_zarr import build_parser

        parsed = build_parser().parse_args(argv)
        assert parsed.skip_done and parsed.overwrite and parsed.include_head_right
        assert argv[argv.index("--output-dir") + 1] == str(release.out())
        assert argv[argv.index("--dataset-root") + 1] == str(release.root.resolve())
        ids_file = Path(argv[argv.index("--recording-ids") + 1])
        assert ids_file.parent == release.state / "chunk_0000"
        assert "--tasks" not in argv and "--episodes" not in argv
    assert (release.state / "logs" / "chunk_0000" / "worker_0.log").is_file()
    assert not (release.state / "chunk_0000.lock").exists()
    entries = release.ledger("convert")
    assert entries[0]["state"] == "start" and entries[0]["details"]["todo"] == 5
    assert [e for e in entries if e["state"] == "progress"]
    end = entries[-1]
    assert end["state"] == "fail" and end["details"]["not_done"] == 1
    assert sorted(w["exit"] for w in end["details"]["workers"])[-1] == 1

    # The rerun converts only the episode that failed.
    for path in release.stub_logs.iterdir():
        path.unlink()
    monkeypatch.delenv("STUB_FAIL_IDS")
    assert release.run("convert", "--cpus-per-worker", "1") == 0
    (run,) = release.stub_runs()
    assert run["ids"] == run["converted"] == [ids[1]]
    start = [e for e in release.ledger("convert") if e["state"] == "start"][-1]
    assert (start["details"]["done"], start["details"]["todo"]) == (4, 1)
    assert release.ledger("convert")[-1]["state"] == "ok"

    # Nothing left: no worker starts.
    for path in release.stub_logs.iterdir():
        path.unlink()
    assert release.run("convert", "--cpus-per-worker", "1") == 0
    assert release.stub_runs() == []
    assert release.run("verify") == 0


def test_convert_shards_split_a_chunk(release, monkeypatch):
    assert release.plan() == 0
    release.hub(monkeypatch)
    assert release.run("fetch") == 0
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(REAL_CPUS[:1]))
    seen = []
    for shard in ("0/2", "1/2"):
        assert release.run("convert", "--shard", shard, "--cpus-per-worker", "1") == 0
        (run,) = release.stub_runs()
        seen += run["converted"]
        for path in release.stub_logs.iterdir():
            path.unlink()
    assert sorted(seen) == sorted(release.chunk()["ids"])
    assert (release.state / "chunk_0000" / "shard-1-of-2.worker_0.ids").is_file()
    logs = release.state / "logs" / "chunk_0000"
    assert (logs / "shard-0-of-2.worker_0.log").is_file()
    shards = [e["details"]["shard"] for e in release.ledger("convert")]
    assert set(shards) == {"0/2", "1/2"}
    assert release.run("verify") == 0


def test_convert_refuses_a_loaded_node(release, monkeypatch):
    assert release.plan() == 0
    release.hub(monkeypatch)
    assert release.run("fetch") == 0
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(REAL_CPUS[:1]))
    monkeypatch.setattr(os, "getloadavg", lambda: (10_000.0, 9_000.0, 8_000.0))
    with pytest.raises(SystemExit, match="ignore-load"):
        release.run("convert", "--cpus-per-worker", "1")
    assert release.ledger("convert") == []
    assert release.run("convert", "--cpus-per-worker", "1", "--ignore-load") == 0
    start = release.ledger("convert")[0]["details"]
    assert start["load"]["load1"] == 10_000.0 and start["load"]["cpu_count"]


def test_sigterm_stops_workers_and_a_rerun_resumes(release, monkeypatch, tmp_path):
    """A Slurm time limit or preemption: the driver stops its workers, records
    the stop and frees the lock; the next run picks up where it left off."""
    assert release.plan() == 0
    release.hub(monkeypatch)
    assert release.run("fetch") == 0
    env = {
        **os.environ,
        "STUB_SLEEP": "120",
        "PYTHONPATH": str(ingest.REPO_ROOT),
        "GIT_DIR": str(tmp_path / "no_git"),  # no git calls from the driver
    }
    argv = [
        "convert",
        "--state-dir",
        str(release.state),
        "--chunk",
        "chunk_0000",
        "--dataset-root",
        str(release.root),
        "--work-root",
        str(release.work),
        "--converter-cmd",
        shlex.join([sys.executable, str(release.stub)]),
        "--cpus-per-worker",
        "1",
        "--workers",
        "1",
        "--ignore-load",
        "--monitor-seconds",
        "0.05",
    ]
    driver = subprocess.Popen(
        [sys.executable, "-m", ingest.TOOL, *argv], env=env, cwd=tmp_path
    )
    try:
        deadline = time.monotonic() + 120
        while not list(release.stub_logs.glob("*.started")):
            assert driver.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        (worker,) = [int(p.stem) for p in release.stub_logs.glob("*.started")]
        driver.send_signal(signal.SIGTERM)
        assert driver.wait(timeout=120) == 143
    finally:
        if driver.poll() is None:
            driver.kill()
    with pytest.raises(ProcessLookupError):
        os.kill(worker, 0)
    end = release.ledger("convert")[-1]
    assert end["state"] == "fail" and "SIGTERM" in end["details"]["reason"]
    assert ingest.lock_files(release.state, "chunk_0000") == []

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(REAL_CPUS[:1]))
    assert release.run("convert", "--cpus-per-worker", "1") == 0
    assert release.run("verify") == 0


def test_locks_stale_live_remote_and_shards(tmp_path, monkeypatch):
    state, host = tmp_path, socket.gethostname()

    def put(**info):
        (state / "c.lock").write_text(json.dumps({"started_at": "then", **info}))

    def take(shard=(0, 1), break_lock=False):
        return ingest.acquire_lock(state, "c", shard, break_lock)

    put(host=host, pid=os.getpid())
    with pytest.raises(ingest.IngestError, match="held by"):
        take()
    path = take(break_lock=True)
    assert json.loads(path.read_text())["pid"] == os.getpid()
    ingest.release_lock(path)
    assert not path.exists()

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    put(host=host, pid=proc.pid)  # dead holder on this host
    ingest.release_lock(take())

    monkeypatch.setattr(ingest, "squeue_lists", lambda job: job == "111")
    put(host="elsewhere", pid=1, slurm_job_id="111")  # still queued
    with pytest.raises(ingest.IngestError, match="held by"):
        take()
    put(host="elsewhere", pid=1, slurm_job_id="222")  # gone from the queue
    ingest.release_lock(take())
    put(host="elsewhere", pid=1, slurm_job_id=None)  # cannot be checked
    with pytest.raises(ingest.IngestError, match="held by"):
        take()
    monkeypatch.setattr(ingest, "squeue_lists", lambda job: None)
    put(host="elsewhere", pid=1, slurm_job_id="222")  # squeue gave no answer
    with pytest.raises(ingest.IngestError, match="held by"):
        take()
    monkeypatch.setenv("SLURM_JOB_ID", "333")
    put(host="elsewhere", pid=1, slurm_job_id="333")  # this job, requeued
    ingest.release_lock(take())
    monkeypatch.delenv("SLURM_JOB_ID")

    # Shards of one split coexist; the whole chunk or another split waits.
    first, second = take((0, 2)), take((1, 2))
    assert first.name == "c.shard-0-of-2.lock" and second.exists()
    for shard in ((0, 1), (0, 3), (1, 2)):
        with pytest.raises(ingest.IngestError, match="held by"):
            take(shard)
    ingest.release_lock(first)
    ingest.release_lock(second)
    assert ingest.lock_files(state, "c") == []


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def test_verify_passes_a_complete_chunk_and_reports_bytes(release):
    assert release.plan() == 0
    release.convert_all()
    assert release.run("verify") == 0
    report = json.loads((release.state / "chunk_0000.report.json").read_text())
    assert report["passed"] and report["checked"] == report["episodes"] == 5
    assert report["frames"] == 500
    assert report["annotations"] == {
        "episodes": 5,
        "with_subtasks": 5,
        "with_skills": 5,
    }
    assert report["data_status"] == {"complete": 5}
    assert report["validator_warnings_by_check"] == {
        "intrinsics": 5,
        "left.cmd_hand_joints": 5,
    }
    assert set(report["labs"].values()) == {"sharpa_dex"}
    assert report["worker_seconds_per_frame"] == pytest.approx(0.03)
    grouped = report["bytes"]["per_frame_by_group"]
    assert grouped["rgb_front_1"] == round(5 * (ARRAYS["images.front_1"] + 2) / 500)
    assert set(grouped) == {*ingest.ARRAY_GROUPS, "numeric_and_other"}
    assert report["bytes"]["mp4_per_episode"] == 500
    assert set(report["hashes"].values()) == set(release.chunk()["ids"])
    assert release.ledger("verify")[-1]["state"] == "ok"
    # Its own earlier report is not a collision.
    assert release.run("verify") == 0


@pytest.mark.parametrize(
    "change, problem",
    [
        ({"missing": True}, "conversion.json"),
        ({"rid": "task_x/season_x/9"}, "is for"),
        ({"errors": ("left.obs_hand_joints",)}, "1 validation error"),
        # No reason, however phrased, lets a non-complete episode through.
        (
            {"status": "structural_sample", "reasons": ("no head_left video",)},
            "data_status structural_sample",
        ),
        ({"tcp_passed": False}, "tcp_check"),
        ({"mp4": False}, ".mp4"),
        ({"lab": "other_lab"}, "lab 'other_lab'"),
        ({"store_rid": "task_a/season_POC1_2026_01_01_00_00_00_train/9"}, "is for"),
        ({"report": {"checks": None}}, "missing"),
        ({"report": {"ok": True, "errors": []}, "tcp_passed": False}, "tcp_check"),
    ],
)
def test_verify_fails_a_defective_episode(release, change, problem):
    assert release.plan() == 0
    release.convert_all()
    rid = release.chunk()["ids"][0]
    name = release.records[rid]["episode_hash"]
    change = dict(change)
    edit = change.pop("report", None)
    if change.pop("missing", False):
        (release.out() / f"{name}.conversion.json").unlink()
    else:
        write_episode(release.out(), change.pop("rid", rid), name, 100, **change)
    if edit is not None:
        path = release.out() / f"{name}.validation.json"
        report = json.loads(path.read_text())
        for key, value in edit.items():
            if value is None:
                report.pop(key)
            else:
                report[key] = value
        path.write_text(json.dumps(report))
    assert release.run("verify") == 1
    report = json.loads((release.state / "chunk_0000.report.json").read_text())
    assert report["passed"] is False
    assert list(report["problems"]) == [rid]
    assert any(problem in p for p in report["problems"][rid]), report["problems"]


def test_verify_has_no_way_to_accept_a_non_complete_episode():
    with pytest.raises(SystemExit):
        ingest.main(
            ["verify", "--allow-reason", "x", "--state-dir", "s", "--chunk", "c"]
        )


def test_episode_done_needs_canonical_ids_and_a_passing_report(release):
    assert release.plan() == 0
    release.convert_all()
    rid = release.chunk()["ids"][0]
    record = release.records[rid]
    out, name = release.out(), record["episode_hash"]
    assert ingest.episode_done(out, record)
    sidecar = out / f"{name}.conversion.json"
    text = sidecar.read_text()
    # The old driver compared the slash id with vendor_recording_id.
    side = json.loads(text)
    side["vendor_recording_id"] = rid
    sidecar.write_text(json.dumps(side))
    assert not ingest.episode_done(out, record)
    sidecar.write_text(text)
    report = out / f"{name}.validation.json"
    good = report.read_text()
    report.write_text(good.replace('"passed": true', '"passed": false', 1))
    assert not ingest.episode_done(out, record)
    report.write_text(good)
    (out / f"{name}.mp4").unlink()
    assert not ingest.episode_done(out, record)


def test_a_manifest_that_breaks_the_id_contract_is_refused(release):
    rid = next(iter(release.records))
    text = release.manifest.read_text()
    release.manifest.write_text(text.replace(f'"{rid}"', f'"{rid}9"', 1))
    with pytest.raises(SystemExit, match="id problem"):
        release.plan()


def test_verify_fails_a_hash_used_by_another_chunk(release):
    assert release.plan() == 0
    release.convert_all()
    name = release.records[release.chunk()["ids"][2]]["episode_hash"]
    (release.state / "chunk_0002.report.json").write_text(
        json.dumps({"hashes": {name: "task_b/x/0"}})
    )
    assert release.run("verify") == 1
    report = json.loads((release.state / "chunk_0000.report.json").read_text())
    assert report["hash_collisions_with_other_chunks"] == {name: "chunk_0002"}


# ---------------------------------------------------------------------------
# upload, status
# ---------------------------------------------------------------------------


def test_upload_dry_run_stages_roots_and_mp4s_last_and_gates_execute(
    release, monkeypatch
):
    assert release.plan() == 0
    with pytest.raises(SystemExit, match="run verify first"):
        release.run("upload")
    release.convert_all(e0={"tcp_passed": False})
    assert release.run("verify") == 1
    with pytest.raises(SystemExit, match="did not pass"):
        release.run("upload")
    release.convert_all()
    assert release.run("verify") == 0

    def spawn(*args, **kwargs):
        raise AssertionError("upload must not start a process here")

    monkeypatch.setattr(ingest.subprocess, "run", spawn)
    monkeypatch.setattr(ingest.shutil, "which", spawn)
    # The merge script takes lab from the folder: it must be the stored lab.
    with pytest.raises(SystemExit, match="the folder must equal the lab"):
        release.run("upload", "--prefix", "processed_v3/sharpa_dex_v2")
    assert release.run("upload") == 0
    out, base = release.out(), "s3://rldb/processed_v3/sharpa_dex"
    names = [release.records[r]["episode_hash"] for r in release.chunk()["ids"]]
    phase1 = (release.state / "chunk_0000.upload.phase1.txt").read_text().splitlines()
    phase2 = (release.state / "chunk_0000.upload.phase2.txt").read_text().splitlines()
    assert sorted(phase1) == sorted(
        f'cp "{out}/{n}.zarr/{a}/*" "{base}/{n}.zarr/{a}/"'
        for n in names
        for a in ARRAYS
    )
    assert sorted(phase2) == sorted(
        [f'cp "{out}/{n}.zarr/zarr.json" "{base}/{n}.zarr/zarr.json"' for n in names]
        + [f'cp "{out}/{n}.mp4" "{base}/{n}.mp4"' for n in names]
    )
    assert release.ledger("upload") == []
    assert release.ledger("upload_dry_run")[-1]["state"] == "ok"

    with pytest.raises(SystemExit, match="licence-approved"):
        release.run("upload", "--execute")
    with pytest.raises(SystemExit, match="R2_ENDPOINT_URL"):
        release.run("upload", "--execute", "--licence-approved")
    assert release.ledger("upload") == []


def test_status_lists_stages_and_skips_a_truncated_ledger_line(release, capsys):
    assert release.plan() == 0
    release.convert_all()
    assert release.run("verify") == 0
    with (release.state / "ledger.jsonl").open("a") as f:
        f.write('{"ts": "2026-10-21T00:00:00+00:00", "chunk": "chunk_0001", "sta')
    assert [e["stage"] for e in release.ledger()] == ["verify"]
    capsys.readouterr()
    assert release.run("status") == 0
    out = capsys.readouterr().out
    rows = {line.split()[0]: line.split() for line in out.splitlines() if line}
    assert rows["chunk_0000"][5:] == ["-", "-", "ok", "-", "-"]
    assert rows["chunk_0001"][5:] == ["-"] * 5
    assert "verify   ok: 1 chunk(s), 5 episode(s), 500 frames" in out
    assert release.run("status", "--json") == 0
    status = json.loads(capsys.readouterr().out)
    assert status["done"]["verify_episodes"] == 5


def test_manifest_reader_agrees_with_sharpadex_manifest(release):
    from egomimic.scripts.sharpa_process.sharpadex_manifest import read_manifest

    header, records = read_manifest(release.manifest)
    mine_header, mine = ingest.read_manifest(release.manifest)
    assert mine_header == header and set(mine) == set(records)
    for rid, record in mine.items():
        for key, value in record.items():
            assert getattr(records[rid], key) == value
