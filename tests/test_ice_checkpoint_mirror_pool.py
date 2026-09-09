#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ICE_DIR = Path(__file__).parents[1] / "scripts" / "ice"
POOL_PATH = ICE_DIR / "ice_checkpoint_mirror_pool.py"
SBATCH_PATH = ICE_DIR / "ice_checkpoint_mirror_pool.sbatch"
sys.path.insert(0, str(ICE_DIR))
SPEC = importlib.util.spec_from_file_location("ice_checkpoint_mirror_pool", POOL_PATH)
POOL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = POOL
SPEC.loader.exec_module(POOL)


def make_validator(root: Path) -> Path:
    validator = root / "validate_checkpoint.py"
    validator.write_text(
        "#!/usr/bin/env python3\n"
        "import json,sys\n"
        "data=json.load(open(sys.argv[1]))\n"
        "print(json.dumps({'global_step':data['global_step']}))\n"
    )
    validator.chmod(0o700)
    return validator


def make_layout(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    scratch = root / "scratch"
    run = scratch / "project" / "run"
    state = scratch / "state" / "mirror"
    run.mkdir(parents=True)
    state.mkdir(parents=True)
    validator = make_validator(root)
    checkpoint = run / "step-10.ckpt"
    checkpoint.write_text(json.dumps({"global_step": 10}))
    sentinel = run / "COMPLETE.json"
    sentinel.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "checkpoint": {
                    "global_step": 10,
                    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                },
            }
        )
    )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runs": [
                    {
                        "id": "run",
                        "local_root": str(run),
                        "checkpoint_glob": "*.ckpt",
                        "validator": str(validator),
                        "remote_dir": "/remote/archive/run",
                        "retain_local": 2,
                        "completion_sentinel": str(sentinel),
                    }
                ],
            }
        )
    )
    return scratch, run, state, checkpoint, manifest


def worker_args(manifest: Path, state: Path, scratch: Path, index: int = 0) -> list[str]:
    return [
        "--manifest",
        str(manifest),
        "--state-dir",
        str(state),
        "--scratch-root",
        str(scratch),
        "--quota-bytes",
        "1000000",
        "--remote-host",
        "Skynet",
        "--worker-index",
        str(index),
        "--worker-count",
        "2",
        "--once",
    ]


class MirrorPoolTest(unittest.TestCase):
    def test_wrapper_requires_matching_array_and_distinct_nodes(self):
        wrapper = SBATCH_PATH.read_text()
        self.assertIn("#SBATCH --exclusive", wrapper)
        self.assertIn("SLURM_ARRAY_TASK_COUNT", wrapper)
        self.assertIn("SLURM_ARRAY_TASK_ID", wrapper)
        self.assertIn("ICE_MIRROR_POOL_SIZE", wrapper)
        self.assertIn('"${ICE_MIRROR_PYTHON}" "${ICE_MIRROR_POOL_SCRIPT}"', wrapper)
        self.assertIn("ICE_MIRROR_INVENTORY_ROOT", wrapper)

    def test_nonblocking_claim_allows_only_one_worker(self):
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw) / "shared" / "state"
            checkpoint = Path(raw) / "shared" / "run" / "step.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            with POOL.claim(state, checkpoint) as first:
                self.assertIsNotNone(first)
                with POOL.claim(state, checkpoint) as second:
                    self.assertIsNone(second)
            with POOL.claim(state, checkpoint) as third:
                self.assertIsNotNone(third)

    def test_pool_mirrors_and_publishes_completion(self):
        with tempfile.TemporaryDirectory() as raw:
            scratch, _run, state, checkpoint, manifest = make_layout(Path(raw))
            remote: dict[str, str] = {}

            def fake_mirror(
                path, digest, remote_dir, host, ssh, rsync, rsync_rsh, temporary_tag=None
            ):
                self.assertEqual(temporary_tag, POOL.task_key(path)[:24])
                destination = f"/remote/{digest}"
                remote[destination] = digest
                return destination

            with (
                mock.patch.object(POOL.core, "mirror_one", side_effect=fake_mirror),
                mock.patch.object(
                    POOL.core,
                    "remote_sha",
                    side_effect=lambda ssh, host, path: remote.get(path),
                ),
                mock.patch.object(POOL.core, "scratch_bytes", return_value=100),
            ):
                code = POOL.main(worker_args(manifest, state, scratch))
            self.assertEqual(code, 0)
            saved = json.loads((state / "mirror-state.json").read_text())
            entry = saved["files"][str(checkpoint.resolve())]
            self.assertTrue(entry["remote_verified"])
            complete = json.loads((state / "mirror-complete.json").read_text())
            self.assertEqual(complete["worker_count"], 2)

    def test_nonmaintenance_worker_never_prunes_or_completes(self):
        with tempfile.TemporaryDirectory() as raw:
            scratch, _run, state, _checkpoint, manifest = make_layout(Path(raw))
            remote: dict[str, str] = {}

            def fake_mirror(
                path, digest, remote_dir, host, ssh, rsync, rsync_rsh, temporary_tag=None
            ):
                destination = f"/remote/{digest}"
                remote[destination] = digest
                return destination

            with (
                mock.patch.object(POOL.core, "mirror_one", side_effect=fake_mirror),
                mock.patch.object(
                    POOL.core,
                    "remote_sha",
                    side_effect=lambda ssh, host, path: remote.get(path),
                ),
                mock.patch.object(
                    POOL,
                    "maintain",
                    side_effect=AssertionError("worker one ran maintenance"),
                ),
            ):
                code = POOL.main(worker_args(manifest, state, scratch, index=1))
            self.assertEqual(code, 0)
            self.assertFalse((state / "mirror-complete.json").exists())

    def test_completion_requires_exact_terminal_checkpoint(self):
        with tempfile.TemporaryDirectory() as raw:
            scratch, run, state, checkpoint, manifest = make_layout(Path(raw))
            sentinel = run / "COMPLETE.json"
            payload = json.loads(sentinel.read_text())
            payload["checkpoint"]["global_step"] = 999
            payload["checkpoint"]["sha256"] = "a" * 64
            sentinel.write_text(json.dumps(payload))
            remote: dict[str, str] = {}

            def fake_mirror(
                path, digest, remote_dir, host, ssh, rsync, rsync_rsh, temporary_tag=None
            ):
                destination = f"/remote/{digest}"
                remote[destination] = digest
                return destination

            with (
                mock.patch.object(POOL.core, "mirror_one", side_effect=fake_mirror),
                mock.patch.object(
                    POOL.core,
                    "remote_sha",
                    side_effect=lambda ssh, host, path: remote.get(path),
                ),
                mock.patch.object(POOL.core, "scratch_bytes", return_value=100),
            ):
                code = POOL.main(worker_args(manifest, state, scratch))
            self.assertEqual(code, 0)
            self.assertFalse((state / "mirror-complete.json").exists())
            saved = json.loads((state / "mirror-state.json").read_text())
            self.assertTrue(saved["files"][str(checkpoint.resolve())]["remote_verified"])

    def test_worker_count_and_index_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            scratch, _run, state, _checkpoint, manifest = make_layout(Path(raw))
            args = worker_args(manifest, state, scratch)
            args[args.index("--worker-count") + 1] = "1"
            with self.assertRaisesRegex(SystemExit, "worker-count"):
                POOL.main(args)


if __name__ == "__main__":
    unittest.main()
