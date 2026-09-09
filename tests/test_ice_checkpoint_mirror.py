#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = (
    Path(__file__).parents[1] / "scripts" / "ice" / "ice_checkpoint_mirror.py"
)
SBATCH_PATH = (
    Path(__file__).parents[1] / "scripts" / "ice" / "ice_checkpoint_mirror.sbatch"
)
SPEC = importlib.util.spec_from_file_location("ice_checkpoint_mirror", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_validator(root: Path) -> Path:
    validator = root / "validate_checkpoint.py"
    validator.write_text(
        "#!/usr/bin/env python3\n"
        "import json,sys\n"
        "try:\n"
        " data=json.load(open(sys.argv[1]))\n"
        "except Exception:\n"
        " raise SystemExit(2)\n"
        "if not data.get('valid', True): raise SystemExit(3)\n"
        "print(json.dumps({'global_step':data['global_step'],'run_id':data.get('run_id','run')}))\n"
    )
    validator.chmod(0o700)
    return validator


def write_checkpoint(path: Path, step: int, *, valid: bool = True) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"global_step": step, "valid": valid, "run_id": "run"}))
    os.replace(temporary, path)


def write_manifest(
    path: Path,
    run: Path,
    validator: Path,
    *,
    retain: int = 2,
    sentinel: Path | None = None,
) -> None:
    row = {
        "id": "run",
        "local_root": str(run),
        "checkpoint_glob": "*.ckpt",
        "validator": str(validator),
        "remote_dir": "/remote/archive/run",
        "retain_local": retain,
    }
    if sentinel is not None:
        row["completion_sentinel"] = str(sentinel)
    path.write_text(json.dumps({"schema_version": 1, "runs": [row]}))


def mirror_args(manifest: Path, state: Path, scratch: Path) -> list[str]:
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
        "--prune-after-verify",
        "--once",
    ]


class CheckpointMetadataTest(unittest.TestCase):
    def test_cpu_wrapper_forwards_configured_du_timeout(self):
        wrapper = SBATCH_PATH.read_text()
        self.assertIn('"${ICE_MIRROR_PYTHON}" "${ICE_MIRROR_SCRIPT}"', wrapper)
        self.assertIn('if [[ -n "${ICE_MIRROR_DU_TIMEOUT_SECONDS:-}" ]]', wrapper)
        self.assertIn(
            'ARGS+=(--du-timeout-seconds "${ICE_MIRROR_DU_TIMEOUT_SECONDS}")',
            wrapper,
        )

    def test_stable_validation_requires_json_global_step(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            checkpoint = root / "x.ckpt"
            write_checkpoint(checkpoint, 7)
            validator = make_validator(root)
            info = MODULE.stable_valid_info(checkpoint.resolve(), validator)
            self.assertIsNotNone(info)
            self.assertEqual(info.global_step, 7)
            self.assertEqual(info.sha256, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            bad_validator = root / "bad_validator"
            bad_validator.write_text("#!/bin/sh\nprintf 'not-json\\n'\n")
            bad_validator.chmod(0o700)
            self.assertIsNone(MODULE.stable_valid_info(checkpoint.resolve(), bad_validator))

    def test_specific_absolute_rejects_relative_path(self):
        with self.assertRaisesRegex(SystemExit, "specific absolute"):
            MODULE.specific_absolute(Path("relative/run/path"), "test")


class MirrorSafetyTest(unittest.TestCase):
    def test_invalid_newer_files_do_not_count_toward_two_valid_retained(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            scratch = root / "scratch"
            run = scratch / "project" / "run"
            state = scratch / "state" / "mirror"
            run.mkdir(parents=True)
            state.mkdir(parents=True)
            validator = make_validator(root)
            first = run / "step-1.ckpt"
            second = run / "step-2.ckpt"
            corrupt_a = run / "step-100.ckpt"
            corrupt_b = run / "step-101.ckpt"
            write_checkpoint(first, 1)
            write_checkpoint(second, 2)
            write_checkpoint(corrupt_a, 100, valid=False)
            write_checkpoint(corrupt_b, 101, valid=False)
            now = time.time()
            os.utime(corrupt_a, (now + 10, now + 10))
            os.utime(corrupt_b, (now + 20, now + 20))
            manifest = root / "manifest.json"
            write_manifest(manifest, run, validator)
            remote: dict[str, str] = {}

            def fake_mirror(path, digest, remote_dir, host, ssh, rsync, rsync_rsh):
                destination = f"/remote/{digest}"
                remote[destination] = digest
                return destination

            with (
                mock.patch.object(MODULE, "mirror_one", side_effect=fake_mirror),
                mock.patch.object(MODULE, "remote_sha", side_effect=lambda ssh, host, path: remote.get(path)),
                mock.patch.object(MODULE, "scratch_bytes", return_value=100),
            ):
                code = MODULE.main(mirror_args(manifest, state, scratch))
            self.assertEqual(code, 1, "invalid candidates must keep completion unresolved")
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            self.assertTrue(corrupt_a.exists())
            self.assertTrue(corrupt_b.exists())

    def test_prune_rechecks_remote_sha_immediately(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            scratch = root / "scratch"
            run = scratch / "project" / "run"
            state = scratch / "state" / "mirror"
            run.mkdir(parents=True)
            state.mkdir(parents=True)
            validator = make_validator(root)
            checkpoints = []
            for step in (1, 2, 3):
                path = run / f"step-{step}.ckpt"
                write_checkpoint(path, step)
                checkpoints.append(path)
            manifest = root / "manifest.json"
            write_manifest(manifest, run, validator)
            files = {}
            remote = {}
            for path in checkpoints:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                remote_path = f"/remote/{digest}"
                files[str(path.resolve())] = {
                    "sha256": digest,
                    "run_id": "run",
                    "remote_path": remote_path,
                    "remote_verified": True,
                }
                remote[remote_path] = digest
            (state / "mirror-state.json").write_text(
                json.dumps({"schema_version": 1, "files": files})
            )
            oldest_remote = files[str(checkpoints[0].resolve())]["remote_path"]
            calls = {oldest_remote: 0}

            def fake_remote_sha(ssh, host, path):
                if path == oldest_remote:
                    calls[path] += 1
                    return remote[path] if calls[path] == 1 else None
                return remote.get(path)

            with (
                mock.patch.object(MODULE, "mirror_one", side_effect=AssertionError("no upload expected")),
                mock.patch.object(MODULE, "remote_sha", side_effect=fake_remote_sha),
                mock.patch.object(MODULE, "scratch_bytes", return_value=100),
            ):
                code = MODULE.main(mirror_args(manifest, state, scratch))
            self.assertEqual(code, 1)
            self.assertTrue(checkpoints[0].exists(), "local file was pruned after remote disappeared")
            final_state = json.loads((state / "mirror-state.json").read_text())
            self.assertFalse(final_state["files"][str(checkpoints[0].resolve())]["remote_verified"])

    def test_du_failure_is_recorded_without_uncaught_exit(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            scratch = root / "scratch"
            run = scratch / "project" / "run"
            state = scratch / "state" / "mirror"
            run.mkdir(parents=True)
            state.mkdir(parents=True)
            validator = make_validator(root)
            manifest = root / "manifest.json"
            write_manifest(manifest, run, validator)
            with mock.patch.object(MODULE, "scratch_bytes", side_effect=RuntimeError("du failed")):
                code = MODULE.main(mirror_args(manifest, state, scratch))
            self.assertEqual(code, 1)
            pressure = json.loads((state / "storage-pressure.json").read_text())
            self.assertEqual(pressure["pressure"], "unknown")
            self.assertIn("du failed", pressure["error"])

    def test_deterministic_resumable_temp_and_full_digest_name(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            checkpoint = root / "step-9.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            remote_commands = []
            local_commands = []
            sha_results = iter((None, digest, digest))

            def fake_remote_command(ssh, host, argv, capture=False):
                remote_commands.append(argv)
                return ""

            def fake_run_checked(command, capture=False, timeout=None):
                local_commands.append(command)
                return subprocess.CompletedProcess(command, 0, "", "")

            with (
                mock.patch.object(MODULE, "remote_sha", side_effect=lambda *args: next(sha_results)),
                mock.patch.object(MODULE, "remote_command", side_effect=fake_remote_command),
                mock.patch.object(MODULE, "run_checked", side_effect=fake_run_checked),
            ):
                final = MODULE.mirror_one(
                    checkpoint,
                    digest,
                    "/remote/archive/run",
                    "Skynet",
                    "ssh",
                    "rsync",
                    None,
                )
            self.assertIn(digest, final)
            self.assertEqual(local_commands[0][-1], f"Skynet:{final}.partial")
            self.assertNotRegex(local_commands[0][-1], r"\.partial\.\d+$")
            self.assertIn(["mv", "-n", "--", f"{final}.partial", final], remote_commands)

    def test_completion_sentinel_stops_monitor_after_final_mirror(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            scratch = root / "scratch"
            run = scratch / "project" / "run"
            state = scratch / "state" / "mirror"
            run.mkdir(parents=True)
            state.mkdir(parents=True)
            validator = make_validator(root)
            checkpoint = run / "step-10.ckpt"
            write_checkpoint(checkpoint, 10)
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
            write_manifest(manifest, run, validator, sentinel=sentinel)
            remote = {}

            def fake_mirror(path, digest, remote_dir, host, ssh, rsync, rsync_rsh):
                destination = f"/remote/{digest}"
                remote[destination] = digest
                return destination

            with (
                mock.patch.object(MODULE, "mirror_one", side_effect=fake_mirror),
                mock.patch.object(MODULE, "remote_sha", side_effect=lambda ssh, host, path: remote.get(path)),
                mock.patch.object(MODULE, "scratch_bytes", return_value=100),
            ):
                code = MODULE.main(mirror_args(manifest, state, scratch))
            self.assertEqual(code, 0)
            complete = json.loads((state / "mirror-complete.json").read_text())
            self.assertEqual(complete["status"], "COMPLETE")


if __name__ == "__main__":
    unittest.main()
