#!/usr/bin/env python3

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = (
    Path(__file__).parents[1] / "scripts" / "ice" / "ice_requeue_runner.py"
)
SPEC = importlib.util.spec_from_file_location("ice_requeue_runner", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_validator(root: Path) -> Path:
    validator = root / "validate_checkpoint.py"
    validator.write_text(
        "#!/usr/bin/env python3\n"
        "import json,os,sys\n"
        "log=os.environ.get('VALIDATOR_LOG')\n"
        "if log:\n"
        " with open(log,'a') as stream: stream.write(os.path.realpath(sys.argv[1])+'\\n')\n"
        "try:\n"
        " data=json.load(open(sys.argv[1]))\n"
        "except Exception:\n"
        " raise SystemExit(2)\n"
        "if not data.get('valid', True): raise SystemExit(3)\n"
        "metadata={'global_step': data['global_step']}\n"
        "for key in ('run_id','wandb_run_id','config_sha256','source_commit'):\n"
        " if key in data: metadata[key]=data[key]\n"
        "print(json.dumps(metadata, sort_keys=True))\n"
    )
    validator.chmod(0o700)
    return validator


def write_checkpoint(path: Path, step: int, *, valid: bool = True, run_id: str = "run") -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "global_step": step,
                "valid": valid,
                "run_id": run_id,
                "wandb_run_id": "wandb-run",
            }
        )
    )
    os.replace(temporary, path)


def make_signaling_scancel(root: Path) -> Path:
    executable = root / "scancel"
    executable.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" > \"$ICE_TEST_SCANCEL_LOG\"\n"
        "kill -USR2 \"$(cat \"$CHILD_PID_FILE\")\"\n"
    )
    executable.chmod(0o700)
    return executable


def base_args(
    state_dir: Path,
    checkpoint_glob: str,
    validator: Path,
    command: list[str],
    *,
    owner: str = "child",
    initial_checkpoint: Path | None = None,
    scancel: Path | str = "/usr/bin/true",
) -> list[str]:
    result = [
        "--state-dir",
        str(state_dir),
    ]
    if initial_checkpoint is not None:
        result.extend(["--initial-checkpoint", str(initial_checkpoint)])
    result.extend(
        [
            "--checkpoint-glob",
            checkpoint_glob,
            "--checkpoint-validator",
            str(validator),
            "--checkpoint-signal",
            "USR2",
            "--checkpoint-forwarding",
            "slurm-steps",
            "--requeue-owner",
            owner,
            "--scancel",
            str(scancel),
            "--poll-seconds",
            "0.05",
        ]
    )
    if owner == "runner":
        result.append("--confirm-child-requeue-disabled")
    result.extend(["--", *command])
    return result


class CheckpointSelectionTest(unittest.TestCase):
    def test_semantic_global_step_beats_newer_mtime(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            validator = make_validator(root)
            step_60 = root / "epoch-6-step-60000.ckpt"
            step_40 = root / "staged-step-40000.ckpt"
            write_checkpoint(step_60, 60_000)
            write_checkpoint(step_40, 40_000)
            now = time.time()
            os.utime(step_60, (now - 20, now - 20))
            os.utime(step_40, (now, now))
            selected = MODULE.select_checkpoint("*.ckpt", root, validator)
            self.assertIsNotNone(selected)
            self.assertEqual(selected.path, step_60.resolve())
            self.assertEqual(selected.global_step, 60_000)

    def test_invalid_and_partial_candidates_are_ignored(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            validator = make_validator(root)
            valid = root / "step-10.ckpt"
            invalid = root / "step-20.ckpt"
            partial = root / "step-30.ckpt.partial"
            write_checkpoint(valid, 10)
            write_checkpoint(invalid, 20, valid=False)
            write_checkpoint(partial, 30)
            selected = MODULE.select_checkpoint("*.ckpt*", root, validator)
            self.assertEqual(selected.path, valid.resolve())

    def test_multiple_roots_select_highest_semantic_step(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            staged = root / "staged"
            live = root / "live"
            staged.mkdir()
            live.mkdir()
            validator = make_validator(root)
            source = staged / "step-40000.ckpt"
            resumed = live / "step-42000.ckpt"
            write_checkpoint(source, 40_000)
            write_checkpoint(resumed, 42_000)
            selected = MODULE.select_checkpoint(
                [str(staged / "*.ckpt"), str(live / "*.ckpt")],
                root,
                validator,
            )
            self.assertEqual(selected.path, resumed.resolve())

    def test_fresh_checkpoint_requires_strictly_higher_global_step(self):
        def info(path: str, step: int, digest: str) -> MODULE.CheckpointInfo:
            return MODULE.CheckpointInfo(
                path=Path(path),
                sha256=digest,
                global_step=step,
                size=1,
                mtime_ns=1,
                inode=1,
                metadata={"global_step": step},
            )

        baseline = info("/tmp/run/a.ckpt", 10, "a" * 64)
        self.assertFalse(MODULE.is_fresh(None, baseline))
        self.assertFalse(MODULE.is_fresh(info("/tmp/run/b.ckpt", 9, "b" * 64), baseline))
        self.assertFalse(MODULE.is_fresh(info("/tmp/run/b.ckpt", 10, "a" * 64), baseline))
        self.assertFalse(MODULE.is_fresh(info("/tmp/run/a.ckpt", 10, "b" * 64), baseline))
        self.assertTrue(MODULE.is_fresh(info("/tmp/run/a.ckpt", 11, "a" * 64), baseline))

    def test_semantic_maximum_same_step_fork_is_rejected(self):
        def info(path: str, digest: str) -> MODULE.CheckpointInfo:
            return MODULE.CheckpointInfo(
                path=Path(path),
                sha256=digest,
                global_step=10,
                size=1,
                mtime_ns=1,
                inode=1,
                metadata={"global_step": 10},
            )

        with self.assertRaisesRegex(SystemExit, "semantic-maximum checkpoints"):
            MODULE.newest_checkpoint(
                info("/tmp/run/a.ckpt", "a" * 64),
                info("/tmp/run/b.ckpt", "b" * 64),
            )

    def test_checkpoint_forwarding_is_exact_scancel_and_never_killpg(self):
        completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with mock.patch.dict(
            os.environ,
            {"SCANCEL_BATCH": "1", "SCANCEL_FULL": "1", "SLURM_CLUSTERS": "wrong"},
        ):
            with (
                mock.patch.object(MODULE.subprocess, "run", return_value=completed) as run,
                mock.patch.object(MODULE.os, "killpg") as killpg,
            ):
                result = MODULE.forward_checkpoint_to_slurm_steps(
                    "/mock/bin/scancel",
                    "12345",
                    signal.SIGUSR2,
                )
        self.assertIs(result, completed)
        run.assert_called_once()
        positional, keyword = run.call_args
        self.assertEqual(positional, (["/mock/bin/scancel", "--signal=USR2", "12345"],))
        self.assertEqual(keyword["text"], True)
        self.assertIs(keyword["stdout"], subprocess.PIPE)
        self.assertIs(keyword["stderr"], subprocess.PIPE)
        self.assertEqual(keyword["check"], False)
        self.assertNotIn("SCANCEL_BATCH", keyword["env"])
        self.assertNotIn("SCANCEL_FULL", keyword["env"])
        self.assertNotIn("SLURM_CLUSTERS", keyword["env"])
        killpg.assert_not_called()


class RunnerSafetyTest(unittest.TestCase):
    def test_signal_handlers_are_installed_before_child_spawn(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            validator = make_validator(root)
            checkpoint_path = root / "step-1.ckpt"
            write_checkpoint(checkpoint_path, 1)
            checkpoint = MODULE.stable_checkpoint_info(checkpoint_path.resolve(), validator)
            self.assertIsNotNone(checkpoint)
            installed: list[signal.Signals] = []
            gate_readers: list[int] = []

            class FinishedChild:
                pid = 9876

                @staticmethod
                def poll() -> int:
                    while gate_readers:
                        os.close(gate_readers.pop())
                    return 0

            def install(received: signal.Signals, _handler: object) -> object:
                installed.append(received)
                return signal.SIG_DFL

            def spawn(*_args: object, **kwargs: object) -> FinishedChild:
                self.assertEqual(
                    installed,
                    [signal.SIGUSR1, signal.SIGTERM, signal.SIGINT],
                )
                pass_fds = kwargs["pass_fds"]
                self.assertIsInstance(pass_fds, tuple)
                gate_readers.append(os.dup(pass_fds[0]))
                return FinishedChild()

            def validate(*_args: object, **_kwargs: object) -> MODULE.CheckpointInfo:
                self.assertEqual(
                    installed,
                    [signal.SIGUSR1, signal.SIGTERM, signal.SIGINT],
                )
                return checkpoint

            args = base_args(state, str(root / "*.ckpt"), validator, ["unused-child"])
            with (
                mock.patch.dict(
                    os.environ,
                    {"SLURM_JOB_ID": "990", "SLURM_RESTART_COUNT": "0"},
                    clear=False,
                ),
                mock.patch.object(MODULE.signal, "signal", side_effect=install),
                mock.patch.object(MODULE, "stable_checkpoint_info", side_effect=validate),
                mock.patch.object(MODULE.subprocess, "Popen", side_effect=spawn) as popen,
            ):
                self.assertEqual(MODULE.main(args), 0)
            popen.assert_called_once()

    def test_usr1_during_strict_discovery_requeues_without_starting_work(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            checkpoint = root / "step-1.ckpt"
            write_checkpoint(checkpoint, 1)
            early_signal_marker = root / "early-signal-sent"
            validator = root / "early_signal_validator.py"
            validator.write_text(
                "#!/usr/bin/env python3\n"
                "import json,os,signal,sys,time\n"
                "marker=os.environ['EARLY_SIGNAL_MARKER']\n"
                "if not os.path.exists(marker):\n"
                " open(marker,'w').write('sent')\n"
                " os.kill(os.getppid(),signal.SIGUSR1)\n"
                " time.sleep(0.1)\n"
                "with open(sys.argv[1]) as stream: data=json.load(stream)\n"
                "print(json.dumps({'global_step':data['global_step'],'run_id':data['run_id']}))\n"
            )
            validator.chmod(0o700)
            child_started = root / "child-started"
            child = root / "child.py"
            child.write_text(
                "import os\n"
                "open(os.environ['CHILD_STARTED'],'w').write('unsafe')\n"
            )
            attempt = state / "requeue" / "job-12400-restart-000" / "attempt.json"
            scancel_log = root / "scancel.log"
            fake_scancel = root / "scancel"
            fake_scancel.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$*\" > \"$ICE_TEST_SCANCEL_LOG\"\n"
            )
            fake_scancel.chmod(0o700)
            scontrol_log = root / "scontrol.log"
            fake_scontrol = root / "scontrol"
            fake_scontrol.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$*\" > \"$SCONTROL_LOG\"\n"
            )
            fake_scontrol.chmod(0o700)
            env = os.environ.copy()
            env.update(
                {
                    "SLURM_JOB_ID": "12400",
                    "SLURM_RESTART_COUNT": "0",
                    "EARLY_SIGNAL_MARKER": str(early_signal_marker),
                    "CHILD_STARTED": str(child_started),
                    "ICE_TEST_SCANCEL_LOG": str(scancel_log),
                    "SCONTROL_LOG": str(scontrol_log),
                }
            )
            args = base_args(
                state,
                str(root / "*.ckpt"),
                validator,
                [sys.executable, str(child)],
                owner="runner",
                scancel=fake_scancel,
            )
            args[args.index("--") : args.index("--")] = [
                "--scontrol",
                str(fake_scontrol),
            ]
            result = subprocess.run(
                [sys.executable, str(MODULE_PATH), *args],
                env=env,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(early_signal_marker.is_file())
            self.assertFalse(child_started.exists())
            self.assertFalse(scancel_log.exists())
            self.assertEqual(scontrol_log.read_text().strip(), "requeue 12400")
            final = json.loads(attempt.read_text())
            self.assertEqual(final["status"], "EARLY_BOUNDARY_REQUEUE_ACCEPTED")
            self.assertNotIn("child_pid", final)
            proof_path = Path(final["early_boundary_proof"])
            proof = json.loads(proof_path.read_text())
            self.assertEqual(
                final["early_boundary_proof_sha256"],
                hashlib.sha256(proof_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                proof["status"],
                "EARLY_BOUNDARY_NO_WORK_CHECKPOINT_AUTHENTICATED",
            )
            self.assertFalse(proof["child_spawned"])
            self.assertFalse(proof["optimizer_work_performed"])
            self.assertFalse(proof["checkpoint_signal_forwarded"])
            self.assertEqual(proof["checkpoint"]["global_step"], 1)
            events = [
                json.loads(line)
                for line in (attempt.parent / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in events].count(
                    "early_boundary_no_work_checkpoint_authenticated"
                ),
                1,
            )

    def test_incremental_restart_revalidates_high_water_and_only_new_candidates(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            validator = make_validator(root)
            checkpoints = [root / f"step-{step}.ckpt" for step in (10, 20, 30)]
            for step, checkpoint in zip((10, 20, 30), checkpoints):
                write_checkpoint(checkpoint, step)
            log = root / "validator.log"
            args = base_args(state, str(root / "*.ckpt"), validator, ["/bin/true"])
            args.insert(args.index("--"), "--dry-run")

            with mock.patch.dict(
                os.environ,
                {"SLURM_RESTART_COUNT": "0", "VALIDATOR_LOG": str(log)},
                clear=False,
            ):
                self.assertEqual(MODULE.main(args), 0)
                first = log.read_text().splitlines()
                self.assertEqual(set(first), {str(path.resolve()) for path in checkpoints})
                self.assertEqual(len(first), 3)

                log.write_text("")
                self.assertEqual(MODULE.main(args), 0)
                self.assertEqual(log.read_text().splitlines(), [str(checkpoints[-1].resolve())])

                newer = root / "step-40.ckpt"
                write_checkpoint(newer, 40)
                log.write_text("")
                self.assertEqual(MODULE.main(args), 0)
                self.assertEqual(
                    set(log.read_text().splitlines()),
                    {str(checkpoints[-1].resolve()), str(newer.resolve())},
                )

                log.write_text("")
                self.assertEqual(MODULE.main(args), 0)
                self.assertEqual(log.read_text().splitlines(), [str(newer.resolve())])

    def test_unbound_observation_index_forces_validator_authoritative_rescan(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            validator = make_validator(root)
            checkpoints = [root / f"step-{step}.ckpt" for step in (10, 20, 30)]
            for step, checkpoint in zip((10, 20, 30), checkpoints):
                write_checkpoint(checkpoint, step)
            log = root / "validator.log"
            args = base_args(state, str(root / "*.ckpt"), validator, ["/bin/true"])
            args.insert(args.index("--"), "--dry-run")
            with mock.patch.dict(
                os.environ,
                {"SLURM_RESTART_COUNT": "0", "VALIDATOR_LOG": str(log)},
                clear=False,
            ):
                self.assertEqual(MODULE.main(args), 0)
                index = state / MODULE.OBSERVATION_INDEX_NAME
                index.write_text(index.read_text() + "\n")
                log.write_text("")
                self.assertEqual(MODULE.main(args), 0)
            self.assertEqual(
                set(log.read_text().splitlines()),
                {str(path.resolve()) for path in checkpoints},
            )

    def test_validator_change_invalidates_observation_index(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            validator = make_validator(root)
            checkpoints = [root / f"step-{step}.ckpt" for step in (10, 20)]
            for step, checkpoint in zip((10, 20), checkpoints):
                write_checkpoint(checkpoint, step)
            log = root / "validator.log"
            args = base_args(state, str(root / "*.ckpt"), validator, ["/bin/true"])
            args.insert(args.index("--"), "--dry-run")
            with mock.patch.dict(
                os.environ,
                {"SLURM_RESTART_COUNT": "0", "VALIDATOR_LOG": str(log)},
                clear=False,
            ):
                self.assertEqual(MODULE.main(args), 0)
                validator.write_text(validator.read_text() + "# authority revision\n")
                log.write_text("")
                self.assertEqual(MODULE.main(args), 0)
            self.assertEqual(
                set(log.read_text().splitlines()),
                {str(path.resolve()) for path in checkpoints},
            )

    def test_validator_change_during_validation_refuses_state_commit(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            checkpoint = root / "step-10.ckpt"
            write_checkpoint(checkpoint, 10)
            marker = root / "validator-mutated"
            validator = root / "mutating_validator.py"
            validator.write_text(
                "#!/usr/bin/env python3\n"
                "import json,os,sys\n"
                "marker=os.environ['VALIDATOR_MUTATION_MARKER']\n"
                "if not os.path.exists(marker):\n"
                " open(marker,'w').write('mutated')\n"
                " with open(__file__,'a') as stream: stream.write('# changed\\n')\n"
                "with open(sys.argv[1]) as stream: data=json.load(stream)\n"
                "print(json.dumps({'global_step':data['global_step'],'run_id':data['run_id']}))\n"
            )
            validator.chmod(0o700)
            args = base_args(state, str(root / "*.ckpt"), validator, ["/bin/true"])
            args.insert(args.index("--"), "--dry-run")
            with mock.patch.dict(
                os.environ,
                {
                    "SLURM_RESTART_COUNT": "0",
                    "VALIDATOR_MUTATION_MARKER": str(marker),
                },
                clear=False,
            ):
                with self.assertRaisesRegex(SystemExit, "validator changed after validation"):
                    MODULE.main(args)
            self.assertTrue(marker.is_file())
            self.assertFalse((state / MODULE.OBSERVATION_INDEX_NAME).exists())
            self.assertFalse((state / "checkpoint-high-water.json").exists())

    def test_failed_candidate_is_not_cached_when_sidecar_appears(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            validator = root / "sidecar_validator.py"
            validator.write_text(
                "#!/usr/bin/env python3\n"
                "import json,os,sys\n"
                "with open(os.environ['VALIDATOR_LOG'],'a') as stream:\n"
                " stream.write(os.path.realpath(sys.argv[1])+'\\n')\n"
                "if not os.path.isfile(sys.argv[1]+'.proof'): raise SystemExit(4)\n"
                "with open(sys.argv[1]) as stream: data=json.load(stream)\n"
                "print(json.dumps({'global_step':data['global_step'],'run_id':data['run_id']}))\n"
            )
            validator.chmod(0o700)
            lower = root / "step-10.ckpt"
            higher = root / "step-20.ckpt"
            write_checkpoint(lower, 10)
            write_checkpoint(higher, 20)
            Path(str(lower) + ".proof").write_text("ready")
            original_higher_stat = higher.stat()
            log = root / "validator.log"
            args = base_args(state, str(root / "*.ckpt"), validator, ["/bin/true"])
            args.insert(args.index("--"), "--dry-run")
            with mock.patch.dict(
                os.environ,
                {"SLURM_RESTART_COUNT": "0", "VALIDATOR_LOG": str(log)},
                clear=False,
            ):
                self.assertEqual(MODULE.main(args), 0)
                first = json.loads(
                    (state / "requeue" / "job-dry-run-restart-000" / "attempt.json").read_text()
                )
                self.assertEqual(first["checkpoint"]["global_step"], 10)

                Path(str(higher) + ".proof").write_text("ready")
                self.assertEqual(
                    MODULE.stat_identity(higher.stat()),
                    MODULE.stat_identity(original_higher_stat),
                )
                log.write_text("")
                self.assertEqual(MODULE.main(args), 0)
            self.assertIn(str(higher.resolve()), log.read_text().splitlines())
            final = json.loads(
                (state / "requeue" / "job-dry-run-restart-000" / "attempt.json").read_text()
            )
            self.assertEqual(final["checkpoint"]["global_step"], 20)

    def test_atomic_replacement_with_preserved_size_and_mtime_is_revalidated(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            validator = make_validator(root)
            historical = root / "step-10.ckpt"
            high_water = root / "step-20.ckpt"
            write_checkpoint(historical, 10)
            write_checkpoint(high_water, 20)
            args = base_args(state, str(root / "*.ckpt"), validator, ["/bin/true"])
            args.insert(args.index("--"), "--dry-run")
            with mock.patch.dict(os.environ, {"SLURM_RESTART_COUNT": "0"}, clear=False):
                self.assertEqual(MODULE.main(args), 0)

            original = historical.stat()
            replacement = root / "replacement.ckpt"
            write_checkpoint(replacement, 30)
            self.assertEqual(replacement.stat().st_size, original.st_size)
            os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
            os.replace(replacement, historical)
            replaced = historical.stat()
            self.assertEqual(replaced.st_size, original.st_size)
            self.assertEqual(replaced.st_mtime_ns, original.st_mtime_ns)
            self.assertNotEqual(
                (replaced.st_dev, replaced.st_ino, replaced.st_ctime_ns),
                (original.st_dev, original.st_ino, original.st_ctime_ns),
            )

            with mock.patch.dict(os.environ, {"SLURM_RESTART_COUNT": "1"}, clear=False):
                self.assertEqual(MODULE.main(args), 0)
            final = json.loads(
                (state / "requeue" / "job-dry-run-restart-001" / "attempt.json").read_text()
            )
            self.assertEqual(final["checkpoint"]["global_step"], 30)

    def test_runner_ownership_requires_explicit_child_disable_confirmation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            validator = make_validator(root)
            args = base_args(
                state,
                str(root / "*.ckpt"),
                validator,
                ["/bin/true"],
                owner="runner",
            )
            args.remove("--confirm-child-requeue-disabled")
            args.insert(args.index("--"), "--dry-run")
            with self.assertRaisesRegex(SystemExit, "disable framework auto-requeue"):
                MODULE.main(args)

    def test_required_initial_checkpoint_is_refused(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            validator = make_validator(root)
            args = base_args(state, str(root / "*.ckpt"), validator, ["/bin/true"])
            args.insert(-2, "--require-initial-checkpoint")
            args.insert(-2, "--dry-run")
            with mock.patch.dict(os.environ, {"SLURM_RESTART_COUNT": "0"}, clear=False):
                with self.assertRaisesRegex(SystemExit, "requires an exact --initial-checkpoint"):
                    MODULE.main(args)

    def test_exact_initial_pins_first_attempt_then_live_wins_restart(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            staged = root / "staged"
            live = root / "live"
            staged.mkdir()
            live.mkdir()
            validator = make_validator(root)
            initial = staged / "step-40000.ckpt"
            newer_live = live / "step-60000.ckpt"
            write_checkpoint(initial, 40_000)
            write_checkpoint(newer_live, 60_000)
            args = base_args(
                state,
                str(live / "*.ckpt"),
                validator,
                ["/bin/true"],
                initial_checkpoint=initial,
            )
            args.insert(args.index("--"), "--require-initial-checkpoint")
            args.insert(args.index("--"), "--dry-run")

            with mock.patch.dict(os.environ, {"SLURM_RESTART_COUNT": "0"}, clear=False):
                self.assertEqual(MODULE.main(args), 0)
            first = json.loads(
                (state / "requeue" / "job-dry-run-restart-000" / "attempt.json").read_text()
            )
            self.assertEqual(first["checkpoint"]["global_step"], 40_000)
            self.assertEqual(first["checkpoint"]["path"], str(initial.resolve()))

            initial.unlink()
            with mock.patch.dict(os.environ, {"SLURM_RESTART_COUNT": "1"}, clear=False):
                self.assertEqual(MODULE.main(args), 0)
            restarted = json.loads(
                (state / "requeue" / "job-dry-run-restart-001" / "attempt.json").read_text()
            )
            self.assertEqual(restarted["checkpoint"]["global_step"], 60_000)
            self.assertEqual(restarted["checkpoint"]["path"], str(newer_live.resolve()))

    def test_explicit_missing_initial_checkpoint_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            validator = make_validator(root)
            args = base_args(
                state,
                str(root / "live" / "*.ckpt"),
                validator,
                ["/bin/true"],
                initial_checkpoint=root / "staged" / "missing.ckpt",
            )
            args.insert(args.index("--"), "--dry-run")
            with self.assertRaisesRegex(SystemExit, "initial checkpoint is unavailable"):
                MODULE.main(args)

    def test_restart_without_checkpoint_is_always_refused(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            validator = make_validator(root)
            args = base_args(state, str(root / "*.ckpt"), validator, ["/bin/true"])
            args.insert(-2, "--dry-run")
            with mock.patch.dict(os.environ, {"SLURM_RESTART_COUNT": "1"}, clear=False):
                with self.assertRaisesRegex(SystemExit, "required resume checkpoint"):
                    MODULE.main(args)

    def test_high_water_refuses_rollback(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            validator = make_validator(root)
            newest = root / "step-60000.ckpt"
            write_checkpoint(newest, 60_000)
            args = base_args(state, str(root / "*.ckpt"), validator, ["/bin/true"])
            args.insert(-2, "--dry-run")
            with mock.patch.dict(os.environ, {"SLURM_RESTART_COUNT": "0"}, clear=False):
                self.assertEqual(MODULE.main(args), 0)
            newest.unlink()
            write_checkpoint(root / "step-40000.ckpt", 40_000)
            with mock.patch.dict(os.environ, {"SLURM_RESTART_COUNT": "1"}, clear=False):
                with self.assertRaisesRegex(SystemExit, "rollback refused"):
                    MODULE.main(args)

    def test_high_water_refuses_newer_same_step_different_sha(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            validator = make_validator(root)
            checkpoint = root / "step-60000.ckpt"
            write_checkpoint(checkpoint, 60_000)
            args = base_args(state, str(root / "*.ckpt"), validator, ["/bin/true"])
            args.insert(-2, "--dry-run")
            with mock.patch.dict(os.environ, {"SLURM_RESTART_COUNT": "0"}, clear=False):
                self.assertEqual(MODULE.main(args), 0)
            time.sleep(0.01)
            checkpoint.unlink()
            checkpoint.write_text(
                json.dumps(
                    {
                        "global_step": 60_000,
                        "valid": True,
                        "run_id": "run",
                        "wandb_run_id": "wandb-run",
                        "different_payload": True,
                    }
                )
            )
            with mock.patch.dict(os.environ, {"SLURM_RESTART_COUNT": "1"}, clear=False):
                with self.assertRaisesRegex(SystemExit, "identity fork refused"):
                    MODULE.main(args)

    def test_state_directory_lock_refuses_duplicate_runner(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            state.mkdir(parents=True)
            validator = make_validator(root)
            write_checkpoint(root / "step-1.ckpt", 1)
            lock = (state / "runner.lock").open("a+")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                args = base_args(state, str(root / "*.ckpt"), validator, ["/bin/true"])
                args.insert(args.index("--"), "--dry-run")
                command = [
                    sys.executable,
                    str(MODULE_PATH),
                    *args,
                ]
                result = subprocess.run(command, text=True, capture_output=True, check=False)
            finally:
                lock.close()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("another requeue runner", result.stderr)

    def test_exports_checkpoint_path_sha_step_and_metadata(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            validator = make_validator(root)
            checkpoint = root / "step-12.ckpt"
            write_checkpoint(checkpoint, 12)
            output = root / "environment.json"
            child = root / "child.py"
            child.write_text(
                "import json,os\n"
                "keys=['ICE_RESUME_CHECKPOINT','ICE_RESUME_CHECKPOINT_SHA256',"
                "'ICE_RESUME_GLOBAL_STEP','ICE_RESUME_CHECKPOINT_METADATA_JSON',"
                "'ICE_REQUEUE_OWNER']\n"
                "open(os.environ['OUTPUT'],'w').write(json.dumps({k:os.environ[k] for k in keys}))\n"
            )
            env = os.environ.copy()
            env.update({"SLURM_JOB_ID": "991", "SLURM_RESTART_COUNT": "0", "OUTPUT": str(output)})
            command = [
                sys.executable,
                str(MODULE_PATH),
                *base_args(state, str(root / "*.ckpt"), validator, [sys.executable, str(child)]),
            ]
            result = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            exported = json.loads(output.read_text())
            self.assertEqual(exported["ICE_RESUME_CHECKPOINT"], str(checkpoint.resolve()))
            self.assertEqual(exported["ICE_RESUME_GLOBAL_STEP"], "12")
            self.assertEqual(
                exported["ICE_RESUME_CHECKPOINT_SHA256"],
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            )
            metadata = json.loads(exported["ICE_RESUME_CHECKPOINT_METADATA_JSON"])
            self.assertEqual(metadata["global_step"], 12)
            self.assertEqual(metadata["checkpoint_path"], str(checkpoint.resolve()))
            self.assertEqual(exported["ICE_REQUEUE_OWNER"], "child")

    def test_signal_baseline_waits_for_post_signal_checkpoint(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            staged = root / "staged"
            live = root / "live"
            staged.mkdir()
            live.mkdir()
            validator = make_validator(root)
            initial = staged / "step-1.ckpt"
            write_checkpoint(initial, 1)
            ready = root / "ready"
            child_pid = root / "child.pid"
            resume_seen = root / "resume-seen"
            child = root / "child.py"
            child.write_text(
                "import json,os,signal,time\n"
                "root=os.environ['ROOT']\n"
                "def write(step):\n"
                " p=os.path.join(root,f'step-{step}.ckpt'); t=p+'.tmp'\n"
                " open(t,'w').write(json.dumps({'global_step':step,'run_id':'run','wandb_run_id':'wandb-run'}))\n"
                " os.replace(t,p)\n"
                "def save(*_):\n"
                " time.sleep(0.6); write(3); raise SystemExit(0)\n"
                "signal.signal(signal.SIGUSR2,save)\n"
                "open(os.environ['CHILD_PID_FILE'],'w').write(str(os.getpid()))\n"
                "open(os.environ['RESUME_SEEN'],'w').write(os.environ['ICE_RESUME_CHECKPOINT'])\n"
                "write(2); open(os.environ['READY'],'w').write('ready')\n"
                "while True: time.sleep(0.05)\n"
            )
            scancel_log = root / "scancel.log"
            fake_scancel = make_signaling_scancel(root)
            scontrol_log = root / "scontrol.log"
            fake_scontrol = root / "scontrol"
            fake_scontrol.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" > \"$SCONTROL_LOG\"\n")
            fake_scontrol.chmod(0o700)
            env = os.environ.copy()
            env.update(
                {
                    "SLURM_JOB_ID": "12345",
                    "SLURM_RESTART_COUNT": "0",
                    "ROOT": str(live),
                    "READY": str(ready),
                    "CHILD_PID_FILE": str(child_pid),
                    "RESUME_SEEN": str(resume_seen),
                    "ICE_TEST_SCANCEL_LOG": str(scancel_log),
                    "SCONTROL_LOG": str(scontrol_log),
                }
            )
            args = base_args(
                state,
                str(live / "*.ckpt"),
                validator,
                [sys.executable, str(child)],
                owner="runner",
                initial_checkpoint=initial,
                scancel=fake_scancel,
            )
            args[args.index("--poll-seconds") : args.index("--poll-seconds") + 2] = [
                "--poll-seconds",
                "0.02",
            ]
            args[args.index("--") : args.index("--")] = [
                "--checkpoint-grace-seconds",
                "4",
                "--scontrol",
                str(fake_scontrol),
            ]
            command = [sys.executable, str(MODULE_PATH), *args]
            process = subprocess.Popen(
                command,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            attempt = state / "requeue" / "job-12345-restart-000" / "attempt.json"
            deadline = time.time() + 5
            while time.time() < deadline:
                if ready.exists() and attempt.exists() and json.loads(attempt.read_text()).get("status") == "RUNNING":
                    break
                time.sleep(0.02)
            else:
                process.kill()
                self.fail("runner/child did not become ready")
            process.send_signal(signal.SIGUSR1)
            time.sleep(0.2)
            self.assertFalse(scontrol_log.exists(), "runner requeued before post-signal save")
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, f"{stdout}\n{stderr}")
            self.assertEqual(resume_seen.read_text(), str(initial.resolve()))
            self.assertEqual(scancel_log.read_text().strip(), "--signal=USR2 12345")
            self.assertEqual(scontrol_log.read_text().strip(), "requeue 12345")
            final = json.loads(attempt.read_text())
            self.assertEqual(final["checkpoint"]["global_step"], 3)
            self.assertTrue(final["fresh_checkpoint_after_signal"])
            events = [
                json.loads(line)
                for line in (attempt.parent / "events.jsonl").read_text().splitlines()
            ]
            signal_event = next(event for event in events if event["event"] == "signal")
            self.assertEqual(signal_event["baseline"]["global_step"], 2)

    def test_runner_refuses_requeue_without_fresh_post_signal_checkpoint(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            staged = root / "staged"
            live = root / "live"
            staged.mkdir()
            live.mkdir()
            validator = make_validator(root)
            initial = staged / "step-1.ckpt"
            write_checkpoint(initial, 1)
            ready = root / "ready"
            child_pid = root / "child.pid"
            child = root / "child.py"
            child.write_text(
                "import os,signal,time\n"
                "signal.signal(signal.SIGUSR2, lambda *_: (_ for _ in ()).throw(SystemExit(0)))\n"
                "open(os.environ['CHILD_PID_FILE'],'w').write(str(os.getpid()))\n"
                "open(os.environ['READY'],'w').write('ready')\n"
                "while True: time.sleep(0.05)\n"
            )
            scancel_log = root / "scancel.log"
            fake_scancel = make_signaling_scancel(root)
            called = root / "scontrol-called"
            fake_scontrol = root / "scontrol"
            fake_scontrol.write_text("#!/bin/sh\ntouch \"$SCONTROL_CALLED\"\n")
            fake_scontrol.chmod(0o700)
            env = os.environ.copy()
            env.update(
                {
                    "SLURM_JOB_ID": "12348",
                    "SLURM_RESTART_COUNT": "0",
                    "READY": str(ready),
                    "CHILD_PID_FILE": str(child_pid),
                    "ICE_TEST_SCANCEL_LOG": str(scancel_log),
                    "SCONTROL_CALLED": str(called),
                }
            )
            args = base_args(
                state,
                str(live / "*.ckpt"),
                validator,
                [sys.executable, str(child)],
                owner="runner",
                initial_checkpoint=initial,
                scancel=fake_scancel,
            )
            args[args.index("--") : args.index("--")] = [
                "--checkpoint-grace-seconds",
                "1",
                "--scontrol",
                str(fake_scontrol),
            ]
            process = subprocess.Popen(
                [sys.executable, str(MODULE_PATH), *args],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            attempt = state / "requeue" / "job-12348-restart-000" / "attempt.json"
            deadline = time.time() + 5
            while time.time() < deadline:
                if ready.exists() and attempt.exists() and json.loads(attempt.read_text()).get("status") == "RUNNING":
                    break
                time.sleep(0.02)
            else:
                process.kill()
                self.fail("runner/child did not become ready")
            process.send_signal(signal.SIGUSR1)
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 74, f"{stdout}\n{stderr}")
            self.assertFalse(called.exists())
            self.assertEqual(scancel_log.read_text().strip(), "--signal=USR2 12348")
            final = json.loads(attempt.read_text())
            self.assertEqual(final["status"], "REQUEUE_REFUSED_NO_FRESH_CHECKPOINT")
            self.assertFalse(final["fresh_checkpoint_after_signal"])

    def test_child_requeue_owner_never_calls_scontrol(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            staged = root / "staged"
            live = root / "live"
            staged.mkdir()
            live.mkdir()
            validator = make_validator(root)
            initial = staged / "step-1.ckpt"
            write_checkpoint(initial, 1)
            child_pid = root / "child.pid"
            child = root / "child.py"
            child.write_text(
                "import json,os,signal,time\n"
                "def save(*_):\n"
                " p=os.path.join(os.environ['ROOT'],'step-2.ckpt'); t=p+'.tmp'\n"
                " open(t,'w').write(json.dumps({'global_step':2,'run_id':'run','wandb_run_id':'wandb-run'}))\n"
                " os.replace(t,p); raise SystemExit(0)\n"
                "signal.signal(signal.SIGUSR2,save)\n"
                "open(os.environ['CHILD_PID_FILE'],'w').write(str(os.getpid()))\n"
                "while True: time.sleep(0.05)\n"
            )
            scancel_log = root / "scancel.log"
            fake_scancel = make_signaling_scancel(root)
            called = root / "scontrol-called"
            fake_scontrol = root / "scontrol"
            fake_scontrol.write_text("#!/bin/sh\ntouch \"$SCONTROL_CALLED\"\n")
            fake_scontrol.chmod(0o700)
            env = os.environ.copy()
            env.update(
                {
                    "SLURM_JOB_ID": "12346",
                    "SLURM_RESTART_COUNT": "0",
                    "ROOT": str(live),
                    "CHILD_PID_FILE": str(child_pid),
                    "ICE_TEST_SCANCEL_LOG": str(scancel_log),
                    "SCONTROL_CALLED": str(called),
                }
            )
            args = base_args(
                state,
                str(live / "*.ckpt"),
                validator,
                [sys.executable, str(child)],
                owner="child",
                initial_checkpoint=initial,
                scancel=fake_scancel,
            )
            args[args.index("--") : args.index("--")] = [
                "--checkpoint-grace-seconds",
                "3",
                "--scontrol",
                str(fake_scontrol),
            ]
            process = subprocess.Popen(
                [sys.executable, str(MODULE_PATH), *args],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            attempt = state / "requeue" / "job-12346-restart-000" / "attempt.json"
            deadline = time.time() + 5
            while time.time() < deadline:
                if (
                    child_pid.exists()
                    and attempt.exists()
                    and json.loads(attempt.read_text()).get("status") == "RUNNING"
                ):
                    break
                time.sleep(0.02)
            else:
                process.kill()
                self.fail("runner did not become ready")
            process.send_signal(signal.SIGUSR1)
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, f"{stdout}\n{stderr}")
            self.assertFalse(called.exists())
            self.assertEqual(scancel_log.read_text().strip(), "--signal=USR2 12346")
            final = json.loads(attempt.read_text())
            self.assertEqual(final["status"], "COMPLETE")

    def test_duplicate_usr1_boundary_invokes_scancel_only_once(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            staged = root / "staged"
            live = root / "live"
            staged.mkdir()
            live.mkdir()
            validator = make_validator(root)
            initial = staged / "step-1.ckpt"
            write_checkpoint(initial, 1)
            ready = root / "ready"
            saved = root / "saved"
            child_pid = root / "child.pid"
            child = root / "child.py"
            child.write_text(
                "import json,os,signal,time\n"
                "def save(*_):\n"
                " p=os.path.join(os.environ['ROOT'],'step-2.ckpt'); t=p+'.tmp'\n"
                " open(t,'w').write(json.dumps({'global_step':2,'run_id':'run','wandb_run_id':'wandb-run'}))\n"
                " os.replace(t,p); open(os.environ['SAVED'],'w').write('saved')\n"
                "signal.signal(signal.SIGUSR2,save)\n"
                "open(os.environ['CHILD_PID_FILE'],'w').write(str(os.getpid()))\n"
                "open(os.environ['READY'],'w').write('ready')\n"
                "while True: time.sleep(0.05)\n"
            )
            scancel_log = root / "scancel.log"
            fake_scancel = root / "scancel"
            fake_scancel.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$*\" >> \"$ICE_TEST_SCANCEL_LOG\"\n"
                "kill -USR2 \"$(cat \"$CHILD_PID_FILE\")\"\n"
            )
            fake_scancel.chmod(0o700)
            env = os.environ.copy()
            env.update(
                {
                    "SLURM_JOB_ID": "12350",
                    "SLURM_RESTART_COUNT": "0",
                    "ROOT": str(live),
                    "READY": str(ready),
                    "SAVED": str(saved),
                    "CHILD_PID_FILE": str(child_pid),
                    "ICE_TEST_SCANCEL_LOG": str(scancel_log),
                }
            )
            args = base_args(
                state,
                str(live / "*.ckpt"),
                validator,
                [sys.executable, str(child)],
                owner="child",
                initial_checkpoint=initial,
                scancel=fake_scancel,
            )
            process = subprocess.Popen(
                [sys.executable, str(MODULE_PATH), *args],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            attempt = state / "requeue" / "job-12350-restart-000" / "attempt.json"
            deadline = time.time() + 5
            while time.time() < deadline:
                if ready.exists() and attempt.exists():
                    break
                time.sleep(0.02)
            else:
                process.kill()
                self.fail("runner/child did not become ready")
            process.send_signal(signal.SIGUSR1)
            deadline = time.time() + 5
            while time.time() < deadline:
                if (
                    saved.exists()
                    and json.loads(attempt.read_text()).get("status")
                    == "CHILD_REQUEUE_OWNER_CHECKPOINT_READY"
                ):
                    break
                time.sleep(0.02)
            else:
                process.kill()
                self.fail("first checkpoint boundary did not complete")
            process.send_signal(signal.SIGUSR1)
            time.sleep(0.2)
            self.assertEqual(scancel_log.read_text().splitlines(), ["--signal=USR2 12350"])
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 128 + signal.SIGTERM, f"{stdout}\n{stderr}")

    def test_scancel_failure_is_persisted_and_never_requeues(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            staged = root / "staged"
            live = root / "live"
            staged.mkdir()
            live.mkdir()
            validator = make_validator(root)
            initial = staged / "step-1.ckpt"
            write_checkpoint(initial, 1)
            ready = root / "ready"
            child_pid = root / "child.pid"
            child = root / "child.py"
            child.write_text(
                "import os,time\n"
                "open(os.environ['CHILD_PID_FILE'],'w').write(str(os.getpid()))\n"
                "open(os.environ['READY'],'w').write('ready')\n"
                "while True: time.sleep(0.05)\n"
            )
            scancel_log = root / "scancel.log"
            fake_scancel = root / "scancel"
            fake_scancel.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$*\" > \"$ICE_TEST_SCANCEL_LOG\"\n"
                "kill -TERM \"$(cat \"$CHILD_PID_FILE\")\"\n"
                "echo signal-refused >&2\n"
                "exit 9\n"
            )
            fake_scancel.chmod(0o700)
            scontrol_called = root / "scontrol-called"
            fake_scontrol = root / "scontrol"
            fake_scontrol.write_text("#!/bin/sh\ntouch \"$SCONTROL_CALLED\"\n")
            fake_scontrol.chmod(0o700)
            env = os.environ.copy()
            env.update(
                {
                    "SLURM_JOB_ID": "12349",
                    "SLURM_RESTART_COUNT": "0",
                    "READY": str(ready),
                    "CHILD_PID_FILE": str(child_pid),
                    "ICE_TEST_SCANCEL_LOG": str(scancel_log),
                    "SCONTROL_CALLED": str(scontrol_called),
                }
            )
            args = base_args(
                state,
                str(live / "*.ckpt"),
                validator,
                [sys.executable, str(child)],
                owner="runner",
                initial_checkpoint=initial,
                scancel=fake_scancel,
            )
            args[args.index("--") : args.index("--")] = ["--scontrol", str(fake_scontrol)]
            process = subprocess.Popen(
                [sys.executable, str(MODULE_PATH), *args],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            attempt = state / "requeue" / "job-12349-restart-000" / "attempt.json"
            deadline = time.time() + 5
            while time.time() < deadline:
                if ready.exists() and attempt.exists():
                    break
                time.sleep(0.02)
            else:
                process.kill()
                self.fail("runner/child did not become ready")
            process.send_signal(signal.SIGUSR1)
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 9, f"{stdout}\n{stderr}")
            self.assertEqual(scancel_log.read_text().strip(), "--signal=USR2 12349")
            self.assertFalse(scontrol_called.exists())
            final = json.loads(attempt.read_text())
            self.assertEqual(final["status"], "CHECKPOINT_SIGNAL_FORWARD_FAILED")
            self.assertEqual(final["checkpoint_signal_returncode"], 9)
            self.assertEqual(
                final["checkpoint_signal_argv"],
                [str(fake_scancel.resolve()), "--signal=USR2", "12349"],
            )
            self.assertIn("signal-refused", final["checkpoint_signal_stderr"])

    def test_scontrol_failure_is_persisted(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            staged = root / "staged"
            live = root / "live"
            staged.mkdir()
            live.mkdir()
            validator = make_validator(root)
            initial = staged / "step-1.ckpt"
            write_checkpoint(initial, 1)
            child_pid = root / "child.pid"
            child = root / "child.py"
            child.write_text(
                "import json,os,signal,time\n"
                "def save(*_):\n"
                " p=os.path.join(os.environ['ROOT'],'step-2.ckpt'); t=p+'.tmp'\n"
                " open(t,'w').write(json.dumps({'global_step':2,'run_id':'run','wandb_run_id':'wandb-run'}))\n"
                " os.replace(t,p); raise SystemExit(0)\n"
                "signal.signal(signal.SIGUSR2,save)\n"
                "open(os.environ['CHILD_PID_FILE'],'w').write(str(os.getpid()))\n"
                "while True: time.sleep(0.05)\n"
            )
            scancel_log = root / "scancel.log"
            fake_scancel = make_signaling_scancel(root)
            fake_scontrol = root / "scontrol"
            fake_scontrol.write_text("#!/bin/sh\necho refused >&2\nexit 7\n")
            fake_scontrol.chmod(0o700)
            env = os.environ.copy()
            env.update(
                {
                    "SLURM_JOB_ID": "12347",
                    "SLURM_RESTART_COUNT": "0",
                    "ROOT": str(live),
                    "CHILD_PID_FILE": str(child_pid),
                    "ICE_TEST_SCANCEL_LOG": str(scancel_log),
                }
            )
            args = base_args(
                state,
                str(live / "*.ckpt"),
                validator,
                [sys.executable, str(child)],
                owner="runner",
                initial_checkpoint=initial,
                scancel=fake_scancel,
            )
            args[args.index("--") : args.index("--")] = [
                "--checkpoint-grace-seconds",
                "3",
                "--scontrol",
                str(fake_scontrol),
            ]
            process = subprocess.Popen(
                [sys.executable, str(MODULE_PATH), *args],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            attempt = state / "requeue" / "job-12347-restart-000" / "attempt.json"
            deadline = time.time() + 5
            while time.time() < deadline:
                if (
                    child_pid.exists()
                    and attempt.exists()
                    and json.loads(attempt.read_text()).get("status") == "RUNNING"
                ):
                    break
                time.sleep(0.02)
            else:
                process.kill()
                self.fail("runner did not become ready")
            process.send_signal(signal.SIGUSR1)
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 7, f"{stdout}\n{stderr}")
            final = json.loads(attempt.read_text())
            self.assertEqual(final["status"], "REQUEUE_FAILED")
            self.assertEqual(final["scontrol_returncode"], 7)
            self.assertIn("refused", final["scontrol_stderr"])

    def test_completion_sentinel_contains_terminal_checkpoint(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state" / "run"
            validator = make_validator(root)
            write_checkpoint(root / "step-5.ckpt", 5)
            sentinel = root / "run" / "COMPLETE.json"
            child = root / "child.py"
            child.write_text(
                "import json,os\n"
                "p=os.path.join(os.environ['ROOT'],'step-6.ckpt'); t=p+'.tmp'\n"
                "open(t,'w').write(json.dumps({'global_step':6,'run_id':'run','wandb_run_id':'wandb-run'}))\n"
                "os.replace(t,p)\n"
            )
            env = os.environ.copy()
            env.update({"SLURM_JOB_ID": "992", "SLURM_RESTART_COUNT": "0", "ROOT": str(root)})
            args = base_args(state, str(root / "*.ckpt"), validator, [sys.executable, str(child)])
            args[args.index("--") : args.index("--")] = ["--completion-sentinel", str(sentinel)]
            result = subprocess.run(
                [sys.executable, str(MODULE_PATH), *args],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(sentinel.read_text())
            self.assertEqual(payload["status"], "COMPLETE")
            self.assertEqual(payload["checkpoint"]["global_step"], 6)


if __name__ == "__main__":
    unittest.main()
