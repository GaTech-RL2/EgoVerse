"""Focused standard-library operational supervision regressions."""

import argparse
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.fold_clothes_moving_opscale.operations import supervise as s


class SupervisionTests(unittest.TestCase):
    def test_partial_array_never_complete_and_failed_task_detected(self):
        rows = s.parse_slurm(
            "50_0|COMPLETED|0:0\n50_1|FAILED|2:0\n50_1.batch|FAILED|2:0\n",
            "50_2|RUNNING\n",
        )
        result = s.job_summary("50", 4, rows)
        self.assertEqual(result["state"], "FAILED")
        self.assertEqual(set(result["failures"]), {"50_1"})
        self.assertEqual(
            result["counts"], {"COMPLETED": 1, "FAILED": 1, "RUNNING": 1, "UNKNOWN": 1}
        )
        self.assertNotIn("50_1.batch", rows)
        self.assertEqual(
            s.job_summary("50", 4, {"50_0": rows["50_0"]})["state"], "UNKNOWN"
        )

    def test_terminal_nonzero_and_complete_array(self):
        rows = s.parse_slurm(
            "50_0|COMPLETED|0:0\n50_1|COMPLETED|0:0\n51|COMPLETED|3:0\n",
            "50_1|COMPLETED\n",
        )
        self.assertEqual(s.job_summary("50", 2, rows)["state"], "COMPLETED")
        self.assertEqual(s.job_summary("51", 0, rows)["state"], "FAILED")

    def test_exact_process_identity_and_pid_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            base = proc / "123"
            base.mkdir()
            (base / "cmdline").write_bytes(
                b"python\0/repo/pipeline.py\0--run-root\0/runs\0--commit\0abc\0"
            )
            (base / "stat").write_text(
                "123 (python worker) " + " ".join(["S"] + ["0"] * 18 + ["456"])
            )
            result = s.process_state(
                123,
                Path("/repo/pipeline.py"),
                Path("/runs"),
                commit="abc",
                proc_root=proc,
            )
            self.assertEqual(result["state"], "RUNNING")
            self.assertEqual(result["start_ticks"], 456)
            self.assertEqual(
                s.process_state(
                    123,
                    Path("/repo/pipeline.py"),
                    Path("/runs"),
                    commit="abc",
                    expected_ticks=455,
                    proc_root=proc,
                )["state"],
                "IDENTITY_MISMATCH",
            )
            self.assertEqual(
                s.process_state(123, Path("/wrong.py"), Path("/runs"), proc_root=proc)[
                    "state"
                ],
                "IDENTITY_MISMATCH",
            )
            self.assertNotIn("argv", result)

    def test_heartbeat_and_change_only_alerts(self):
        self.assertEqual(s.heartbeat_age({"heartbeat_epoch": 100}, 1000), 900)
        self.assertIsNone(s.heartbeat_age({}, 1000))
        alert = {"a": {"kind": "heartbeat_stale", "severity": "error"}}
        self.assertEqual(s.alert_changes(alert, alert), [])
        self.assertEqual(s.alert_changes({}, alert)[0]["event"], "raised")
        self.assertEqual(s.alert_changes(alert, {})[0]["event"], "resolved")
        before = {
            "state": "SUPERVISING",
            "phase": "VERIFYING_FINAL_DATA",
            "processes": {"pipeline": {"state": "RUNNING", "heartbeat_age_seconds": 2}},
            "data_progress": {"verified_episodes": 100},
        }
        after = {
            **before,
            "updated_at": "later",
            "data_progress": {"verified_episodes": 200},
            "processes": {"pipeline": {"state": "RUNNING", "heartbeat_age_seconds": 4}},
        }
        self.assertEqual(s.transition_view(before), s.transition_view(after))

    def test_bounded_log_does_not_expose_raw_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.out"
            path.write_text(
                "private-secret-must-not-escape\n"
                + json.dumps(
                    {
                        "checked": 100,
                        "total": 12669,
                        "failed": 2,
                        "signed_url": "private-secret-must-not-escape",
                    }
                )
                + "\n"
            )
            result = s.tail_progress(path, 1000)
            self.assertEqual(result["progress"]["failed"], 2)
            self.assertNotIn("private-secret", json.dumps(result))

    def test_superseded_stage_and_current_data_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pipeline_jobs").mkdir()
            s.save(
                root / "pipeline.json",
                {
                    "actions": {"final-data-check": {"job_id": "60", "array_count": 0}},
                    "accepted_staging": {
                        "job_id": "50",
                        "transfer_reached_verification": True,
                    },
                },
            )
            s.save(
                root / "pipeline_status.json",
                {
                    "state": "ACTIVE",
                    "phase": "VERIFYING_FINAL_DATA",
                    "pid": 1,
                    "heartbeat_epoch": 1000,
                },
            )
            s.save(
                root / "status.json",
                {
                    "state": "PREFLIGHT_ACTIVE",
                    "monitor_pid": 2,
                    "heartbeat_epoch": 1000,
                    "staging": {"state": "FAILED", "exit_code": "2:0"},
                    "runs": {},
                },
            )
            args = argparse.Namespace(
                run_root=root,
                repo=Path("/repo"),
                commit="a" * 40,
                pipeline_pid=1,
                monitor_pid=2,
                progress_stale=7200,
            )
            (root / "pipeline_jobs/final-data-check-59_4294967294.out").write_text(
                '{"checked":100,"failed":99,"total":12669}\n'
            )
            healthy = {
                "state": "RUNNING",
                "alive": True,
                "identity_match": True,
                "start_ticks": 123,
            }
            with (
                patch.object(s, "process_state", return_value=healthy),
                patch.object(
                    s, "run_command", side_effect=["60|RUNNING|0:0\n", "60|RUNNING\n"]
                ),
            ):
                result = s.collect(args, {}, current=1000)
            self.assertEqual(result["alerts"], {})
            self.assertEqual(
                result["superseded_staging"]["state"], "SUPERSEDED_BY_FINAL_DATA_CHECK"
            )
            (root / "pipeline_jobs/final-data-check-60_4294967294.out").write_text(
                '{"checked":100,"failed":2,"total":12669}\n'
            )
            with (
                patch.object(s, "process_state", return_value=healthy),
                patch.object(
                    s, "run_command", side_effect=["60|RUNNING|0:0\n", "60|RUNNING\n"]
                ),
            ):
                result = s.collect(args, {}, current=2001)
            self.assertIn("data:verification", result["alerts"])
            self.assertIn("heartbeat:pipeline", result["alerts"])
            self.assertIn("heartbeat:monitor", result["alerts"])
            self.assertNotIn("50", json.dumps(result["alerts"]))

    def test_preflight_complete_is_healthy_expected_exit_not_training_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            s.save(root / "pipeline.json", {"actions": {}})
            s.save(
                root / "pipeline_status.json",
                {
                    "state": "PREFLIGHT_COMPLETE",
                    "phase": "PREFLIGHT_COMPLETE",
                    "pid": 1,
                    "heartbeat_epoch": 1,
                },
            )
            s.save(
                root / "status.json",
                {
                    "state": "PREFLIGHT_COMPLETE",
                    "monitor_pid": 2,
                    "heartbeat_epoch": 1,
                    "runs": {},
                },
            )
            args = argparse.Namespace(
                run_root=root,
                repo=Path("/repo"),
                commit="a" * 40,
                pipeline_pid=1,
                monitor_pid=2,
                progress_stale=7200,
            )
            with (
                patch.object(
                    s,
                    "process_state",
                    return_value={
                        "state": "EXITED",
                        "alive": False,
                        "identity_match": True,
                        "start_ticks": 123,
                    },
                ),
                patch.object(s, "run_command", return_value=""),
            ):
                result = s.collect(args, {}, current=5000)
            self.assertEqual(result["state"], "PREFLIGHT_COMPLETE")
            self.assertEqual(result["alerts"], {})
            self.assertTrue(
                all(p["expected_exit"] for p in result["processes"].values())
            )

    def test_healthy_once_creates_empty_handoff_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = [
                "supervise",
                "--run-root",
                str(root),
                "--repo",
                "/repo",
                "--commit",
                "a" * 40,
                "--pipeline-pid",
                "1",
                "--monitor-pid",
                "2",
                "--once",
            ]
            value = {
                "state": "SUPERVISING",
                "phase": "VERIFYING_FINAL_DATA",
                "alerts": {},
            }
            with (
                patch("sys.argv", argv),
                patch.object(s, "collect", return_value=value),
                patch.object(s, "process_state", return_value={}),
                patch("builtins.print"),
            ):
                s.main()
            self.assertEqual((root / "operations/alerts.jsonl").read_text(), "")
            self.assertEqual(
                json.loads((root / "operations/active_alerts.json").read_text()), {}
            )
            self.assertTrue((root / "operations/supervision.json").is_file())
            self.assertEqual(
                len((root / "operations/events.jsonl").read_text().splitlines()), 1
            )

    def test_supervisor_exception_is_observer_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = [
                "supervise",
                "--run-root",
                str(root),
                "--repo",
                "/repo",
                "--commit",
                "a" * 40,
                "--pipeline-pid",
                "1",
                "--monitor-pid",
                "2",
                "--once",
            ]
            with (
                patch("sys.argv", argv),
                patch.object(s, "collect", side_effect=RuntimeError("private details")),
                patch.object(s, "process_state", return_value={}),
                patch("builtins.print"),
            ):
                s.main()
            alert = json.loads((root / "operations/active_alerts.json").read_text())[
                "supervisor"
            ]
            self.assertIs(alert["automatic_repair"], False)
            self.assertNotIn("private details", json.dumps(alert))


class RecoveryHealthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ops = self.root / "operations"
        self.ops.mkdir()
        self.args = argparse.Namespace(
            run_root=self.root,
            repo=Path("/repo"),
            commit="a" * 40,
            pipeline_pid=1,
            monitor_pid=2,
            progress_stale=7200,
        )
        self.identity = {
            "pid": 123,
            "start_ticks": 456,
            "pgid": 123,
            "host": socket.gethostname(),
        }
        self.live = patch.object(
            s,
            "operation_process",
            return_value={"state": "RUNNING", "alive": True, "pid": 123},
        )
        self.live.start()
        self.addCleanup(self.live.stop)

    def setup_services(self, dispatcher=True):
        s.save(
            self.ops / "recovery_config.json",
            {"run_root": str(self.root), "repo": "/repo", "commit": "a" * 40},
        )
        s.save(
            self.ops / "recovery_watch_status.json",
            {"identity": self.identity, "state": "RUNNING", "heartbeat_epoch": 1000},
        )
        s.save(self.ops / "dispatcher_control.json", {"enabled": dispatcher})
        s.save(
            self.ops / "dispatcher_status.json",
            {"identity": self.identity, "state": "IDLE", "heartbeat_epoch": 1000},
        )

    def maintenance(self, deadline=2000):
        plan = {
            "transaction_id": "migration-1",
            "run_root": str(self.root),
            "old_repo": "/repo",
            "old_commit": "a" * 40,
            "new_repo": "/repo-new",
            "new_commit": "b" * 40,
        }
        path = self.ops / "plan.json"
        s.save(path, plan)
        s.save(
            self.ops / "promotion_maintenance.json",
            {
                **plan,
                "active": True,
                "plan_path": str(path),
                "started_at": s.dt.datetime.fromtimestamp(
                    900, s.dt.timezone.utc
                ).isoformat(),
                "deadline_epoch": deadline,
                "owner_identity": self.identity,
            },
        )

    def test_not_initialized_or_activated_has_no_service_alerts(self):
        self.assertEqual(s.recovery_health(self.args, 1000)["alerts"], {})
        s.save(self.ops / "dispatcher_control.json", {"enabled": False})
        self.assertEqual(s.recovery_health(self.args, 1000)["alerts"], {})

    def test_healthy_idle_controller_and_dispatcher_are_quiet(self):
        self.setup_services()
        self.assertEqual(s.recovery_health(self.args, 1000)["alerts"], {})
        self.assertEqual(s.recovery_health(self.args, 1100)["alerts"], {})

    def test_stale_and_blocked_services_alert_without_automatic_repair(self):
        self.setup_services()
        s.save(
            self.ops / "dispatcher_status.json",
            {"identity": self.identity, "state": "BLOCKED", "heartbeat_epoch": 1000},
        )
        value = s.recovery_health(self.args, 1700)
        self.assertIn("operations:dispatcher:blocked", value["alerts"])
        self.assertIn("operations:dispatcher:heartbeat", value["alerts"])
        self.assertIn("operations:recovery:heartbeat", value["alerts"])
        self.assertTrue(
            all(a["automatic_repair"] is False for a in value["alerts"].values())
        )

    def test_dead_service_identity_and_active_manual_incident_are_visible(self):
        self.setup_services()
        s.save(
            self.ops / "recovery_state.json",
            {
                "incidents": {
                    "problem": {"state": "NEEDS_REVIEW", "active": True},
                    "old": {"state": "NEEDS_REVIEW", "active": False},
                    "expired": {"state": "LEASE_EXPIRED", "active": True},
                }
            },
        )
        with patch.object(
            s,
            "operation_process",
            return_value={"state": "IDENTITY_MISMATCH", "alive": True},
        ):
            alerts = s.recovery_health(self.args, 1000)["alerts"]
        self.assertIn("operations:dispatcher:process", alerts)
        self.assertIn("operations:incident:problem", alerts)
        self.assertIn("operations:incident:expired", alerts)
        self.assertNotIn("operations:incident:old", alerts)
        self.assertTrue(all(a["automatic_repair"] is False for a in alerts.values()))

    def test_valid_maintenance_suppresses_expected_worker_changes_only(self):
        self.setup_services()
        self.maintenance()
        self.assertTrue(s.recovery_health(self.args, 1000)["maintenance"])
        s.save(
            self.root / "pipeline.json",
            {"actions": {"final-data-check": {"job_id": "60", "array_count": 0}}},
        )
        (self.root / "pipeline_jobs").mkdir()
        with (
            patch.object(
                s, "process_state", return_value={"state": "MISSING", "alive": False}
            ),
            patch.object(s, "run_command", side_effect=["60|FAILED|2:0\n", ""]),
        ):
            result = s.collect(self.args, {}, current=1000)
        self.assertEqual(result["state"], "PROMOTION_MAINTENANCE")
        self.assertFalse(
            any(k.startswith(("process:", "heartbeat:")) for k in result["alerts"])
        )
        self.assertIn("job:prepare:final-data-check:60", result["alerts"])
        self.assertTrue(
            all(a["automatic_repair"] is False for a in result["alerts"].values())
        )

    def test_expired_or_dead_owner_maintenance_never_hides_health(self):
        self.setup_services()
        self.maintenance(deadline=950)
        result = s.recovery_health(self.args, 1000)
        self.assertFalse(result["maintenance"])
        self.assertEqual(
            result["alerts"]["operations:promotion"]["kind"],
            "promotion_maintenance_expired",
        )
        self.maintenance()
        with patch.object(
            s, "operation_process", return_value={"state": "MISSING", "alive": False}
        ):
            result = s.recovery_health(self.args, 1000)
        self.assertFalse(result["maintenance"])
        self.assertEqual(
            result["alerts"]["operations:promotion"]["kind"],
            "promotion_maintenance_owner_unverified",
        )

    def test_unbounded_or_wrong_transaction_maintenance_is_rejected(self):
        self.setup_services()
        self.maintenance(deadline=999999)
        self.assertFalse(s.recovery_health(self.args, 1000)["maintenance"])
        self.maintenance()
        plan = s.load(self.ops / "plan.json")
        plan["transaction_id"] = "other"
        s.save(self.ops / "plan.json", plan)
        self.assertFalse(s.recovery_health(self.args, 1000)["maintenance"])

    def test_completed_controller_exit_allowed_only_after_verified_sweep(self):
        self.setup_services(dispatcher=False)
        s.save(
            self.ops / "recovery_watch_status.json",
            {"identity": self.identity, "state": "COMPLETE", "heartbeat_epoch": 1},
        )
        with patch.object(
            s, "operation_process", return_value={"state": "MISSING", "alive": False}
        ):
            self.assertEqual(
                s.recovery_health(self.args, 1000, expected_complete=True)["alerts"], {}
            )
            self.assertIn(
                "operations:recovery:completion",
                s.recovery_health(self.args, 1000)["alerts"],
            )

    def test_operation_identity_pins_host_start_group_and_command(self):
        self.live.stop()
        proc = self.root / "proc"
        base = proc / "123"
        base.mkdir(parents=True)
        fields = ["S"] + ["0"] * 18 + ["456"]
        fields[2] = "123"
        (base / "stat").write_text("123 (python worker) " + " ".join(fields))
        script = self.ops / "agent_dispatcher.py"
        (base / "cmdline").write_bytes(
            (
                "python\0"
                + str(script)
                + "\0serve\0--run-root\0"
                + str(self.root)
                + "\0"
            ).encode()
        )

        def check(saved):
            return s.operation_process(
                saved, script=script, root=self.root, mode="serve", proc_root=proc
            )["state"]

        self.assertEqual(check(self.identity), "RUNNING")
        self.assertEqual(
            check({**self.identity, "start_ticks": 457}), "IDENTITY_MISMATCH"
        )
        self.assertEqual(check({**self.identity, "pgid": 124}), "IDENTITY_MISMATCH")
        self.assertEqual(
            check({**self.identity, "host": "another-host"}),
            "REMOTE_IDENTITY_UNVERIFIABLE",
        )
        (base / "cmdline").write_bytes(
            (
                "python\0"
                + str(script)
                + "\0run-attempt\0--run-root\0"
                + str(self.root)
                + "\0"
            ).encode()
        )
        self.assertEqual(check(self.identity), "IDENTITY_MISMATCH")


if __name__ == "__main__":
    unittest.main()
