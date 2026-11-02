import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from experiments.fold_clothes_moving_opscale.operations import recovery_controller as r
from experiments.fold_clothes_moving_opscale.operations import supervise as s


class RecoveryTests(unittest.TestCase):
    def incident(self):
        state = r.empty_state()
        r.reconcile_incidents(
            state,
            {
                "data:verification": {
                    "kind": "data_verification_failures",
                    "severity": "error",
                }
            },
            100,
        )
        return state, next(iter(state["incidents"]))

    def test_repeated_alert_one_incident_and_recurrence_new_occurrence(self):
        state, key = self.incident()
        self.assertEqual(
            r.reconcile_incidents(
                state, {"data:verification": state["incidents"][key]["alert"]}, 200
            ),
            [],
        )
        self.assertEqual(len(state["incidents"]), 1)
        r.reconcile_incidents(state, {}, 300)
        self.assertEqual(state["incidents"][key]["state"], "RESOLVED")
        r.reconcile_incidents(
            state,
            {
                "data:verification": {
                    "kind": "data_verification_failures",
                    "severity": "error",
                }
            },
            400,
        )
        self.assertEqual(len(state["incidents"]), 2)

    def test_claim_exclusive_lease_expiry_never_duplicates(self):
        state, key = self.incident()
        r.claim_incident(state, key, "agent1", 60, 110)
        with self.assertRaises(RuntimeError):
            r.claim_incident(state, key, "agent2", 60, 120)
        r.reconcile_incidents(
            state, {"data:verification": state["incidents"][key]["alert"]}, 171
        )
        self.assertEqual(state["incidents"][key]["state"], "LEASE_EXPIRED")
        with self.assertRaises(RuntimeError):
            r.claim_incident(state, key, "agent2", 60, 200)
        with self.assertRaises(RuntimeError):
            r.finish_incident(state, key, "agent2", "fixed", "receipt", 201)

    def test_fixed_requires_actual_observation_and_cooldown(self):
        state, key = self.incident()
        r.claim_incident(state, key, "agent1", 60, 110)
        item = r.finish_incident(state, key, "agent1", "fixed", "receipt", 120)
        self.assertEqual(item["state"], "AWAITING_VERIFICATION")
        alerts = {"data:verification": item["alert"]}
        r.reconcile_incidents(state, alerts, 121)
        self.assertEqual(item["state"], "AWAITING_VERIFICATION")
        r.reconcile_incidents(state, alerts, 1920)
        self.assertEqual(item["state"], "QUEUED")
        r.reconcile_incidents(state, {}, 1921)
        self.assertEqual(item["state"], "RESOLVED")

    def test_attempt_limit_and_failed_cooldown(self):
        state, key = self.incident()
        for attempt in range(3):
            now = 100 + attempt * 2000
            r.claim_incident(state, key, "agent", 60, now)
            item = r.finish_incident(state, key, "agent", "failed", "receipt", now + 1)
            if attempt < 2:
                with self.assertRaises(RuntimeError):
                    r.claim_incident(state, key, "agent", 60, now + 2)
        self.assertEqual(item["state"], "NEEDS_REVIEW")
        with self.assertRaises(RuntimeError):
            r.claim_incident(state, key, "agent", 60, 10000)

    def test_dead_worker_only_bounded_restart(self):
        dead = {"state": "MISSING", "alive": False}
        report = {"expected_exit": False}
        observed = {"consecutive_dead": 2}
        self.assertEqual(r.restart_decision(dead, report, [], observed, 100), "READY")
        self.assertTrue(
            r.restart_decision(
                {"state": "RUNNING", "alive": True}, report, [], observed, 100
            ).startswith("DENY")
        )
        self.assertEqual(
            r.restart_decision(dead, {"expected_exit": True}, [], observed, 100),
            "DENY_EXPECTED_EXIT",
        )
        self.assertEqual(
            r.restart_decision(dead, report, [], observed, 100, True),
            "DENY_AMBIGUOUS_SUBMISSION",
        )
        self.assertEqual(
            r.restart_decision(dead, report, [], {"consecutive_dead": 1}, 100),
            "WAIT_SECOND_DEAD_OBSERVATION",
        )
        self.assertEqual(
            r.restart_decision(
                dead, report, [{"created_at": 0, "state": "STARTING"}], observed, 2000
            ),
            "DENY_UNCONFIRMED_RESTART",
        )
        self.assertEqual(
            r.restart_decision(
                dead, report, [{"created_at": 90, "state": "VERIFIED"}], observed, 100
            ),
            "WAIT_COOLDOWN",
        )
        self.assertEqual(
            r.restart_decision(
                dead,
                report,
                [{"created_at": i, "state": "VERIFIED"} for i in range(3)],
                observed,
                4000,
            ),
            "DENY_RESTART_LIMIT",
        )

    def test_dynamic_target_requires_matching_restart_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            op = root / "operations"
            op.mkdir()
            defaults = {
                "pipeline": {"pid": 1, "start_ticks": 10},
                "monitor": {"pid": 2, "start_ticks": 20},
            }
            changed = {"pid": 3, "start_ticks": 30, "restart_id": "known"}
            s.save(
                op / "process_targets.json",
                {
                    "commit": "abc",
                    "run_root": str(root),
                    "targets": {**defaults, "monitor": changed},
                },
            )
            found, bad = s.dynamic_targets(root, "abc", defaults)
            self.assertEqual(found, defaults)
            self.assertEqual(bad, ["monitor"])
            s.save(
                op / "recovery_state.json",
                {
                    "restart_attempts": {
                        "monitor": [
                            {
                                "id": "known",
                                "pid": 3,
                                "start_ticks": 30,
                                "state": "VERIFIED",
                            }
                        ]
                    }
                },
            )
            found, bad = s.dynamic_targets(root, "abc", defaults)
            self.assertEqual(found["monitor"], changed)
            self.assertEqual(bad, [])

    def test_verified_restart_updates_audit_dynamic_and_public_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            op = root / "operations"
            op.mkdir()
            config = {
                "run_root": str(root),
                "repo": "/repo",
                "commit": "a" * 40,
                "python": "/python",
                "gpu": "h200",
                "initial_targets": {
                    "pipeline": {"pid": 1, "start_ticks": 10},
                    "monitor": {"pid": 2, "start_ticks": 20},
                },
            }
            s.save(root / "pipeline.json", {"commit": "a" * 40, "actions": {}})
            s.save(
                root / "pipeline_status.json", {"commit": "a" * 40, "state": "ACTIVE"}
            )
            state = r.empty_state()
            state["observations"]["monitor"] = {"consecutive_dead": 1}
            dead = {"state": "MISSING", "alive": False}
            healthy = {"state": "RUNNING", "alive": True, "start_ticks": 30}
            child = Mock(pid=3)
            child.poll.return_value = None
            with (
                patch.object(
                    r, "process_state", side_effect=[healthy, dead, dead, healthy]
                ),
                patch.object(r, "source_guard"),
                patch.object(r.time, "sleep"),
                patch.object(r.subprocess, "Popen", return_value=child) as spawn,
            ):
                result = r.inspect_local(config, op, state, 100, True)
            self.assertEqual(result["monitor"]["restart_decision"], "RESTARTED")
            self.assertEqual((root / "monitor.pid").read_text().strip(), "3")
            targets, bad = s.dynamic_targets(
                root, config["commit"], config["initial_targets"]
            )
            self.assertEqual(targets["monitor"]["pid"], 3)
            self.assertEqual(bad, [])
            spawn.assert_called_once()

    def test_read_only_reconcile_never_spawns_even_dead(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            op = root / "operations"
            op.mkdir()
            config = {
                "run_root": str(root),
                "repo": "/repo",
                "commit": "a" * 40,
                "python": "/python",
                "gpu": "h200",
                "initial_targets": {
                    "pipeline": {"pid": 1, "start_ticks": 10},
                    "monitor": {"pid": 2, "start_ticks": 20},
                },
            }
            s.save(op / "recovery_config.json", config)
            s.save(
                op / "supervision.json",
                {
                    "heartbeat_epoch": 100,
                    "state": "ATTENTION_REQUIRED",
                    "alerts": {
                        "data:verification": {
                            "kind": "data_verification_failures",
                            "severity": "error",
                        }
                    },
                },
            )
            s.save(root / "pipeline.json", {"commit": "a" * 40, "actions": {}})
            s.save(
                root / "pipeline_status.json", {"commit": "a" * 40, "state": "ACTIVE"}
            )
            with (
                patch.object(
                    r,
                    "process_state",
                    return_value={"state": "MISSING", "alive": False},
                ),
                patch.object(r.time, "time", return_value=100),
                patch.object(r.subprocess, "Popen") as spawn,
            ):
                r.reconcile(root)
                result = r.reconcile(root)
                spawn.assert_not_called()
            self.assertEqual(
                result["local_processes"]["pipeline"]["restart_decision"], "READY"
            )
            state = s.load(op / "recovery_state.json")
            self.assertEqual(len(state["incidents"]), 1)


if __name__ == "__main__":
    unittest.main()
