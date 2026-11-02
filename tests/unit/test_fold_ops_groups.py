"""One repair budget per submitted action, with all task/cell evidence retained."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.fold_clothes_moving_opscale.operations import agent_dispatcher as d
from experiments.fold_clothes_moving_opscale.operations import recovery_controller as c


class Groups(unittest.TestCase):
    def setUp(self):
        self.pipeline = {
            "actions": {
                "normalization": {"job_id": "123", "array_count": 36},
                "evaluation-smoke": {"job_id": "456"},
            }
        }
        self.raw = {
            f"job:prepare:normalization:123_{i}": {
                "kind": "preflight_job_failed",
                "severity": "error",
                "action": "normalization",
                "job_id": f"123_{i}",
            }
            for i in range(36)
        }
        self.raw.update(
            {
                f"norm:cell{i}": {
                    "kind": "normalization_failed",
                    "severity": "error",
                    "cell": f"cell{i}",
                }
                for i in range(36)
            }
        )
        self.raw["pipeline:blocked"] = {"kind": "pipeline_blocked", "severity": "error"}

    def test_73_symptoms_are_one_action_incident_without_losing_detail(self):
        grouped = c.group_action_alerts(self.raw, self.pipeline)
        self.assertEqual(list(grouped), ["action:normalization:job:123"])
        self.assertEqual(
            grouped["action:normalization:job:123"]["member_alerts"], self.raw
        )
        self.assertEqual(len(self.raw), 73)

    def test_members_arriving_or_clearing_do_not_reset_budget(self):
        state = c.empty_state()
        key = "action:normalization:job:123"
        first = {next(iter(self.raw)): next(iter(self.raw.values()))}
        c.reconcile_incidents(state, c.group_action_alerts(first, self.pipeline), 10)
        incident = next(iter(state["incidents"].values()))
        identity = incident["id"]
        c.claim_incident(state, identity, "test", 1800, 11)
        c.finish_incident(state, identity, "test", "blocked", "receipt", 12)
        c.reconcile_incidents(state, c.group_action_alerts(self.raw, self.pipeline), 15)
        self.assertEqual(len(state["incidents"]), 1)
        self.assertEqual(incident["attempts"], 1)
        self.assertEqual(incident["state"], "NEEDS_REVIEW")
        self.assertEqual(incident["alert_id"], key)
        c.reconcile_incidents(
            state,
            c.group_action_alerts(
                {"norm:cell35": self.raw["norm:cell35"]}, self.pipeline
            ),
            16,
        )
        self.assertEqual(incident["id"], identity)
        self.assertEqual(incident["attempts"], 1)

    def test_existing_three_duplicate_attempts_merge_without_new_budget(self):
        state = c.empty_state()
        c.reconcile_incidents(state, self.raw, 10)
        original = list(state["incidents"].values())[:3]
        for i, item in enumerate(original):
            c.claim_incident(state, item["id"], f"owner{i}", 1800, 20 + i)
            c.finish_incident(
                state, item["id"], f"owner{i}", "blocked", f"receipt{i}", 21 + i
            )
        c.reconcile_incidents(state, c.group_action_alerts(self.raw, self.pipeline), 30)
        current = [i for i in state["incidents"].values() if i["active"]]
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["attempts"], 3)
        self.assertEqual(current[0]["state"], "NEEDS_REVIEW")
        self.assertEqual(len(current[0]["outcomes"]), 3)
        self.assertEqual(
            sum(i["state"] == "COALESCED" for i in state["incidents"].values()), 72
        )
        c.reconcile_incidents(state, c.group_action_alerts(self.raw, self.pipeline), 31)
        self.assertEqual(current[0]["attempts"], 3)
        self.assertEqual(len(current[0]["outcomes"]), 3)

    def test_active_owner_keeps_same_id_claim_and_renewal(self):
        state = c.empty_state()
        c.reconcile_incidents(state, self.raw, 10)
        old = list(state["incidents"].values())[4]
        identity = old["id"]
        c.claim_incident(state, identity, "owner", 1800, 11)
        c.reconcile_incidents(state, c.group_action_alerts(self.raw, self.pipeline), 12)
        self.assertIs(c.owned_incident(state, identity, "owner"), old)
        self.assertEqual(old["state"], "CLAIMED")
        self.assertEqual(old["attempts"], 1)

    def test_distinct_actions_and_old_job_ids_stay_separate(self):
        raw = {
            **self.raw,
            "job:old": {
                "action": "normalization",
                "job_id": "122_2",
                "severity": "error",
            },
            "smoke": {"kind": "evaluation_smoke_failed", "severity": "error"},
        }
        grouped = c.group_action_alerts(raw, self.pipeline)
        self.assertIn("action:normalization:job:122", grouped)
        self.assertIn("action:normalization:job:123", grouped)
        self.assertIn("action:evaluation-smoke:job:456", grouped)
        self.assertIs(grouped["pipeline:blocked"]["automatic_repair"], False)

    def test_member_and_group_manual_holds_are_respected(self):
        with tempfile.TemporaryDirectory() as temp:
            ops = Path(temp)
            key = "action:normalization:job:123"
            incident = {
                "state": "QUEUED",
                "active": True,
                "alert_id": key,
                "alert": c.group_action_alerts(self.raw, self.pipeline)[key],
            }
            for alias in [key, "norm:cell7", "pipeline:blocked"]:
                d.write(
                    ops / "manual_ownership.json",
                    {"alert_ids": [alias], "expires_at": None},
                )
                self.assertFalse(d.eligible(ops, "incident", incident))

    def test_observer_only_alerts_never_get_promoted_into_actionable_group(self):
        raw = {
            "ops": {
                "severity": "error",
                "automatic_repair": False,
                "action": "normalization",
                "job_id": "123",
            }
        }
        self.assertEqual(c.group_action_alerts(raw, self.pipeline), raw)

    def test_cleared_group_then_new_job_has_independent_occurrence(self):
        state = c.empty_state()
        c.reconcile_incidents(state, c.group_action_alerts(self.raw, self.pipeline), 10)
        old = next(iter(state["incidents"].values()))
        c.reconcile_incidents(state, {}, 11)
        self.assertEqual(old["state"], "RESOLVED")
        pipeline = {"actions": {"normalization": {"job_id": "999"}}}
        c.reconcile_incidents(
            state,
            c.group_action_alerts({"norm:cell0": self.raw["norm:cell0"]}, pipeline),
            12,
        )
        active = [i for i in state["incidents"].values() if i["active"]]
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["alert_id"], "action:normalization:job:999")
        self.assertEqual(active[0]["attempts"], 0)


class LeaseRenewal(unittest.TestCase):
    def setUp(self):
        self.record = {
            "run_root": "/tmp/synthetic",
            "incident_id": "incident",
            "owner": "owner",
            "deadline": 4000,
            "lease_expires_at": 1900,
        }

    def test_one_transport_timeout_retries_only_same_lease_then_confirms(self):
        timeout = subprocess.TimeoutExpired(["controller"], 45)
        with (
            patch.object(d.time, "time", return_value=100),
            patch.object(d.time, "sleep") as wait,
            patch.object(d, "controller", side_effect=[timeout, {}]) as call,
        ):
            d.renew_lease(self.record)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(self.record["lease_renewal_retries"], 1)
        self.assertEqual(self.record["lease_expires_at"], 1900)
        wait.assert_called_once_with(5)
        self.assertTrue(
            all(
                x.args[1] == "renew" and x.kwargs["owner"] == "owner"
                for x in call.call_args_list
            )
        )

    def test_explicit_rejection_is_never_retried(self):
        with (
            patch.object(d.time, "time", return_value=100),
            patch.object(
                d, "controller", side_effect=RuntimeError("wrong owner")
            ) as call,
        ):
            with self.assertRaisesRegex(RuntimeError, "wrong owner"):
                d.renew_lease(self.record)
        self.assertEqual(call.call_count, 1)

    def test_three_timeouts_stop_without_extending_known_validity(self):
        with (
            patch.object(d.time, "time", return_value=100),
            patch.object(d.time, "sleep"),
            patch.object(
                d,
                "controller",
                side_effect=subprocess.TimeoutExpired(["controller"], 45),
            ) as call,
        ):
            with self.assertRaisesRegex(RuntimeError, "three times"):
                d.renew_lease(self.record)
        self.assertEqual(call.call_count, 3)
        self.assertEqual(self.record["lease_expires_at"], 1900)

    def test_unknown_or_near_expired_lease_cannot_authorize_retry(self):
        for expiry in [None, 160]:
            record = {**self.record, "lease_expires_at": expiry}
            with (
                patch.object(d.time, "time", return_value=100),
                patch.object(
                    d,
                    "controller",
                    side_effect=subprocess.TimeoutExpired(["controller"], 45),
                ) as call,
            ):
                with self.assertRaises(RuntimeError):
                    d.renew_lease(record)
            self.assertEqual(call.call_count, 1 if expiry is None else 0)

    def test_initial_renewal_cannot_outlive_confirmed_lease(self):
        record = {**self.record, "lease_expires_at": 170}
        with (
            patch.object(d.time, "time", return_value=100),
            patch.object(d, "controller", return_value={}) as call,
        ):
            d.renew_lease(record)
        self.assertEqual(call.call_args.kwargs["timeout_seconds"], 10)
        for expiry in [100, 99, 160]:
            with (
                patch.object(d.time, "time", return_value=100),
                patch.object(d, "controller") as call,
            ):
                with self.assertRaises(RuntimeError):
                    d.renew_lease({**self.record, "lease_expires_at": expiry})
                call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
