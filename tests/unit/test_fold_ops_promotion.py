import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.fold_clothes_moving_opscale.operations import (
    promote_preproduction as p,
)
from experiments.fold_clothes_moving_opscale.operations import recovery_controller as r


class PromotionTests(unittest.TestCase):
    def test_production_registry_always_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "registry.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "Production registry"):
                p.recorded_jobs(root)

    def test_terminal_arrays_must_account_every_task(self):
        with (
            patch.object(
                p, "command", return_value="10_0|COMPLETED|0:0\n10_1|FAILED|2:0"
            ),
            patch.object(p, "run_command", return_value=""),
        ):
            self.assertEqual(len(p.terminal_jobs({"10": 2})["10"]), 2)
            with self.assertRaisesRegex(
                RuntimeError, "active, pending, or unaccounted"
            ):
                p.terminal_jobs({"10": 3})
        with (
            patch.object(
                p, "command", return_value="10_0|COMPLETED|0:0\n10_1|RUNNING|0:0"
            ),
            patch.object(p, "run_command", return_value="10_1|RUNNING"),
        ):
            with self.assertRaises(RuntimeError):
                p.terminal_jobs({"10": 2})

    def test_exact_reviewed_diff_and_science_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            old = base / "old"
            new = base / "new"
            root = base / "runs/task"
            root.mkdir(parents=True)
            rel = "experiments/fold_clothes_moving_opscale/check_readiness.py"
            for repo in (old, new):
                (repo / rel).parent.mkdir(parents=True)
                (repo / rel).write_text("corrected")
                (repo / "egomimic/hydra_configs").mkdir(parents=True)
                (repo / "egomimic/hydra_configs/model.yaml").write_text("batch: 64")
            review = base / "review.json"
            proof = {
                "outcome": "approved",
                "verified": True,
                "science_unchanged": True,
                "tests_passed": True,
                "candidate_repo": str(new),
                "candidate_commit": "b" * 40,
                "candidate_files": [
                    {
                        "path": str(new / rel),
                        "destination": str(new / rel),
                        "sha256": p.sha(new / rel),
                    }
                ],
            }
            p.save(review, proof)
            cfg = {"repo": str(old), "run_root": str(root), "commit": "a" * 40}
            with (
                patch.object(p, "source_guard"),
                patch.object(p, "command", side_effect=["", f"M\t{rel}"]),
            ):
                changed, frozen = p.review_candidate(cfg, new, "b" * 40, review)
                self.assertEqual(changed, {rel: p.sha(new / rel)})
            (new / "egomimic/hydra_configs/model.yaml").write_text("batch: 32")
            with (
                patch.object(p, "source_guard"),
                patch.object(p, "command", side_effect=["", f"M\t{rel}"]),
            ):
                with self.assertRaisesRegex(RuntimeError, "Scientific"):
                    p.review_candidate(cfg, new, "b" * 40, review)
            proof["candidate_commit"] = "c" * 40
            p.save(review, proof)
            with (
                patch.object(p, "source_guard"),
                patch.object(p, "command", return_value=""),
            ):
                with self.assertRaisesRegex(RuntimeError, "exact candidate"):
                    p.review_candidate(cfg, new, "b" * 40, review)

    def test_worktree_under_runs_forbidden(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "runs/task"
            root.mkdir(parents=True)
            cfg = {"repo": str(Path(d) / "old"), "run_root": str(root)}
            with self.assertRaisesRegex(RuntimeError, "cannot live under"):
                p.review_candidate(
                    cfg, root / "candidate", "b" * 40, root / "review.json"
                )

    def test_archive_resumes_without_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            root.mkdir()
            archive = Path(d) / "archive"
            (root / "readiness.json").write_text("original")
            p.archive_one(root, archive, "readiness.json")
            p.archive_one(root, archive, "readiness.json")
            self.assertEqual((archive / "readiness.json").read_text(), "original")
            (root / "readiness.json").write_text("new")
            with self.assertRaisesRegex(RuntimeError, "Archive conflict"):
                p.archive_one(root, archive, "readiness.json")

    def test_quiesce_exact_identity_only(self):
        spec = {
            "pid": 123,
            "script": "/known/pipeline.py",
            "commit": "a" * 40,
            "start_ticks": 42,
        }
        with (
            patch.object(
                p, "process_state", return_value={"state": "IDENTITY_MISMATCH"}
            ),
            patch.object(p.os, "kill") as kill,
        ):
            with self.assertRaises(RuntimeError):
                p.stop_exact(spec, Path("/run"))
            kill.assert_not_called()
        with (
            patch.object(
                p,
                "process_state",
                side_effect=[{"state": "RUNNING"}, {"state": "EXITED"}],
            ),
            patch.object(p.time, "sleep"),
            patch.object(p.os, "kill") as kill,
        ):
            p.stop_exact(spec, Path("/run"))
            kill.assert_called_once_with(123, p.signal.SIGTERM)

    def test_ambiguous_spawn_never_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            tx = {
                "new_workers": {"pipeline": {"state": "STARTING"}},
                "plan": {"new_repo": "/repo"},
            }
            with patch.object(p.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "no duplicate"):
                    p.spawn_known(
                        tx,
                        root / "transaction.json",
                        "pipeline",
                        [],
                        Path("/repo/pipeline.py"),
                        "a" * 40,
                        root,
                    )
                spawn.assert_not_called()
            tx["new_workers"]["pipeline"].update(pid=123)
            with patch.object(p.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "no duplicate"):
                    p.spawn_known(
                        tx,
                        root / "transaction.json",
                        "pipeline",
                        [],
                        Path("/repo/pipeline.py"),
                        "a" * 40,
                        root,
                    )
                spawn.assert_not_called()
            tx["new_workers"]["pipeline"].update(start_ticks=42)
            with (
                patch.object(
                    p,
                    "process_state",
                    return_value={"state": "RUNNING", "start_ticks": 42},
                ),
                patch.object(p.subprocess, "Popen") as spawn,
            ):
                self.assertEqual(
                    p.spawn_known(
                        tx,
                        root / "transaction.json",
                        "pipeline",
                        [],
                        Path("/repo/pipeline.py"),
                        "a" * 40,
                        root,
                    )["state"],
                    "VERIFIED",
                )
                spawn.assert_not_called()

    def test_complete_crash_before_maintenance_clear_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ops = root / "operations"
            identity = "a" * 32
            folder = ops / "promotions" / identity
            folder.mkdir(parents=True)
            config = {"run_root": str(root), "repo": "/old", "commit": "old"}
            p.save(ops / "recovery_config.json", config)
            plan = {
                "schema_version": 1,
                "run_root": str(root),
                "transaction_id": identity,
                "archive_paths": list(p.ARCHIVE),
                "recovery_config": config,
                "old_repo": "/old",
                "old_commit": "old",
                "new_repo": "/new",
                "new_commit": "new",
            }
            path = ops / "plan.json"
            p.save(path, plan)
            p.save(
                folder / "transaction.json",
                {"state": "COMPLETE", "plan_sha256": p.sha(path)},
            )
            p.save(
                folder / "complete.json",
                {
                    "state": "COMPLETE",
                    "transaction_id": identity,
                    "completed_at": "now",
                },
            )
            p.save(
                ops / "promotion_maintenance.json",
                {"active": True, "transaction_id": identity},
            )
            result = p.apply_plan(path)
            self.assertEqual(result["state"], "COMPLETE")
            self.assertFalse(p.load(ops / "promotion_maintenance.json")["active"])

    def test_maintenance_pauses_controller(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "operations").mkdir()
            p.save(
                root / "operations/promotion_maintenance.json",
                {
                    "active": True,
                    "transaction_id": "id",
                    "deadline_epoch": r.time.time() + 3600,
                },
            )
            with patch.object(r, "inspect_local") as local:
                self.assertEqual(r.reconcile(root, True)["state"], "MAINTENANCE")
                local.assert_not_called()

    def test_expired_maintenance_blocks_without_actions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "operations").mkdir()
            p.save(
                root / "operations/promotion_maintenance.json",
                {"active": True, "transaction_id": "id", "deadline_epoch": 1},
            )
            with patch.object(r, "inspect_local") as local:
                self.assertEqual(r.reconcile(root, True)["state"], "BLOCKED")
                local.assert_not_called()
            self.assertTrue(
                p.load(root / "operations/promotion_maintenance.json")["active"]
            )

    def test_resume_rechecks_review_and_scientific_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            review = root / "review.json"
            review.write_text("{}")
            plan = {
                "run_root": str(root),
                "private_files": {"split": "hash"},
                "review_receipt": str(review),
                "review_receipt_sha256": p.sha(review),
                "recovery_config": {},
                "new_repo": "/new",
                "new_commit": "b" * 40,
                "reviewed_files": {"code": "sha"},
                "scientific_files": {"config": "sha"},
            }
            with (
                patch.object(p, "private_files", return_value={"split": "hash"}),
                patch.object(
                    p,
                    "review_candidate",
                    return_value=({"code": "sha"}, {"config": "sha"}),
                ),
            ):
                p.validate_evidence(plan)
                review.write_text('{"outcome":"rejected"}')
                with self.assertRaisesRegex(RuntimeError, "Review receipt changed"):
                    p.validate_evidence(plan)

    def test_manifest_identity_changes_preserve_science(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p.save(
                root / "run_manifest.json",
                {
                    "commit": "old",
                    "training_batch": 64,
                    "monitoring": {"interval_seconds": 120},
                },
            )
            plan = {
                "old_commit": "old",
                "new_commit": "new",
                "new_repo": "/new",
                "transaction_id": "id",
            }
            workers = {
                "pipeline": {"pid": 1},
                "monitor": {"pid": 2},
                "supervisor": {"pid": 3},
            }
            p.update_manifest(root, plan, workers)
            result = p.load(root / "run_manifest.json")
            self.assertEqual(result["training_batch"], 64)
            self.assertEqual(result["commit"], "new")
            self.assertEqual(result["monitoring"]["supervisor_pid"], 3)
            self.assertEqual((root / "supervisor.pid").read_text().strip(), "3")

    def test_plan_rejects_archive_escape_uuid_and_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ops = root / "operations"
            ops.mkdir()
            config = {"run_root": str(root), "repo": "/old", "commit": "old"}
            p.save(ops / "recovery_config.json", config)
            plan = {
                "schema_version": 1,
                "run_root": str(root),
                "transaction_id": "a" * 32,
                "archive_paths": list(p.ARCHIVE),
                "recovery_config": config,
                "old_repo": "/old",
                "old_commit": "old",
                "new_repo": "/new",
                "new_commit": "new",
            }
            p.validate_plan_structure(plan, ops / "plan.json")
            for bad in (
                {**plan, "archive_paths": list(p.ARCHIVE) + ["../outside"]},
                {**plan, "transaction_id": "../outside"},
                {**plan, "old_repo": "/other"},
                {**plan, "recovery_config": {**config, "run_root": "/other"}},
            ):
                with self.assertRaises(RuntimeError):
                    p.validate_plan_structure(bad, ops / "plan.json")

    def test_watch_only_receipt_is_stable_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "operations").mkdir()
            argv = ["controller", "watch", "--run-root", str(root)]
            identity = {"pid": 123, "start_ticks": 42, "host": "pace", "pgid": 123}
            with (
                patch("sys.argv", argv),
                patch.object(r, "watch_identity", return_value=identity),
                patch.object(r, "reconcile", return_value={"state": "COMPLETE"}),
                patch("builtins.print"),
            ):
                r.main()
            report = p.load(root / "operations/recovery_watch_status.json")
            self.assertEqual(report["identity"], identity)
            self.assertEqual(report["state"], "COMPLETE")


if __name__ == "__main__":
    unittest.main()
