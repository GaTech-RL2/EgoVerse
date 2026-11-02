"""Hermetic dispatcher tests: synthetic incidents and fake executable, no model/GPU calls."""

import fcntl
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.fold_clothes_moving_opscale.operations import agent_dispatcher as d


class DispatcherTests(unittest.TestCase):
    def setUp(self):
        self.guard_patch = patch.object(d, "source_guard", return_value=None)
        self.guard_patch.start()
        self.addCleanup(self.guard_patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ops = self.root / "operations"
        self.ops.mkdir()
        d.write(self.root / "production_control.json", {"allow_training": False})
        self.folder = self.ops / "agents/attempt"
        self.folder.mkdir(parents=True)
        self.incident = {
            "alert_id": "test:failure",
            "alert": {"severity": "error"},
            "active": True,
            "state": "QUEUED",
            "attempts": 0,
            "available_after": 1,
        }
        self.record = {
            "incident_id": "abc",
            "alert_id": "test:failure",
            "alert": {},
            "owner": "test:owner",
            "run_root": str(self.root),
            "folder": str(self.folder),
            "repo": str(self.root / "pinned"),
            "commit": "a" * 40,
            "deadline": time.time() + 20,
            "created_at": d.stamp(),
            "state": "START_INTENT",
        }
        self.evidence = self.folder / "evidence.log"
        self.evidence.write_text("verified synthetic process state\n")

    def tearDown(self):
        self.temp.cleanup()

    def answer(self, outcome="fixed", **kwargs):
        result = {
            "outcome": outcome,
            "verified": True,
            "science_unchanged": True,
            "tests_passed": True,
            "code_changes": False,
            "approval_rejected": False,
            "summary": "synthetic result",
            "commands": [],
            "job_ids": [],
            "changed_files": [],
            "verification": ["synthetic verification"],
            "evidence_files": [str(self.evidence)],
            "candidate_files": [],
        }
        return {**result, **kwargs}

    def candidate(self):
        path = self.folder / "candidate.py"
        path.write_text("print(1)\n")
        return {
            "path": str(path),
            "sha256": d.digest(path),
            "destination": str(self.root / "wrapper.py"),
        }

    def fake_executable(self, answer=None, hang=False):
        executable = self.folder / "fake_codex"
        executable.write_text(
            "#!/usr/bin/env python3\nimport json,sys,time\n"
            + ("time.sleep(30)\n" if hang else "")
            + "p=sys.argv[sys.argv.index('--output-last-message')+1]\n"
            + 'with open(p,"w") as f: json.dump('
            + repr(answer or self.answer())
            + ",f)\n"
        )
        executable.chmod(0o700)
        self.record["codex"] = str(executable)

    def test_only_active_queued_actionable_incidents(self):
        self.assertTrue(d.eligible(self.ops, "abc", self.incident))
        for field, value in [
            ("active", False),
            ("state", "CLAIMED"),
            ("state", "LEASE_EXPIRED"),
            ("attempts", 3),
            ("available_after", time.time() + 30),
        ]:
            self.assertFalse(
                d.eligible(self.ops, "abc", {**self.incident, field: value})
            )

    def test_manual_ownership_expiry_and_persistent_hold(self):
        d.write(
            self.ops / "manual_ownership.json",
            {"owner": "root", "alert_ids": ["test:failure"], "expires_at": None},
        )
        self.assertFalse(d.eligible(self.ops, "abc", self.incident))
        d.write(
            self.ops / "manual_ownership.json",
            {
                "owner": "root",
                "alert_ids": ["test:failure"],
                "expires_at": time.time() - 1,
            },
        )
        self.assertTrue(d.eligible(self.ops, "abc", self.incident))

    def test_local_attempts_limit_cooldown_and_unknown_start(self):
        d.write(self.folder / "attempt.json", {**self.record, "state": "FINISHED"})
        self.assertFalse(d.eligible(self.ops, "abc", self.incident))
        old = {**self.record, "created_at": time.time() - 1801, "state": "START_INTENT"}
        d.write(self.folder / "attempt.json", old)
        self.assertFalse(d.eligible(self.ops, "abc", self.incident))
        for i in range(3):
            d.write(self.ops / f"agents/{i}/attempt.json", {**old, "state": "FINISHED"})
        self.assertFalse(d.eligible(self.ops, "abc", self.incident))

    def test_command_preserves_approval_and_model_configuration(self):
        cmd = d.agent_command(
            "/tmp/codex", self.folder, "repair", self.ops / "schema.json"
        )
        self.assertIn("--approve-for-me", cmd)
        self.assertIn("--json", cmd)
        self.assertIn("--output-schema", cmd)
        self.assertIn("--output-last-message", cmd)
        self.assertNotIn("--model", cmd)
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", cmd)
        self.assertNotIn("--ignore-rules", cmd)

    def test_fixed_requires_real_evidence_and_independent_code_review(self):
        with self.assertRaises(ValueError):
            d.checked_answer(self.answer(evidence_files=[]), "repair", self.folder)
        with self.assertRaises(ValueError):
            d.checked_answer(self.answer(code_changes=True), "repair", self.folder)
        self.assertEqual(
            d.checked_answer(self.answer(), "repair", self.folder)["outcome"], "fixed"
        )
        self.assertEqual(
            d.checked_answer(
                self.answer(approval_rejected=True), "repair", self.folder
            )["outcome"],
            "blocked",
        )

    def test_candidates_cannot_escape_workspace_or_change_after_review(self):
        c = self.candidate()
        a = self.answer("review", code_changes=True, candidate_files=[c])
        self.assertEqual(d.candidate_manifest(a, self.folder), [c])
        Path(c["path"]).write_text("changed")
        with self.assertRaises(ValueError):
            d.candidate_manifest(a, self.folder)
        c["path"] = str(self.evidence.parent.parent / "outside")
        Path(c["path"]).write_text("outside")
        c["sha256"] = d.digest(c["path"])
        with self.assertRaises(ValueError):
            d.candidate_manifest(a, self.folder)

    def test_fixed_code_change_requires_exact_reviewed_bytes_at_destination(self):
        c = self.candidate()
        a = self.answer(
            "fixed",
            code_changes=True,
            candidate_files=[c],
            changed_files=[c["destination"]],
        )
        with self.assertRaises(ValueError):
            d.checked_answer(a, "apply", self.folder, [c])
        Path(c["destination"]).write_text("wrong bytes")
        with self.assertRaises(ValueError):
            d.checked_answer(a, "apply", self.folder, [c])
        Path(c["destination"]).write_bytes(Path(c["path"]).read_bytes())
        self.assertEqual(
            d.checked_answer(a, "apply", self.folder, [c])["outcome"], "fixed"
        )
        with self.assertRaises(ValueError):
            d.checked_answer({**a, "changed_files": []}, "apply", self.folder, [c])
        with self.assertRaises(ValueError):
            d.checked_answer({**a, "candidate_files": []}, "apply", self.folder, [c])

    def test_review_attests_exact_tested_manifest(self):
        c = self.candidate()
        a = self.answer("approved", candidate_files=[c])
        self.assertEqual(
            d.checked_answer(a, "review", self.folder, [c])["outcome"], "approved"
        )
        with self.assertRaises(ValueError):
            d.checked_answer(a, "review", self.folder, [])
        with self.assertRaises(ValueError):
            d.checked_answer({**a, "tests_passed": False}, "review", self.folder, [c])

    def test_pid_reuse_or_remote_host_is_not_live_identity(self):
        current = d.identity(os.getpid())
        self.assertTrue(d.alive(current))
        self.assertFalse(
            d.alive({**current, "start_ticks": current["start_ticks"] + 1})
        )
        self.assertFalse(d.alive({**current, "host": "another-host"}))

    def test_ambiguous_crash_is_blocked_without_model_relaunch(self):
        d.write(self.folder / "attempt.json", self.record)
        with patch.object(d, "controller", return_value={}) as control:
            self.assertTrue(d.reconcile_local(self.ops, self.root))
        self.assertEqual(d.read(self.folder / "attempt.json")["state"], "BLOCKED")
        self.assertEqual(control.call_args.kwargs["outcome"], "blocked")

    def test_unknown_remote_runner_prevents_dispatch(self):
        d.write(
            self.folder / "attempt.json",
            {**self.record, "runner": {"host": "different-host"}},
        )
        with patch.object(d, "controller") as control:
            self.assertFalse(d.reconcile_local(self.ops, self.root))
            control.assert_not_called()

    def test_launch_intent_and_inherited_lock_precede_popen(self):
        args = SimpleNamespace(
            run_root=self.root,
            repo=self.root / "pinned",
            commit="a" * 40,
            codex_bin=Path("/tmp/codex"),
        )
        observed = []

        def popen(command, **kwargs):
            rec = d.read(Path(command[command.index("--attempt") + 1]))
            observed.append(rec["state"])
            self.assertEqual(kwargs["pass_fds"], (123,))
            return SimpleNamespace(pid=99999999)

        with (
            patch.object(d, "controller", return_value={}),
            patch.object(d.subprocess, "Popen", side_effect=popen),
        ):
            d.start_attempt(args, "abc", self.incident, 123)
        self.assertEqual(observed, ["START_INTENT"])

    def test_actual_fake_agent_process_and_lease_renewal(self):
        self.fake_executable()
        with patch.object(d, "controller", return_value={}) as control:
            answer = d.run_phase(self.record, "repair")
        self.assertEqual(answer["outcome"], "fixed")
        self.assertTrue(any(c.args[1] == "renew" for c in control.call_args_list))
        self.assertEqual(d.read(self.folder / "attempt.json")["phase_state"], "EXITED")
        self.assertTrue((self.folder / "repair.launch.json").exists())

    def test_lease_rejection_stops_identified_child(self):
        self.fake_executable(hang=True)
        with patch.object(
            d, "controller", side_effect=[{}, RuntimeError("lease lost")]
        ):
            with self.assertRaisesRegex(RuntimeError, "lease lost"):
                d.run_phase(self.record, "repair")
        saved = d.read(self.folder / "attempt.json")["child"]
        self.assertFalse(d.alive(saved))

    def test_invalid_initial_lease_never_starts_a_model(self):
        self.fake_executable()
        with (
            patch.object(d, "controller", side_effect=RuntimeError("expired lease")),
            patch.object(d.subprocess, "Popen") as launch,
        ):
            with self.assertRaisesRegex(RuntimeError, "expired lease"):
                d.run_phase(self.record, "repair")
            launch.assert_not_called()

    def test_no_model_call_after_deadline(self):
        self.record["deadline"] = time.time() - 1
        with patch.object(d.subprocess, "Popen") as popen:
            with self.assertRaises(TimeoutError):
                d.run_phase(self.record, "repair")
            popen.assert_not_called()

    def test_independent_review_before_apply(self):
        c = self.candidate()
        repair = self.answer("review", code_changes=True, candidate_files=[c])
        review = self.answer("approved", candidate_files=[c])
        applied = self.answer("fixed", code_changes=True)
        d.write(self.folder / "attempt.json", self.record)
        with (
            patch.object(
                d, "run_phase", side_effect=[repair, review, applied]
            ) as phases,
            patch.object(d, "controller", return_value={}),
        ):
            d.run_attempt(self.folder / "attempt.json")
        self.assertEqual(
            [c.args[1] for c in phases.call_args_list], ["repair", "review", "apply"]
        )
        self.assertEqual(
            d.read(self.folder / "receipt.json")["result"]["outcome"], "fixed"
        )

    def test_rejected_review_never_applies(self):
        c = self.candidate()
        repair = self.answer("review", code_changes=True, candidate_files=[c])
        d.write(self.folder / "attempt.json", self.record)
        with (
            patch.object(
                d, "run_phase", side_effect=[repair, self.answer("rejected")]
            ) as phases,
            patch.object(d, "controller", return_value={}),
        ):
            d.run_attempt(self.folder / "attempt.json")
        self.assertEqual(len(phases.call_args_list), 2)
        self.assertEqual(
            d.read(self.folder / "receipt.json")["result"]["outcome"], "blocked"
        )

    def test_failed_claim_does_not_consume_local_repair_attempt(self):
        d.write(
            self.folder / "attempt.json",
            {**self.record, "state": "BLOCKED", "outcome": "claim_rejected"},
        )
        self.assertTrue(d.eligible(self.ops, "abc", self.incident))

    def test_initialized_controller_identity_must_match(self):
        args = SimpleNamespace(
            run_root=self.root, repo=self.root / "pinned", commit="a" * 40
        )
        with self.assertRaises(RuntimeError):
            d.control_identity(args)
        d.write(
            self.ops / "recovery_config.json",
            {"run_root": str(self.root), "repo": str(args.repo), "commit": args.commit},
        )
        d.control_identity(args)
        args.commit = "b" * 40
        with self.assertRaises(RuntimeError):
            d.control_identity(args)

    def test_another_active_controller_claim_prevents_even_a_claim_attempt(self):
        args = SimpleNamespace(
            run_root=self.root,
            repo=self.root / "pinned",
            commit="a" * 40,
            codex_bin=Path("/tmp/codex"),
            once=True,
        )
        d.write(self.ops / "dispatcher_control.json", {"enabled": True})
        d.write(
            self.ops / "recovery_state.json",
            {"incidents": {"busy": {"state": "LEASE_EXPIRED"}, "abc": self.incident}},
        )
        with (
            patch.object(d, "control_identity"),
            patch.object(d, "controller", return_value={}),
            patch.object(d, "start_attempt") as launch,
        ):
            d.serve(args)
            launch.assert_not_called()

    def test_observer_only_health_alert_is_never_dispatched(self):
        held = {
            **self.incident,
            "alert": {"severity": "error", "automatic_repair": False},
        }
        self.assertFalse(d.eligible(self.ops, "abc", held))

    def test_no_code_label_cannot_bypass_review_with_changed_files(self):
        with self.assertRaises(ValueError):
            d.checked_answer(
                self.answer(changed_files=["/tmp/edited.py"]), "repair", self.folder
            )
        with self.assertRaises(ValueError):
            d.checked_answer(
                self.answer(candidate_files=[self.candidate()]), "repair", self.folder
            )

    def test_duplicate_attempt_runner_is_locked_and_terminal_attempt_cannot_restart(
        self,
    ):
        d.write(self.folder / "attempt.json", self.record)
        with (
            (self.folder / "runner.lock").open("a+") as lock,
            patch.object(d, "run_phase") as phase,
        ):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(d.run_attempt(self.folder / "attempt.json"), 2)
            phase.assert_not_called()
        d.write(self.folder / "attempt.json", {**self.record, "state": "FINISHED"})
        with patch.object(d, "run_phase") as phase:
            self.assertEqual(d.run_attempt(self.folder / "attempt.json"), 2)
            phase.assert_not_called()

    def test_pinned_source_rechecked_after_agent_exit(self):
        self.fake_executable()
        with (
            patch.object(d, "controller", return_value={}),
            patch.object(
                d, "source_guard", side_effect=[None, RuntimeError("source changed")]
            ) as guard,
        ):
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                d.run_phase(self.record, "repair")
        self.assertEqual(guard.call_count, 2)

    def test_completed_promotion_adoption_requires_exact_transaction_and_config_hash(
        self,
    ):
        args = SimpleNamespace(
            run_root=self.root, repo=self.root / "pinned", commit="a" * 40
        )
        config = {
            "run_root": str(self.root),
            "repo": str(self.root / "candidate"),
            "commit": "b" * 40,
        }
        d.write(self.ops / "recovery_config.json", config)
        d.write(
            self.ops / "promotion_maintenance.json",
            {"active": False, "transaction_id": "transaction"},
        )
        receipt = {
            "state": "COMPLETE",
            "transaction_id": "transaction",
            "old_repo": str(args.repo),
            "old_commit": args.commit,
            "new_repo": config["repo"],
            "new_commit": config["commit"],
            "recovery_config_sha256": "bad",
        }
        d.write(self.ops / "promotion_complete.json", receipt)
        with self.assertRaises(RuntimeError):
            d.control_identity(args)
        receipt["recovery_config_sha256"] = d.digest(self.ops / "recovery_config.json")
        d.write(self.ops / "promotion_complete.json", receipt)
        self.assertTrue(d.control_identity(args))
        self.assertEqual(args.commit, "b" * 40)
        self.assertEqual(str(args.repo), config["repo"])
        self.assertTrue((self.ops / "dispatcher_adoption.json").exists())

    def test_promotion_maintenance_suppresses_dispatch_and_identity_migration(self):
        args = SimpleNamespace(
            run_root=self.root, repo=self.root / "pinned", commit="a" * 40
        )
        d.write(
            self.ops / "promotion_maintenance.json",
            {"active": True, "transaction_id": "in-progress"},
        )
        self.assertFalse(d.control_identity(args))
        self.assertEqual(args.commit, "a" * 40)

    def test_end_to_end_real_controller_fake_agents_review_apply_and_resolve(self):
        repo = self.root / "pinned"
        scripts = repo / "experiments/fold_clothes_moving_opscale"
        scripts.mkdir(parents=True)
        for name in ["pipeline", "monitor"]:
            (scripts / f"{name}.py").write_text("import time\ntime.sleep(90)\n")
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=Dispatcher Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-qm",
                "synthetic workers",
            ],
            check=True,
        )
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        workers = []
        try:
            for name in ["pipeline", "monitor"]:
                cmd = [
                    sys.executable,
                    str(scripts / f"{name}.py"),
                    "--run-root",
                    str(self.root),
                ]
                if name == "pipeline":
                    cmd += ["--commit", commit]
                workers.append(
                    subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                )
            d.controller(
                self.root,
                "init",
                repo=repo,
                commit=commit,
                python=sys.executable,
                pipeline_pid=workers[0].pid,
                monitor_pid=workers[1].pid,
                gpu="h200",
            )
            d.write(
                self.ops / "supervision.json",
                {
                    "state": "ACTIVE",
                    "updated_at": d.stamp(),
                    "alerts": {
                        "test:wrapper": {
                            "kind": "synthetic_wrapper_failure",
                            "severity": "error",
                        }
                    },
                },
            )
            d.write(self.ops / "dispatcher_control.json", {"enabled": True})
            fake = self.root / "fake_codex"
            fake.write_text("""#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[sys.argv.index('--output-last-message')+1]);folder=out.parent
root=folder.parents[2];candidate=folder/'candidate.py';dest=root/'fixed_wrapper.py'
phase=out.stem
if phase=='repair':candidate.write_text('print("reviewed synthetic fix")\\n')
checksum=hashlib.sha256(candidate.read_bytes()).hexdigest()
evidence=folder/'test_evidence.log';evidence.write_text('synthetic wrapper validation passed\\n')
if phase=='apply':dest.write_bytes(candidate.read_bytes())
result={'outcome':{'repair':'review','review':'approved','apply':'fixed'}[phase],
 'summary':'synthetic isolated end-to-end repair','verified':True,'science_unchanged':True,
 'tests_passed':True,'code_changes':True,'approval_rejected':False,'commands':['synthetic verification'],
 'job_ids':[],'changed_files':[str(dest)] if phase=='apply' else [],
 'verification':['synthetic wrapper bytes checked'],'evidence_files':[str(evidence)],
 'candidate_files':[{'path':str(candidate),'sha256':checksum,'destination':str(dest)}],
 'candidate_repo':'','candidate_commit':''}
out.write_text(json.dumps(result))
""")
            fake.chmod(0o700)
            args = SimpleNamespace(
                run_root=self.root, repo=repo, commit=commit, codex_bin=fake, once=True
            )
            self.assertEqual(d.serve(args), 0)
            deadline = time.time() + 25
            finished = []
            while time.time() < deadline:
                finished = [
                    d.read(p)
                    for p in (self.ops / "agents").glob("*/attempt.json")
                    if d.read(p, {}).get("state") == "FINISHED"
                ]
                if finished:
                    break
                time.sleep(0.1)
            self.assertEqual(
                len(finished),
                1,
                [
                    (str(p), d.read(p))
                    for p in (self.ops / "agents").glob("*/attempt.json")
                ],
            )
            self.assertEqual(finished[0]["outcome"], "fixed")
            state = d.read(self.ops / "recovery_state.json")
            incident = state["incidents"][finished[0]["incident_id"]]
            self.assertEqual(incident["state"], "AWAITING_VERIFICATION")
            folder = Path(finished[0]["folder"])
            self.assertTrue(
                all(
                    (folder / f"{phase}.launch.json").is_file()
                    for phase in ["repair", "review", "apply"]
                )
            )
            self.assertEqual(
                (self.root / "fixed_wrapper.py").read_bytes(),
                (folder / "candidate.py").read_bytes(),
            )
            d.write(
                self.ops / "supervision.json",
                {"state": "ACTIVE", "updated_at": d.stamp(), "alerts": {}},
            )
            d.controller(self.root, "reconcile")
            self.assertEqual(
                d.read(self.ops / "recovery_state.json")["incidents"][incident["id"]][
                    "state"
                ],
                "RESOLVED",
            )
        finally:
            for worker in workers:
                worker.terminate()
                worker.wait(timeout=5)

    def test_source_worktrees_are_siblings_outside_runtime_artifacts(self):
        pinned = self.root / "project/EgoVerse"
        root = self.root / "project/egoverse-runs/run"
        valid = self.root / "project/fold-moving-repair-abc"
        self.assertEqual(d.candidate_location(pinned, root, valid), valid.resolve())
        for invalid in [
            pinned,
            root / "operations/fold-moving-repair-abc",
            self.root / "project/other",
            self.root / "elsewhere/fold-moving-repair-abc",
        ]:
            with self.assertRaises(ValueError):
                d.candidate_location(pinned, root, invalid)

    def test_scope_policy_cannot_be_missing_or_implicitly_allow_production(self):
        d.scope_guard(self.record)
        d.write(self.root / "production_control.json", {"allow_training": True})
        with self.assertRaises(RuntimeError):
            d.scope_guard(self.record)
        (self.root / "production_control.json").unlink()
        with self.assertRaises(RuntimeError):
            d.scope_guard(self.record)

    def production_control(self):
        control = {
            "schema_version": 1,
            "allow_training": True,
            "scope": "production",
            "user_instruction": "Launch the existing 36 runs and monitor them.",
        }
        d.write(self.root / "production_control.json", control)
        return control

    def test_production_agent_receives_current_authorization_and_pinned_source(self):
        self.production_control()
        self.fake_executable()
        with patch.object(d, "controller", return_value={}):
            result = d.run_phase(self.record, "repair")
        self.assertEqual(result["outcome"], "fixed")
        prompt = (self.folder / "repair.prompt.md").read_text()
        self.assertIn("CURRENT AUTHORIZED SCOPE: production", prompt)
        self.assertIn("existing pipeline owns initial production submission", prompt)
        self.assertIn(self.record["commit"], prompt)
        self.assertTrue(self.record["authorization_scope"]["allow_training"])

    def test_scope_change_between_review_and_apply_never_starts_another_agent(self):
        d.scope_guard(self.record)
        self.production_control()
        with patch.object(d.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(RuntimeError, "authorization changed"):
                d.run_phase(self.record, "apply")
        launch.assert_not_called()

    def test_revocation_stops_running_agent_without_waiting_for_lease_expiry(self):
        self.production_control()
        self.fake_executable(hang=True)
        renewals = 0

        def renew(*args, **kwargs):
            nonlocal renewals
            renewals += 1
            if renewals == 2:
                d.write(
                    self.root / "production_control.json", {"allow_training": False}
                )
            return {}

        with patch.object(d, "controller", side_effect=renew):
            with self.assertRaisesRegex(RuntimeError, "authorization changed"):
                d.run_phase(self.record, "repair")
        saved = d.read(self.folder / "attempt.json")["child"]
        self.assertFalse(d.alive(saved))
        self.assertEqual(renewals, 2)

    def test_revocation_during_initial_renewal_never_starts_agent(self):
        self.production_control()
        self.fake_executable()

        def revoke(*args, **kwargs):
            d.write(self.root / "production_control.json", {"allow_training": False})
            return {}

        with (
            patch.object(d, "controller", side_effect=revoke),
            patch.object(d.subprocess, "Popen") as launch,
        ):
            with self.assertRaisesRegex(RuntimeError, "authorization changed"):
                d.run_phase(self.record, "repair")
        launch.assert_not_called()

    def test_invalid_control_does_not_claim_or_launch(self):
        d.write(self.root / "production_control.json", {"allow_training": True})
        args = SimpleNamespace(run_root=self.root)
        with (
            patch.object(d, "controller") as claim,
            patch.object(d.subprocess, "Popen") as launch,
        ):
            with self.assertRaisesRegex(RuntimeError, "Invalid production control"):
                d.start_attempt(args, "abc", self.incident, 123)
        claim.assert_not_called()
        launch.assert_not_called()

    def test_disabled_dispatcher_makes_no_controller_or_model_calls(self):
        args = SimpleNamespace(
            run_root=self.root,
            repo=self.root / "pinned",
            commit="a" * 40,
            codex_bin=Path("/tmp/codex"),
            once=True,
        )
        with (
            patch.object(d, "controller") as control,
            patch.object(d, "start_attempt") as launch,
        ):
            d.serve(args)
            control.assert_not_called()
            launch.assert_not_called()
        self.assertEqual(
            d.read(self.ops / "dispatcher_status.json")["state"], "DISABLED"
        )


if __name__ == "__main__":
    unittest.main()
