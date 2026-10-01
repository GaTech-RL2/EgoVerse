"""ASPIRE-inspired search over demo programs, followed by frozen evaluation.

This learns an external program/library, not pi0.5 parameters. The two arms keep
separate libraries. Only real simulator outcomes populate the search archive.
"""

import copy
import json
from pathlib import Path

import numpy as np

from .action_adapter import ActionAdapter, ActionSpec
from .demo_segments import write_json
from .demo_skill_agent import build_request, snapshot_wire
from .demo_skill_conditioning import InputSkillConditioner
from .demo_skill_program import (
    BehaviorLibrary,
    ProgramExecutor,
    composed_reference,
    has_effect,
)
from .intervention_rollout import run_rollout
from .records import Recorder, digest, file_sha256, to_numpy

NATIVE = {"native": True, "stages": []}
SOLVER = {"solver": "euler", "steps": 10, "time_power": 1.0}


def load_protocol(path=None):
    path = path or Path(__file__).parent / "configs/demo_skill_library_v1.json"
    value = json.loads(Path(path).read_text())
    if (
        value["schema_version"] != "demo-skill-library-1"
        or value["execute_steps"] != 5
        or value["action_budget"] != 300
    ):
        raise ValueError("Unexpected demo-skill experiment protocol")
    splits = [
        value[key]
        for key in ("development_states", "validation_states", "evaluation_states")
    ]
    if any(
        not s or len(set(s)) != len(s) or any(type(i) is not int or i < 0 for i in s)
        for s in splits
    ):
        raise ValueError("Reset splits must contain distinct nonnegative integers")
    if any(set(a) & set(b) for i, a in enumerate(splits) for b in splits[i + 1 :]):
        raise ValueError("Development, validation and evaluation resets overlap")
    if value["source_task_ids"] != list(range(40)) or value["task_ids"] != list(
        range(10)
    ):
        raise ValueError(
            "Full protocol requires 40 standard sources and all 20 OOD tasks"
        )
    if value["astra"] != {
        "model": "gpt-6-astra",
        "reasoning_effort": "medium",
        "timeout_seconds": 240,
    }:
        raise ValueError("Unversioned Astra harness change")
    if not 1 <= value["population"] <= 4 or not 1 <= value["rounds"] <= 5:
        raise ValueError("Search exceeds its declared finite budget")
    return value


def evidence_for(result):
    return {
        "reset_id": result["episode_id"],
        "split": result["split"],
        "success": result["success"],
        "attempt_id": result["attempt_id"],
        "trace_sha256": result["trace_sha256"],
    }


def visible_history(result, program):
    return {
        "attempt_id": result["attempt_id"],
        "success": result["success"],
        "actions_executed": result["actions_executed"],
        "program": copy.deepcopy(program),
        "snapshots": [
            snapshot_wire(s, f"REAL completed attempt {result['attempt_id']}")
            for s in result["snapshots"]
        ],
    }


def archive_history(row):
    # Prefer a failure trace when a program only succeeded on one of two resets.
    result = next((r for r in row["results"] if not r["success"]), row["results"][-1])
    return {
        **visible_history(result, row["program"]),
        "development_successes": row["score"],
        "development_trials": [
            {"episode_id": r["episode_id"], "success": r["success"]}
            for r in row["results"]
        ],
    }


class DemoSkillExperiment:
    def __init__(
        self,
        policy,
        create_env,
        benchmark,
        bank,
        protocol,
        directory,
        arm,
        client,
        progress=None,
        text_banks=None,
        recovery=None,
    ):
        self.policy, self.create_env, self.benchmark = policy, create_env, benchmark
        self.bank, self.protocol, self.arm, self.client = bank, protocol, arm, client
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.library = BehaviorLibrary(self.directory / "knowledge", bank.bank_id, arm)
        self.progress = progress or (lambda: None)
        self.reset_audits, self.selected, self.request_index = {}, {}, 0
        self.parity_checked = False
        self.recovery = recovery
        self.conditioner = InputSkillConditioner(policy, bank, text_banks or {})
        self.report = {
            "schema_version": "demo-skill-experiment-1",
            "status": "development",
            "arm": arm,
            "bank_id": bank.bank_id,
            "protocol": protocol,
            "physical_rollouts": [],
            "tasks": {},
            "frozen_policy": True,
        }
        if recovery is not None:
            from .demo_skill_recovery import RecoveryClient

            self.client = RecoveryClient(client, recovery)
            self.report["recovery"] = copy.deepcopy(recovery.receipt)
        self.save()

    def save(self):
        self.report["provider_records"] = copy.deepcopy(self.client.records)
        write_json(self.directory / "summary.json", self.report)
        # Recovered bytes already have an immutable remote archive. Publish the
        # reconstructed state once before fresh work, rather than re-uploading
        # its growing prefix after every historical event.
        if self.recovery is None or not self.recovery.replaying:
            self.progress()

    def run_program(self, entry, program, attempt_id, split):
        if split == "evaluation" and not self.library.data["frozen"]:
            raise ValueError("Evaluation requires a frozen library snapshot")
        path = self.directory / "rollouts" / attempt_id
        if path.exists():
            raise FileExistsError(
                "Physical attempt IDs cannot be replayed or overwritten"
            )
        if self.recovery is not None:
            recovered = self.recovery.rollout(entry, program, attempt_id, split, path)
            if recovered is not None:
                self.reset_audits.setdefault(
                    entry["episode_id"], recovered["reset_audit"]
                )
                self.report["physical_rollouts"].append(
                    {k: v for k, v in recovered.items() if k != "snapshots"}
                )
                self.save()
                return recovered
        recorder = Recorder(path)
        executor = ProgramExecutor(program, self.bank, self.arm)
        env, task, _ = self.create_env(entry["task_id"], entry["seed"])
        if task.language != entry["instruction"]:
            env.close()
            raise ValueError("Reset instruction differs from the environment task")
        spec = ActionSpec.from_environment(
            env, self.policy.horizon, self.policy.action_dim
        )
        adapter = ActionAdapter(
            spec, self.policy.input_transform, self.policy.output_transform
        )
        counters = {
            "velocity_evaluations": 0,
            "intervened_replans": 0,
            "donor_state_replans": 0,
        }
        recorder.event(
            "program_start",
            program=program,
            arm=self.arm,
            entry=entry,
            split=split,
            action_spec=spec,
        )

        def act(live, step):
            choice = executor.select(live, step)
            active = has_effect(choice, self.arm)
            conditioning_receipt = None
            if self.arm == "input_skill_library":
                condition, conditioning_receipt, effective = self.conditioner.prepare(
                    live, entry["instruction"], choice
                )
            else:
                effective = live
                condition = self.policy.prepare(
                    live, digest(live), entry["instruction"]
                )
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [entry["seed"], int(digest(entry["episode_id"])[:8], 16), step, 0]
                )
            )
            noise = self.policy.noise(rng)
            sample = self.policy.sample(condition, noise, **SOLVER)
            counters["velocity_evaluations"] += sample.velocity_evaluations
            actions, clipping = adapter.decode(sample.value, condition.state)
            if not self.parity_checked and not active:
                upstream = self.policy.reference_actions(condition, noise, steps=10)
                decoded = self.policy.output_transform(
                    {"actions": to_numpy(sample.value)[0]}
                )["actions"]
                error = float(np.max(np.abs(np.asarray(upstream) - decoded)))
                if error > 1e-5:
                    raise ValueError("Native LeRobot sampling parity failed")
                recorder.event("native_parity", max_abs=error)
                counters["velocity_evaluations"] += 10
                self.parity_checked = True
            operator = None
            if active and self.arm == "action_composition":
                reference, operator = composed_reference(self.bank, choice, actions)
                encoded = adapter.encode(reference, live)
                encoded[..., 7:] = 0
                inverse = self.policy.invert(
                    condition, self.policy.tensor(encoded), **SOLVER
                )
                steered_noise = to_numpy(inverse.value).copy()
                steered_noise[..., 7:] = to_numpy(noise)[..., 7:]
                generated = self.policy.sample(
                    condition, self.policy.tensor(steered_noise), **SOLVER
                )
                actions, clipping = adapter.decode(generated.value, condition.state)
                counters["velocity_evaluations"] += (
                    inverse.velocity_evaluations + generated.velocity_evaluations
                )
                recorder.event(
                    "demo_frs",
                    reference=reference,
                    encoded=encoded,
                    inverse_noise=steered_noise,
                    generated=generated.value,
                )
            counters["intervened_replans"] += int(active)
            counters["donor_state_replans"] += int(
                active
                and self.arm == "input_skill_library"
                and choice["state_mode"] == "donor"
                and choice["alpha"] > 0
            )
            recorder.event(
                "replan",
                step=step,
                live_observation=live,
                effective_observation=effective,
                choice=choice,
                conditioning_receipt=conditioning_receipt,
                operator=operator,
                actions=actions,
                noise=noise,
                clipping=clipping,
            )
            return actions

        result = run_rollout(
            env,
            entry,
            self.benchmark,
            act,
            execute_steps=5,
            action_budget=300,
            expected_reset=self.reset_audits.get(entry["episode_id"]),
            policy_image_size=224,
            video_path=path / "rollout.mp4",
        )
        self.reset_audits.setdefault(entry["episode_id"], result["reset_audit"])
        result.update(
            attempt_id=attempt_id,
            split=split,
            program_sha256=digest(program),
            suite=entry["suite"],
            task_id=entry["task_id"],
            initial_state_id=entry["initial_state_id"],
            **counters,
        )
        recorder.event("rollout_result", result=result)
        trace = {
            str(p.relative_to(path)): file_sha256(p)
            for p in sorted(path.rglob("*"))
            if p.is_file()
        }
        result["trace_sha256"] = digest(trace)
        write_json(
            path / "trace_manifest.json",
            {"sha256": result["trace_sha256"], "files": trace},
        )
        summary = {key: value for key, value in result.items() if key != "snapshots"}
        write_json(path / "summary.json", summary)
        self.report["physical_rollouts"].append(summary)
        self.save()
        return result

    def propose(self, entry, initial, history, role, selected=()):
        self.request_index += 1
        attempt_id = f"{entry['suite']}_{entry['task_id']}_request{self.request_index}"
        if self.recovery is not None:
            attempt_id = self.recovery.attempt_name(self.request_index, attempt_id)
        request = build_request(
            role=role,
            arm=self.arm,
            task=entry["instruction"],
            episode_id=entry["episode_id"],
            attempt_id=attempt_id,
            request_index=self.request_index,
            bank=self.bank,
            initial_snapshot=initial,
            history=history,
            library=self.library.retrieve(entry["instruction"]),
            selected_sources=selected,
            text_bank_sources=list(self.conditioner.text_banks),
        )
        if self.recovery is not None:
            request = self.recovery.bind_request(request)
        write_json(
            self.directory / "requests" / f"{self.request_index:05d}.json", request
        )
        try:
            proposal = self.client.propose(request)
        except Exception:
            self.report["status"] = "provider_or_contract_error"
            self.save()
            raise  # Never substitute another model or call a failed response success.
        write_json(
            self.directory / "decisions" / f"{self.request_index:05d}.json", proposal
        )
        self.save()
        return proposal

    def develop(self, entries):
        """Each call handles one task; retained records transfer to later tasks."""
        provider_start = len(self.client.records)
        by_state = {e["initial_state_id"]: e for e in entries}
        dev = [by_state[i] for i in self.protocol["development_states"]]
        val = [by_state[i] for i in self.protocol["validation_states"]]
        key = f"{dev[0]['suite']}_task{dev[0]['task_id']}"
        baseline = [
            self.run_program(e, NATIVE, f"{key}_native_dev{i}", "development")
            for i, e in enumerate(dev)
        ]
        best_program, best_score, best_evidence = (
            copy.deepcopy(NATIVE),
            sum(r["success"] for r in baseline),
            [],
        )
        archive = [
            {"score": best_score, "program": copy.deepcopy(NATIVE), "results": baseline}
        ]
        task_report = {
            "native_development_successes": best_score,
            "candidates": [],
            "provider_record_start": provider_start,
            "first_success_candidate": 0 if best_score else None,
            "first_success_provider_record_end": provider_start if best_score else None,
        }
        self.report["tasks"][key] = task_report
        if best_score < len(dev):
            for round_index in range(self.protocol["rounds"]):
                top = sorted(archive, key=lambda row: row["score"], reverse=True)[:3]
                history = [archive_history(row) for row in top]
                sources = self.propose(
                    dev[0], baseline[0]["snapshots"][0], history, "select_sources"
                )["selected_sources"]
                for candidate_index in range(self.protocol["population"]):
                    # Include the latest failed candidate as well as the best, to diversify repairs.
                    recent = top[:2] + ([archive[-1]] if len(archive) > 1 else [])
                    history = [archive_history(row) for row in recent]
                    proposal = self.propose(
                        dev[0], baseline[0]["snapshots"][0], history, "program", sources
                    )
                    program = proposal["program"]
                    name = f"{key}_round{round_index}_candidate{candidate_index}"
                    outcomes = [
                        self.run_program(e, program, f"{name}_dev{i}", "development")
                        for i, e in enumerate(dev)
                    ]
                    score = sum(r["success"] for r in outcomes)
                    archive.append(
                        {"score": score, "program": program, "results": outcomes}
                    )
                    evidence = [evidence_for(r) for r in outcomes]
                    row = {
                        "round": round_index,
                        "candidate": candidate_index,
                        "program": program,
                        "development_successes": score,
                        "provider_record_end": len(self.client.records),
                        "failure_hypothesis": proposal["failure_hypothesis"],
                        "expected_effect": proposal["expected_effect"],
                    }
                    task_report["candidates"].append(row)
                    if (
                        any(r["success"] for r in outcomes)
                        and task_report["first_success_candidate"] is None
                    ):
                        task_report["first_success_candidate"] = len(
                            task_report["candidates"]
                        )
                        task_report["first_success_provider_record_end"] = len(
                            self.client.records
                        )
                    if score > best_score:
                        best_program, best_score, best_evidence = (
                            copy.deepcopy(program),
                            score,
                            evidence,
                        )
                    self.library.record(
                        task=dev[0]["instruction"],
                        program=program,
                        hypothesis=proposal["failure_hypothesis"],
                        evidence=evidence,
                    )
                    self.save()
                if best_score == len(dev):
                    break
        validation = [
            self.run_program(e, best_program, f"{key}_selected_val{i}", "validation")
            for i, e in enumerate(val)
        ]
        if best_evidence:
            self.library.record(
                task=dev[0]["instruction"],
                program=best_program,
                hypothesis="Selected by actual development success; validation on separate resets.",
                evidence=best_evidence + [evidence_for(r) for r in validation],
            )
        self.selected[key] = copy.deepcopy(best_program)
        task_report.update(
            selected_program=best_program,
            selected_development_successes=best_score,
            validation_successes=sum(r["success"] for r in validation),
            budget_exhausted=bool(best_score < len(dev)),
            provider_record_end=len(self.client.records),
            first_success_censored=task_report["first_success_candidate"] is None,
        )
        self.save()

    def freeze(self):
        self.report["library_snapshot_sha256"] = self.library.freeze()
        self.report["selected_programs_sha256"] = digest(self.selected)
        write_json(self.directory / "selected_programs.json", self.selected)
        self.report["status"] = "evaluation"
        self.save()

    def evaluate(self, entries):
        if digest(self.selected) != self.report.get("selected_programs_sha256"):
            raise ValueError("Evaluation programs changed after freezing")
        calls_before = len(self.client.records)
        for entry in entries:
            key = f"{entry['suite']}_task{entry['task_id']}"
            if entry["initial_state_id"] not in self.protocol["evaluation_states"]:
                raise ValueError("Evaluation attempted a development/validation reset")
            suffix = f"{key}_eval{entry['initial_state_id']}"
            self.run_program(entry, NATIVE, suffix + "_native", "evaluation")
            self.run_program(
                entry, self.selected[key], suffix + "_composed", "evaluation"
            )
        if (
            len(self.client.records) != calls_before
            or digest(self.library.data) != self.report["library_snapshot_sha256"]
        ):
            raise ValueError("Evaluation changed the library or invoked Astra")
        self.save()

    def complete(self):
        if set(self.selected) != set(
            self.report.get("planned_task_keys", self.selected)
        ):
            raise ValueError("Selected programs do not cover the declared task scope")
        evaluated = {
            (
                r["suite"],
                r["task_id"],
                r["initial_state_id"],
                r["attempt_id"].endswith("_native"),
            )
            for r in self.report["physical_rollouts"]
            if r["split"] == "evaluation"
        }
        expected = {
            (key.rsplit("_task", 1)[0], int(key.rsplit("_task", 1)[1]), reset, native)
            for key in self.selected
            for reset in self.protocol["evaluation_states"]
            for native in (True, False)
        }
        if not expected or evaluated != expected:
            raise ValueError(
                "Cannot complete a study with missing or extra paired evaluation rollouts"
            )
        if self.recovery is not None:
            self.recovery.exhausted()
        self.report["status"] = "complete"
        self.save()
