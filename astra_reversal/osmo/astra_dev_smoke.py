"""One genuine Stage-1 closed-loop LIBERO-10 development interface smoke.

Uses the OOD driver's unchanged Astra client/configuration and selected numerical
gates. This runs exactly task 0 / prescribed state 0, not an OOD test or a success
rate experiment. No controller, prompt, solver, or fallback behavior is changed.
"""

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from astra_reversal.config import BenchmarkConfig, RunConfig
from astra_reversal.evaluate import read_episodes, read_events
from astra_reversal.libero_runner import load_task_manifest
from astra_reversal.osmo.experiment import RESULTS, ROOT, Archive
from astra_reversal.osmo.ood_steering import (
    ENDPOINT,
    MODEL,
    aggregate_costs,
    copy_frozen,
    make_config,
    metrics_pass,
    require,
    select_solver,
    terminate,
    verify_assets,
)
from astra_reversal.osmo.runtime_probe import SOLVER_OPTIONS, TOLERANCES, write_json
from astra_reversal.records import digest, file_sha256

LIBERO = "astra_reversal/.deps/libero"
EPISODE_ID = "libero_10:task0:state0"
COVERAGE = {
    "minimum_accepted_plans": 4,
    "minimum_replans": 3,
    "minimum_fresh_observation_reuses": 3,
}


def build_smoke_config(base, steps):
    require(steps == 100, "This smoke uses the already selected cubic RK4/100 gate")
    config = make_config(base, "libero_10", "reversal", steps)
    config["benchmark"] = asdict(BenchmarkConfig.preset("libero_10"))
    config["evaluation"]["split"] = "development"
    require(
        config["evaluation"]["save_rollout_videos"] is True,
        "Keep the ordinary rollout video",
    )
    truncated = json.loads(json.dumps(config))
    truncated["benchmark"]["task_action_budget"] = 80
    decision = {"requested_development_budget": 80}
    try:
        RunConfig.from_dict(truncated)
    except ValueError as exc:
        decision.update(truncated_budget_permitted=False, validator_reason=str(exc))
    else:
        config = truncated
        decision.update(truncated_budget_permitted=True, validator_reason=None)
    config = asdict(RunConfig.from_dict(config))
    decision.update(
        actual_action_budget=config["benchmark"]["task_action_budget"],
        scope="Truncated interface smoke; no benchmark success-rate claim"
        if decision["truncated_budget_permitted"]
        else "One full-budget standard LIBERO-10 development episode; no benchmark success-rate claim",
    )
    return config, decision


def progress_snapshot(path):
    progress = {
        "accepted_plans": 0,
        "rejected_responses": 0,
        "executed_actions": 0,
        "inversions": 0,
        "generations": 0,
        "fallbacks": 0,
    }
    if not path.exists():
        return progress
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # A concurrently appended final line is read on the next poll.
        kind = event["kind"]
        if kind == "agent_response":
            progress[
                "accepted_plans" if event["accepted"] else "rejected_responses"
            ] += 1
        elif kind == "control_step":
            progress["executed_actions"] += 1
        elif kind == "inversion":
            progress["inversions"] += 1
        elif kind == "flow" and event["role"] == "generation":
            progress["generations"] += 1
        elif kind == "fallback":
            progress["fallbacks"] += 1
    return progress


def verify_live_proposals(directory, api_log):
    """Bind every accepted controller proposal to its actual provider response."""
    providers = [
        json.loads(line) for line in api_log.read_text().splitlines() if line.strip()
    ]
    bindings = []
    for event in read_events(directory):
        if event["kind"] != "agent_response" or event["accepted"] is not True:
            continue
        proposal = event["proposal"]
        matches = [
            row
            for row in providers
            if row["accepted"] is True
            and row["request_fingerprint"] == proposal["request_fingerprint"]
        ]
        require(
            len(matches) == 1,
            "Accepted controller proposal lacks a unique live provider response",
        )
        provider = matches[0]
        choices = provider["response"]["choices"]
        require(
            provider["http_status"] == 200
            and provider["stage"] == 1
            and provider["requested_model"]
            == provider["response"]["model"]
            == proposal["model_version"]
            == MODEL
            and len(choices) == 1
            and choices[0]["finish_reason"] == "stop"
            and choices[0]["message"]["content"].strip()
            == proposal["raw_response"].strip(),
            "Recorded Stage-1 proposal does not match genuine configured-model output",
        )
        bindings.append(
            {
                "plan_id": proposal["plan_id"],
                "observation_step": proposal["observation_step"],
                "request_fingerprint": proposal["request_fingerprint"],
                "provider_response_id": provider["response"]["id"],
                "actual_model": provider["response"]["model"],
            }
        )
    return {
        "provider_log_sha256": file_sha256(api_log),
        "recorded_provider_calls": len(providers),
        "accepted_controller_proposals_verified": len(bindings),
        "bindings": bindings,
    }


def main():
    import torch
    from lerobot.policies.pi05 import modeling_pi05  # noqa: F401

    require(
        torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(0),
        "Development smoke requires exactly one allocated L40S",
    )
    require(
        bool(os.environ.get("NVIDIA_INFERENCE_API_KEY")),
        "Missing injected NVIDIA inference credential",
    )
    RESULTS.mkdir(parents=True, exist_ok=False)
    archive = Archive()
    processes, streams = [], []
    run_directory = RESULTS / "reversal"
    api_log = RESULTS / "astra_provider.jsonl"
    status = {
        "status": "initializing",
        "scope": "One genuine closed-loop development interface smoke; no OOD or benchmark improvement claim",
        "workflow": os.environ["ASTRA_RUN_ID"],
        "artifact_prefix": archive.prefix,
    }
    exit_code = 1

    def command(label, argv):
        stream = (RESULTS / f"{label}.log").open("w")
        streams.append(stream)
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="0",
            MUJOCO_EGL_DEVICE_ID="0",
            NVIDIA_TF32_OVERRIDE="0",
        )
        write_json(
            RESULTS / f"{label}.command.json",
            {
                "argv": argv,
                "CUDA_VISIBLE_DEVICES": "0",
                "MUJOCO_EGL_DEVICE_ID": "0",
                "tf32": False,
            },
        )
        process = subprocess.Popen(
            argv,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append(process)
        started = time.perf_counter()
        while True:
            progress = {
                "phase": label,
                "elapsed_seconds": time.perf_counter() - started,
                **progress_snapshot(run_directory / "events.jsonl"),
            }
            write_json(RESULTS / "progress.json", progress)
            print(json.dumps(progress), flush=True)
            archive.sync()
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                continue
            stream.flush()
            archive.sync()
            require(
                code == 0, f"{label} failed with exit code {code}; see its archived log"
            )
            return

    try:
        report = json.loads(
            (
                ROOT / "astra_reversal/.deps/ood-inputs/runtime_diagnostics.json"
            ).read_text()
        )
        steps = select_solver(report)
        base = json.loads(
            (
                ROOT / "astra_reversal/configs/pi05_libero_openpi_inputs_baseline.json"
            ).read_text()
        )
        config, budget = build_smoke_config(base, steps)
        frozen, hashes = RESULTS / "frozen", {}
        diagnostic = frozen / "runtime_diagnostics.json"
        copy_frozen(
            ROOT / "astra_reversal/.deps/ood-inputs/runtime_diagnostics.json",
            diagnostic,
            hashes,
        )
        for name in ("request.json", "response.json", "provider.jsonl"):
            copy_frozen(
                ROOT / "astra_reversal/.deps/astra-proposal-input" / name,
                frozen / "astra-proposal-input" / name,
                hashes,
            )
        for path in sorted((ROOT / "astra_reversal").glob("*.py")) + sorted(
            (ROOT / "astra_reversal/osmo").glob("*.py")
        ):
            copy_frozen(path, frozen / "source" / path.relative_to(ROOT), hashes)
        copy_frozen(
            Path(__file__).with_name("astra_dev_smoke_l40s.yaml"),
            frozen / "workflow.yaml",
            hashes,
        )
        config_path = frozen / "development_config.json"
        write_json(config_path, config)
        hashes[str(config_path.relative_to(RESULTS))] = file_sha256(config_path)
        assets = verify_assets(config, report)
        require(
            torch.__version__ == report["checkpoint"]["torch_version"],
            "Torch runtime differs from the selected numerical gate",
        )
        write_json(
            RESULTS / "runtime.json",
            {
                "workflow": status["workflow"],
                "payload_sha256": os.environ["PAYLOAD_SHA256"],
                "gpu": torch.cuda.get_device_name(0),
                "visible_gpu_count": 1,
                "tf32": False,
                "selected_solver": report["selected_solver"],
                "verified_asset_sha256": assets,
                "astra_model": MODEL,
                "astra_endpoint": ENDPOINT,
                "astra_cache": {"no-cache": True},
                "budget_decision": budget,
            },
        )
        status.update(status="preflight", budget_decision=budget)
        write_json(RESULTS / "status.json", status)

        manifest_path = frozen / "development_manifest.json"
        command(
            "manifest",
            [
                sys.executable,
                "-m",
                "astra_reversal",
                "make-manifest",
                "--config",
                str(config_path),
                "--libero-root",
                LIBERO,
                "--tasks",
                "0",
                "--trials",
                "1",
                "--output",
                str(manifest_path),
            ],
        )
        manifest = load_task_manifest(manifest_path, RunConfig.from_dict(config))
        entries = manifest["episodes"]
        request = json.loads((frozen / "astra-proposal-input/request.json").read_text())
        require(
            len(entries) == 1,
            "Development smoke must contain exactly one prescribed episode",
        )
        entry = entries[0]
        require(
            entry["episode_id"] == EPISODE_ID
            and entry["suite"] == "libero_10"
            and entry["task_id"] == entry["initial_state_id"] == 0
            and entry["seed"] == config["seed"] == 7
            and entry["instruction"] == request["task_instruction"]
            and entry["initially_successful"] is False
            and manifest["split"] == "development"
            and manifest["benchmark"]["reset_source"] == "prescribed_initial_states",
            "Frozen smoke manifest is not the intended standard LIBERO-10 task-0/state-0 episode",
        )
        hashes[str(manifest_path.relative_to(RESULTS))] = file_sha256(manifest_path)
        plan = {
            "episode_id": EPISODE_ID,
            "split": "development",
            "seed": 7,
            "stage": 1,
            "method": "reversal",
            "selected_solver": report["selected_solver"],
            "budget_decision": budget,
            "required_interface_coverage": COVERAGE,
            "task_manifest_sha256": manifest["sha256"],
            "frozen_file_sha256": hashes,
            "scope": status["scope"],
            "selection_note": "No configuration, prompt, solver or outcome-based retries are selected from this smoke",
        }
        plan["plan_sha256"] = digest(plan)
        write_json(RESULTS / "frozen_plan.json", plan)

        proposal_path = RESULTS / "astra_proposal_diagnostics.json"
        command(
            "proposal_preflight",
            [
                sys.executable,
                "-m",
                "astra_reversal.osmo.proposal_probe",
                "--runtime-diagnostics",
                str(diagnostic),
                "--proposal-input",
                str(frozen / "astra-proposal-input"),
                "--device",
                "cuda",
                "--output",
                str(proposal_path),
            ],
        )
        proposal = json.loads(proposal_path.read_text())
        require(
            proposal["status"] == "passed"
            and proposal["actual_model"] == MODEL
            and proposal["checkpoint"] == report["checkpoint"]
            and proposal["action_spec_id"] == report["action_spec_id"]
            and proposal["tolerances"] == TOLERANCES
            and proposal["solver"] == "rk4"
            and proposal["solver_options"] == SOLVER_OPTIONS
            and proposal["actual_controller_proposal"]["passed"] is True
            and 0
            <= proposal["actual_controller_proposal"]["reconstruction"]["max_abs"]
            <= TOLERANCES["action_atol"]
            and any(
                row["steps"] == steps and row["passed"] is True and metrics_pass(row)
                for row in proposal["results"]
            ),
            "The genuine development proposal failed unchanged numerical preflight",
        )
        write_json(
            RESULTS / "accepted_gates.json",
            {
                "runtime_diagnostics_sha256": file_sha256(diagnostic),
                "astra_proposal_diagnostics_sha256": file_sha256(proposal_path),
                "selected_steps": steps,
                "tolerances": TOLERANCES,
            },
        )
        agent = [
            sys.executable,
            "-m",
            "astra_reversal.astra_client",
            "--endpoint",
            ENDPOINT,
            "--model",
            MODEL,
            "--response-log",
            str(api_log),
            "--reasoning-effort",
            "low",
            "--max-completion-tokens",
            "8192",
            "--timeout",
            "170",
        ]
        status["status"] = "running_development_episode"
        write_json(RESULTS / "status.json", status)
        command(
            "closed_loop",
            [
                sys.executable,
                "-m",
                "astra_reversal.osmo.ood_steering",
                "--worker",
                "run",
                "--config",
                str(config_path),
                "--manifest",
                str(manifest_path),
                "--libero-root",
                LIBERO,
                "--diagnostics",
                str(diagnostic),
                "--agent-command",
                shlex.join(agent),
                "--output",
                str(run_directory),
            ],
        )
        episodes = read_episodes(run_directory)
        require(
            len(episodes) == 1 and episodes[0]["episode_id"] == EPISODE_ID,
            "Smoke produced incomplete or unintended episode coverage",
        )
        episode = episodes[0]
        recorded_manifest = json.loads((run_directory / "manifest.json").read_text())
        require(
            recorded_manifest["checkpoint"] == report["checkpoint"]
            and recorded_manifest["task_manifest_sha256"] == manifest["sha256"]
            and recorded_manifest["config"] == config,
            "Executed smoke provenance differs from its frozen plan",
        )

        audit_path = RESULTS / "runtime_inversion_audit.json"
        command(
            "runtime_inversion_audit",
            [
                sys.executable,
                "-m",
                "astra_reversal.audit_astra_inversions",
                str(run_directory),
                "--output",
                str(audit_path),
            ],
        )
        audit = json.loads(audit_path.read_text())
        live = verify_live_proposals(run_directory, api_log)
        write_json(RESULTS / "live_proposal_provenance.json", live)
        costs = aggregate_costs(
            [{"output": str(run_directory), "api_log": str(api_log)}]
        )
        count = audit["counts"]
        accepted_steps = sorted({item["observation_step"] for item in live["bindings"]})
        checks = {
            "completed_intended_episode_without_execution_error": episode.get("failure")
            is None,
            "several_genuine_replans": live["accepted_controller_proposals_verified"]
            >= COVERAGE["minimum_accepted_plans"]
            and len(accepted_steps[1:]) >= COVERAGE["minimum_replans"],
            "all_accepted_proposals_have_recorded_inversions": count[
                "accepted_proposals_without_inversion"
            ]
            == 0
            and count["astra_inversions"]
            == live["accepted_controller_proposals_verified"],
            "same_condition_roundtrips_pass": audit["all_roundtrips_passed"],
            "recovered_latents_reused_exactly": audit["all_latents_match_recovered"]
            and count["subsequent_generations"]
            >= COVERAGE["minimum_fresh_observation_reuses"],
            "fresh_observations_used_during_reuse": count[
                "subsequent_steps_not_advanced"
            ]
            == 0
            and count["subsequent_observation_ids_changed"]
            >= COVERAGE["minimum_fresh_observation_reuses"],
            "no_hidden_policy_reference_proposals": costs["counts"].get(
                "reference_generation_solves", 0
            )
            == costs["counts"].get("reference_inversion_solves", 0)
            == 0,
            "no_fallback_actions": costs["counts"]["fallback_actions"] == 0,
            "configured_model_observed": set(costs["actual_api_models"]) == {MODEL},
        }
        passed = all(checks.values())
        summary = {
            "status": "complete_passed" if passed else "complete_with_issues",
            "scope": status["scope"],
            "budget_decision": budget,
            "interface_checks": checks,
            "required_interface_coverage": COVERAGE,
            "episode_outcome": episode,
            "costs": costs,
            "audit_counts": count,
            "maximum_recorded_roundtrip_error": audit["maximum_full_internal_error"],
            "actual_model": MODEL,
            "accepted_proposal_steps": accepted_steps,
            "live_proposals_sha256": file_sha256(
                RESULTS / "live_proposal_provenance.json"
            ),
            "runtime_inversion_audit_sha256": file_sha256(audit_path),
            "task_manifest_sha256": manifest["sha256"],
            "benchmark_success_rate_claimed": False,
            "ood_episodes_run": 0,
            "artifact_prefix": archive.prefix,
        }
        write_json(RESULTS / "development_smoke.json", summary)
        status.update(status=summary["status"], interface_checks=checks)
        exit_code = 0 if passed else 2
    except BaseException as exc:
        terminate(processes)
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        exit_code = 1
    finally:
        for stream in streams:
            stream.close()
        status["exit_code"] = exit_code
        write_json(RESULTS / "status.json", status)
        archive.sync(include_arrays=True)
        # bootstrap.sh already honors this marker. Retain artifacts and return
        # the failure code immediately instead of occupying a GPU in its debug wait.
        if exit_code:
            Path("/tmp/astra-recovery-exit.status").write_text(str(exit_code) + "\n")
    print(json.dumps(status), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
