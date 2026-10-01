"""Two owned L40S workers: demo-action composition and reusable input skills."""

import json
import os
import shutil
from pathlib import Path

from astra_reversal.codex_relay import CodexRelayClient, ensure_server
from astra_reversal.config import BenchmarkConfig
from astra_reversal.demo_segments import DemoBank, build_bank, write_json
from astra_reversal.demo_skill_experiment import DemoSkillExperiment, load_protocol
from astra_reversal.demo_skill_metrics import write_metrics
from astra_reversal.demo_skill_recovery import RecoveryArchive
from astra_reversal.intervention_rollout import capture_reset_manifest
from astra_reversal.libero_runner import configure_libero
from astra_reversal.osmo.demo_skill_recovery import fetch_archive
from astra_reversal.osmo.experiment import RESULTS, ROOT
from astra_reversal.osmo.frs_policy_improvement import frozen_parameter_receipt
from astra_reversal.osmo.interpolation import (
    load_banks,
    load_frozen_policy,
    native_preflight,
)
from astra_reversal.osmo.ood_distributed import WorkerArchive
from astra_reversal.records import digest, file_sha256


def main():
    import torch

    worker = int(os.environ["ASTRA_WORKER_INDEX"])
    protocol = load_protocol()
    if worker not in (0, 1):
        raise ValueError("This study has exactly two separate-arm workers")
    if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
        raise RuntimeError("An allocated OSMO L40S is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    RESULTS.mkdir(parents=True, exist_ok=False)
    archive = WorkerArchive(worker)
    pilot = os.environ.get("ASTRA_DEMO_PILOT", "0") == "1"
    arm = protocol["arms"][worker]
    write_json(
        RESULTS / "runtime.json",
        {
            "workflow": os.environ["ASTRA_RUN_ID"],
            "worker": worker,
            "arm": arm,
            "gpu": torch.cuda.get_device_name(0),
            "payload_sha256": os.environ["PAYLOAD_SHA256"],
            "pilot": pilot,
            "scope": "two-task cached-source pilot"
            if pilot
            else "all20 OOD tasks, all40 standard demo sources",
        },
    )
    ensure_server()
    archive.sync()
    try:
        native_preflight(archive)
        prior = None
        recovery_manifest = None
        if os.environ.get("ASTRA_DEMO_RECOVERY_FILE"):
            source = Path(os.environ["ASTRA_DEMO_RECOVERY_FILE"])
            if file_sha256(source) != os.environ["ASTRA_DEMO_RECOVERY_SHA256"]:
                raise ValueError("Recovery manifest bytes changed after submission")
            recovery_manifest = json.loads(source.read_text())
            prior = fetch_archive(
                archive.client,
                recovery_manifest,
                worker,
                ROOT / "astra_reversal/.deps/demo-recovery",
            )
            if json.loads((prior / "runtime.json").read_text())["pilot"] != pilot:
                raise ValueError("Cannot change pilot/full scope during recovery")
            write_json(RESULTS / "recovery_manifest.json", recovery_manifest)
        root = ROOT / "astra_reversal/.deps/demo-skill-inputs/source_cache"
        bank_path = RESULTS / "demo_bank"
        if prior is None:
            build_bank(root, bank_path, cached_only=pilot)
        else:
            shutil.copytree(prior / "demo_bank", bank_path)
        bank = DemoBank(bank_path)
        if not pilot and not bank.metadata["complete_standard_catalog"]:
            raise RuntimeError("The full study cannot silently use a partial demo bank")
        text_banks = (
            load_banks(
                archive,
                ROOT / "astra_reversal/.deps/interpolation-inputs/bank_inventory.json",
            )
            if arm == "input_skill_library"
            else {}
        )
        libero_root = (
            ROOT / "astra_reversal/.deps/libero-ood/third_party/modified_libero"
        )
        manifests = {}
        reset_indexes = (
            protocol["development_states"]
            + protocol["validation_states"]
            + protocol["evaluation_states"]
        )
        for suite in protocol["suites"]:
            task_ids = (
                ([1] if suite == "libero_goal_ood" else [4])
                if pilot
                else protocol["task_ids"]
            )
            benchmark = BenchmarkConfig.preset(suite)
            manifests[suite] = capture_reset_manifest(
                libero_root,
                benchmark,
                seed=protocol["seed"],
                cases=[(task, reset) for task in task_ids for reset in reset_indexes],
                output=RESULTS / f"{suite}_resets.json",
                split="followup_adaptation",
            )
            if prior is not None:
                old = json.loads((prior / f"{suite}_resets.json").read_text())
                if old["episodes"] != manifests[suite]["episodes"]:
                    raise ValueError("Recovered environment reset manifest changed")
        policy = load_frozen_policy()
        before = frozen_parameter_receipt(policy)
        write_json(RESULTS / "frozen_weights_before.json", before)
        recovery = None
        if prior is not None:
            if json.loads((prior / "frozen_weights_before.json").read_text()) != before:
                raise ValueError("Recovered policy parameter bytes changed")
            recovery = RecoveryArchive(
                prior,
                arm=arm,
                bank_id=bank.bank_id,
                protocol=protocol,
                namespace=os.environ["ASTRA_RUN_ID"],
            )
            recovery.receipt["source_archive"] = recovery_manifest["workers"][worker]
            write_json(RESULTS / "recovery.json", recovery.receipt)
        client = CodexRelayClient(
            family="demo_skills",
            response_log=RESULTS / "provider.jsonl",
            model=protocol["astra"]["model"],
            reasoning_effort=protocol["astra"]["reasoning_effort"],
            timeout=protocol["astra"]["timeout_seconds"],
            max_completion_tokens=None,
        )
        first_suite = protocol["suites"][0]
        benchmark = BenchmarkConfig.preset(first_suite)
        _, create = configure_libero(libero_root, benchmark, RESULTS / "libero_runtime")
        experiment = DemoSkillExperiment(
            policy,
            create,
            benchmark,
            bank,
            protocol,
            RESULTS / "experiment",
            arm,
            client,
            progress=archive.sync,
            text_banks=text_banks,
            recovery=recovery,
        )
        experiment.report.update(
            pilot=pilot,
            planned_task_keys=sorted(
                {
                    f"{suite}_task{entry['task_id']}"
                    for suite, manifest in manifests.items()
                    for entry in manifest["episodes"]
                }
            ),
        )
        experiment.save()
        # Accumulate experience across both suites before freezing any evaluation.
        for suite, manifest in manifests.items():
            experiment.benchmark = BenchmarkConfig.preset(suite)
            _, experiment.create_env = configure_libero(
                libero_root, experiment.benchmark, RESULTS / "libero_runtime"
            )
            for task in sorted({e["task_id"] for e in manifest["episodes"]}):
                experiment.develop(
                    [e for e in manifest["episodes"] if e["task_id"] == task]
                )
        experiment.freeze()
        for suite, manifest in manifests.items():
            experiment.benchmark = BenchmarkConfig.preset(suite)
            _, experiment.create_env = configure_libero(
                libero_root, experiment.benchmark, RESULTS / "libero_runtime"
            )
            experiment.evaluate(
                [
                    e
                    for e in manifest["episodes"]
                    if e["initial_state_id"] in protocol["evaluation_states"]
                ]
            )
        after = frozen_parameter_receipt(policy)
        if before != after:
            raise RuntimeError("Frozen pi0.5 weights changed")
        experiment.complete()
        write_metrics(experiment.report, RESULTS / "experiment")
        write_json(
            RESULTS / "completion.json",
            {
                "status": "complete",
                "pilot": pilot,
                "arm": arm,
                "physical_rollouts": len(experiment.report["physical_rollouts"]),
                "library_sha256": digest(experiment.library.data),
                "frozen_weights": after,
            },
        )
    except Exception as exc:
        write_json(
            RESULTS / "failure.json",
            {
                "status": "incomplete",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        archive.sync()
        raise
    finally:
        archive.sync(include_arrays=True)


if __name__ == "__main__":
    main()
