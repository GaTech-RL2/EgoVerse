"""Bounded six-episode OSMO L40S interface/latency commissioning pilot."""

import json
import os
from dataclasses import asdict

from astra_reversal.config import BenchmarkConfig
from astra_reversal.demo_segments import DemoBank, build_bank, write_json
from astra_reversal.intervention_rollout import capture_reset_manifest
from astra_reversal.libero_runner import configure_libero
from astra_reversal.meta_harness.evaluate import (
    evaluate_episode,
    metrics,
    paired_difference,
)
from astra_reversal.meta_harness.harness import Harness
from astra_reversal.meta_harness.preflight import policy_gate
from astra_reversal.meta_harness.relay_agent import ProfileRelayWorker
from astra_reversal.meta_harness.schema import Compiler, Limits, make_cards
from astra_reversal.osmo.experiment import RESULTS, ROOT
from astra_reversal.osmo.frs_policy_improvement import frozen_parameter_receipt
from astra_reversal.osmo.interpolation import load_frozen_policy, native_preflight
from astra_reversal.osmo.ood_distributed import WorkerArchive
from astra_reversal.records import file_sha256


def main():
    import torch

    from astra_reversal.codex_relay import ensure_server

    if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
        raise RuntimeError("The pilot requires one allocated OSMO L40S")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    RESULTS.mkdir(parents=True, exist_ok=False)
    archive = WorkerArchive(0)
    protocol_path = ROOT / "astra_reversal/configs/meta_harness_v1.json"
    protocol = json.loads(protocol_path.read_text())
    limits = Limits(**protocol["limits"])
    runtime = {
        "protocol_sha256": file_sha256(protocol_path),
        "limits": asdict(limits),
        "source_revision": os.environ["ASTRA_SOURCE_REVISION"],
        "payload_sha256": os.environ["PAYLOAD_SHA256"],
        "workflow": os.environ["ASTRA_RUN_ID"],
        "gpu": torch.cuda.get_device_name(0),
        "scope": "interface and latency pilot; NOT main comparison",
        "token_caps_enforced": False,
        "main_launch_allowed": False,
    }
    write_json(RESULTS / "manifest.json", runtime)
    ensure_server()
    archive.sync()
    try:
        native_preflight(archive)
        build_bank(
            ROOT / "astra_reversal/.deps/demo-skill-inputs/source_cache",
            RESULTS / "demo_bank",
            cached_only=True,
        )
        bank = DemoBank(RESULTS / "demo_bank")
        cards = make_cards(bank)
        compiler = Compiler(bank, cards)
        write_json(
            RESULTS / "cards.json",
            {
                "cards": cards,
                "manifest_sha256": compiler.identity,
                "complete_standard_catalog": bank.metadata["complete_standard_catalog"],
            },
        )
        policy = load_frozen_policy()
        before = frozen_parameter_receipt(policy)
        write_json(RESULTS / "frozen_weights_before.json", before)
        gate = policy_gate(policy, bank)
        write_json(RESULTS / "weighted_interface_gate.json", gate)
        archive.sync()
        if gate["status"] != "passed":
            raise ValueError("Frozen-checkpoint policy interface gate failed")
        initial = (ROOT / "astra_reversal/meta_harness/initial_harness.py").read_text()
        results = {arm: [] for arm in protocol["pilot"]["arms"]}
        libero_root = (
            ROOT / "astra_reversal/.deps/libero-ood/third_party/modified_libero"
        )
        for suite, task in protocol["pilot"]["tasks"]:
            benchmark = BenchmarkConfig.preset(suite)
            manifest = capture_reset_manifest(
                libero_root,
                benchmark,
                seed=protocol["pilot"]["seed"],
                cases=[(task, reset) for reset in protocol["pilot"]["resets"]],
                output=RESULTS / f"{suite}_resets.json",
                split="development",
            )
            _, create = configure_libero(
                libero_root, benchmark, RESULTS / "libero_runtime"
            )
            for entry in manifest["episodes"]:
                reset_audit = None
                for arm in protocol["pilot"]["arms"]:
                    destination = (
                        RESULTS
                        / "episodes"
                        / arm
                        / f"{suite}_task{task}_state{entry['initial_state_id']}"
                    )
                    worker = (
                        None
                        if arm == "native"
                        else ProfileRelayWorker(RESULTS / "provider.jsonl")
                    )
                    result = evaluate_episode(
                        policy=policy,
                        create_env=create,
                        benchmark=benchmark,
                        entry=entry,
                        compiler=compiler,
                        limits=limits,
                        harness=Harness(initial),
                        worker=worker,
                        directory=destination,
                        mode=arm,
                        expected_reset=reset_audit,
                        video=suite == protocol["pilot"]["tasks"][0][0],
                    )
                    reset_audit = result["reset_audit"]
                    results[arm].append(result)
                    write_json(RESULTS / "partial_results.json", results)
                    archive.sync()
        after = frozen_parameter_receipt(policy)
        write_json(RESULTS / "frozen_weights_after.json", after)
        if before != after:
            raise ValueError("System 1 weights changed during the pilot")
        report = {
            "scope": runtime["scope"],
            "arms": {arm: metrics(rows) for arm, rows in results.items()},
            "paired_async_vs_native": paired_difference(
                results["native"], results["async"]
            ),
            "main_launch_allowed": False,
            "next_gate": "profile tokens/latency; validate the capped transport and frozen model version",
        }
        write_json(RESULTS / "pilot_report.json", report)
    except BaseException as error:
        write_json(
            RESULTS / "failure.json",
            {"error": type(error).__name__, "status": "pilot_failed"},
        )
        raise
    finally:
        archive.sync(include_arrays=True)


if __name__ == "__main__":
    main()
