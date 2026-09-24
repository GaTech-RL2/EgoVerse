"""Validate cubic RK4 on the 14 recorded LIBERO-10 development conditions.

The candidate endpoint is regenerated from each recorded, full [1, 10, 32]
noise tensor at the candidate resolution. Inverting the old RK4/50 endpoint
would confound forward and inverse discretization errors. This probe never
loads an OOD trajectory, calls Astra, or uses control success for selection.

Run ``python -m astra_reversal.osmo.runtime_probe --help`` for worker paths.
``--pack-input DESTINATION`` creates a small, hash-verified input bundle without
loading a model; ``--verify-records-only`` audits that bundle without a GPU.
"""

import argparse
import gc
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from astra_reversal.action_adapter import ActionSpec
from astra_reversal.evaluate import read_events
from astra_reversal.flow import error_metrics
from astra_reversal.records import digest, file_sha256, to_numpy

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TOLERANCES = {"action_atol": 0.02, "noise_atol": 0.1, "parity_atol": 1e-5}
EXPECTED_CONDITIONS = 14
SOLVER_OPTIONS = {"time_power": 3.0}


def write_json(path, value):
    """Replace reports atomically so interruption leaves a readable checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class RecordedArrays:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.inventory = {}

    def read(self, reference):
        path = (self.root / reference["array"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Recorded array path escapes the input directory")
        value = np.load(path, allow_pickle=False)
        if (
            digest(value) != reference["sha256"]
            or list(value.shape) != reference["shape"]
            or str(value.dtype) != reference["dtype"]
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"Recorded array integrity mismatch: {reference['array']}")
        self.inventory[reference["array"]] = {
            **reference,
            "file_sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        return value


def _same(reference, actual, label):
    if not np.array_equal(reference, actual):
        raise ValueError(f"Recorded {label} does not match its reference")


def load_records(directory):
    """Reconstruct only the known-noise inversion conditions, verifying inputs."""
    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    config = manifest["config"]
    if (
        config["benchmark"]["suite"] != "libero_10"
        or config["evaluation"]["split"] != "development"
        or config["method"] != "inversion_only"
        or config["policy"]["input_profile"] != "openpi_libero"
    ):
        raise ValueError(
            "Solver selection requires recorded LIBERO-10 development inversions"
        )
    spec = ActionSpec(**manifest["action_spec"])
    spec_payload = {
        key: manifest["action_spec"][key]
        for key in (
            "horizon",
            "model_action_dim",
            "timestep_seconds",
            "lower",
            "upper",
            "semantics",
        )
    }
    if digest(spec_payload) != spec.action_spec_id:
        raise ValueError("Recorded action specification digest mismatch")
    arrays = RecordedArrays(root)
    episodes, conditions, generations = {}, {}, {}
    rows, pending = [], None
    for event in read_events(root):
        kind = event["kind"]
        if kind == "episode_start":
            episodes[event["episode_id"]] = event
        elif kind == "condition":
            conditions[(event["episode_id"], event["condition_id"])] = event
        elif kind == "flow" and event["role"] == "reference_generation":
            generations[(event["episode_id"], event["condition_id"])] = event
        elif kind == "inversion":
            result = event["result"]
            reference = result.get("reference")
            if not reference or "known_noise" not in reference:
                raise ValueError(
                    "Every source inversion must have actual known reference noise"
                )
            episode_id = event["episode_id"]
            condition_id = result["inverse_condition_id"]
            condition = conditions[(episode_id, condition_id)]
            episode = episodes[episode_id]
            generation = generations[(episode_id, condition_id)]
            if (
                episode["metadata"].get("split") != "development"
                or episode["metadata"].get("suite") != "libero_10"
                or reference["source"] != "frozen_pi05_raw_conditioning"
                or reference["action_space"] != "normalized_model_tensor"
                or reference["source_condition_id"] != condition_id
                or reference["source_observation_id"] != condition["observation_id"]
                or reference["source_prompt"] != condition["prompt"]
                or condition["prompt"] != episode["instruction"]
                or reference["observation_step"] != condition["observation_step"]
                or generation["observation_step"] != condition["observation_step"]
                or reference["checkpoint"] != manifest["checkpoint"]
                or result["checkpoint"] != manifest["checkpoint"]
            ):
                raise ValueError("Recorded inversion conditioning/provenance mismatch")
            observation = {
                key: arrays.read(value)
                for key, value in condition["observation"].items()
            }
            if (
                digest(observation) != condition["observation_id"]
                or digest({**observation, "prompt": condition["prompt"]})
                != condition_id
            ):
                raise ValueError("Recorded observation or condition digest mismatch")
            known = arrays.read(reference["known_noise"])
            endpoint = arrays.read(result["model_actions"])
            recovered = arrays.read(result["noise"])
            expected_shape = (1, spec.horizon, spec.model_action_dim)
            if any(
                x.shape != expected_shape or x.dtype != np.float32
                for x in (known, endpoint, recovered)
            ):
                raise ValueError(
                    "Known noise and endpoints must retain the full float32 model tensor"
                )
            _same(endpoint, arrays.read(reference["model_actions"]), "model endpoint")
            _same(known, arrays.read(generation["input"]), "reference-generation input")
            _same(
                endpoint,
                arrays.read(generation["output"]),
                "reference-generation output",
            )
            noise_key = reference["noise_key"]
            if noise_key != [episode["seed"], condition["observation_step"], 0]:
                raise ValueError("Unexpected recorded reference-noise seed key")
            expected_noise = (
                np.random.default_rng(np.random.SeedSequence(noise_key))
                .standard_normal(expected_shape)
                .astype(np.float32)
            )
            _same(known, expected_noise, "seeded full noise")
            recovery = error_metrics(known, recovered)
            source = {
                "episode_id": episode_id,
                "plan_id": result["plan_id"],
                "observation_step": condition["observation_step"],
                "observation_id": condition["observation_id"],
                "condition_id": condition_id,
                "prompt": condition["prompt"],
                "noise_key": noise_key,
                "condition_sequence": condition["sequence"],
                "inversion_sequence": event["sequence"],
                "reference_generation_sequence": generation["sequence"],
                "known_noise_sha256": digest(known),
                "recorded_endpoint_sha256": digest(endpoint),
                "recorded_solver": result["solver"],
                "recorded_steps": len(result["grid"]) - 1,
                "recorded_known_noise_recovery": recovery,
                "recorded_passed_noise_tolerance": recovery["max_abs"]
                <= TOLERANCES["noise_atol"],
            }
            rows.append(
                {
                    "source": source,
                    "observation": observation,
                    "known_noise": known,
                    "endpoint": endpoint,
                    "recovered": recovered,
                }
            )
            pending = rows[-1]
        elif kind == "flow" and event["role"] == "generation" and pending is not None:
            source = pending["source"]
            if (event["episode_id"], event["condition_id"]) == (
                source["episode_id"],
                source["condition_id"],
            ):
                _same(
                    pending["recovered"],
                    arrays.read(event["input"]),
                    "same-condition replay input",
                )
                source["recorded_internal_reconstruction"] = error_metrics(
                    pending["endpoint"], arrays.read(event["output"])
                )
                source["replay_sequence"] = event["sequence"]
            pending = None
    if (
        len(rows) != EXPECTED_CONDITIONS
        or len({r["source"]["condition_id"] for r in rows}) != EXPECTED_CONDITIONS
    ):
        raise ValueError(
            f"Expected all {EXPECTED_CONDITIONS} distinct recorded inversion conditions"
        )
    if any("recorded_internal_reconstruction" not in row["source"] for row in rows):
        raise ValueError("A recorded inversion is missing its same-condition replay")
    provenance = {
        "source_root": str(root),
        "manifest_sha256": file_sha256(root / "manifest.json"),
        "events_sha256": file_sha256(root / "events.jsonl"),
        "task_manifest_sha256": manifest["task_manifest_sha256"],
        "split": "development",
        "suite": "libero_10",
        "conditions": len(rows),
        "verified_arrays": sorted(arrays.inventory.values(), key=lambda x: x["array"]),
        "recorded_passing_conditions": sum(
            r["source"]["recorded_passed_noise_tolerance"] for r in rows
        ),
        "recorded_maximum_noise_error": max(
            r["source"]["recorded_known_noise_recovery"]["max_abs"] for r in rows
        ),
    }
    bundle_path = root / "bundle.json"
    if bundle_path.exists():
        bundle = json.loads(bundle_path.read_text())
        for key in (
            "manifest_sha256",
            "events_sha256",
            "task_manifest_sha256",
            "conditions",
            "verified_arrays",
        ):
            if bundle[key] != provenance[key]:
                raise ValueError(f"Packed input provenance mismatch: {key}")
        provenance["bundle_sha256"] = file_sha256(bundle_path)
        provenance["original_source_root"] = bundle["source_root"]
    return manifest, rows, provenance


def pack_records(source, destination):
    """Keep original event/manifest bytes and only arrays used by the 14 cases."""
    _, rows, provenance = load_records(source)
    source, destination = Path(source).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("manifest.json", "events.jsonl"):
        shutil.copyfile(source / name, destination / name)
    for item in provenance["verified_arrays"]:
        target = destination / item["array"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / item["array"], target)
    bundle = {
        **provenance,
        "schema_version": "1.0",
        "purpose": "Recorded development-condition solver validation only",
        "complete_event_log": True,
        "array_scope": "Only references read by the 14 known-noise condition audits; unrelated event arrays are omitted",
        "source_conditions": [r["source"] for r in rows],
    }
    write_json(destination / "bundle.json", bundle)
    load_records(destination)
    return bundle


def _check_checkpoint(policy, manifest):
    # Physical path spelling may change in a packaged worker. All weight,
    # tokenizer, input, source-code, and package identities must still match.
    ignored = {"artifact", "requested_artifact", "tokenizer_path", "device"}
    before = {k: v for k, v in manifest["checkpoint"].items() if k not in ignored}
    after = {k: v for k, v in policy.metadata.items() if k not in ignored}
    differences = [
        key
        for key in sorted(before.keys() | after.keys())
        if before.get(key) != after.get(key)
    ]
    if differences:
        raise ValueError(
            f"Loaded policy differs from recorded numerical provenance: {differences}"
        )
    if any(p.requires_grad for p in policy.policy.parameters()):
        raise ValueError("The numerical probe requires frozen checkpoint parameters")
    return {"matched": True, "ignored_physical_location_fields": sorted(ignored)}


def _decoded(policy, condition, value):
    # Match diagnostics.py: compare raw decoded controller inputs before clipping.
    return policy.output_transform(
        {
            "actions": to_numpy(value)[0].copy(),
            "state": to_numpy(condition.state)[0].copy(),
        }
    )["actions"]


def _save_arrays(path, **values):
    values = {key: to_numpy(value).copy() for key, value in values.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)
    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "arrays": {
            name: {
                "sha256": digest(value),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in values.items()
        },
    }


def _parity(policy, condition, noise):
    began = time.perf_counter()
    euler = policy.sample(condition, noise, steps=10, solver="euler")
    native = policy.reference_actions(condition, noise, steps=10)
    metrics = error_metrics(native, _decoded(policy, condition, euler.value))
    return {
        "steps": 10,
        "error": metrics,
        "velocity_evaluations": euler.velocity_evaluations + 10,
        "latency_seconds": time.perf_counter() - began,
        "passed": metrics["max_abs"] <= TOLERANCES["parity_atol"],
    }


def probe_condition(policy, condition, record, steps, parity, output_directory, phase):
    """One matched forward/inverse/replay experiment, with lossless endpoints."""
    noise = policy.tensor(record["known_noise"])
    began = time.perf_counter()
    kwargs = {"steps": steps, "solver": "rk4", **SOLVER_OPTIONS}
    forward = policy.sample(condition, noise, **kwargs)
    inverse = policy.invert(condition, forward.value, **kwargs)
    replay = policy.sample(condition, inverse.value, **kwargs)
    recovery = error_metrics(noise, inverse.value)
    reconstruction = error_metrics(forward.value, replay.value)
    controller = error_metrics(
        _decoded(policy, condition, forward.value),
        _decoded(policy, condition, replay.value),
    )
    source = record["source"]
    artifact = _save_arrays(
        Path(output_directory)
        / f"rk4_{steps}"
        / f"step_{source['observation_step']:04d}_{source['condition_id'][:12]}.npz",
        known_noise=noise,
        endpoint=forward.value,
        recovered_noise=inverse.value,
        replay=replay.value,
    )
    solves = {
        name: {
            "grid": solve.grid,
            "velocity_evaluations": solve.velocity_evaluations,
            "latency_seconds": solve.latency_seconds,
        }
        for name, solve in (
            ("forward", forward),
            ("inverse", inverse),
            ("replay", replay),
        )
    }
    return {
        **source,
        "steps": steps,
        "phase": phase,
        "full_internal_shape": list(record["known_noise"].shape),
        "full_internal_reconstruction": reconstruction,
        "controller_reconstruction": controller,
        "known_noise_recovery": recovery,
        "known_noise_action_channels": error_metrics(
            record["known_noise"][..., :7], to_numpy(inverse.value)[..., :7]
        ),
        "known_noise_padding_channels": error_metrics(
            record["known_noise"][..., 7:], to_numpy(inverse.value)[..., 7:]
        ),
        "reference_sampler_parity": parity["error"],
        "reference_sampler_parity_steps": parity["steps"],
        "endpoint_change_from_recorded_rk4_50": error_metrics(
            record["endpoint"], forward.value
        ),
        "solves": solves,
        "velocity_evaluations": sum(s["velocity_evaluations"] for s in solves.values()),
        "latency_seconds": time.perf_counter() - began,
        "artifacts": artifact,
        "passed": reconstruction["max_abs"] <= TOLERANCES["action_atol"]
        and controller["max_abs"] <= TOLERANCES["action_atol"]
        and recovery["max_abs"] <= TOLERANCES["noise_atol"]
        and parity["passed"],
    }


def summarize_candidate(steps, rows, status):
    result = {
        "steps": steps,
        "status": status,
        "conditions_tested": len(rows),
        "conditions_required": EXPECTED_CONDITIONS,
        "passing_conditions": sum(row["passed"] for row in rows),
        "velocity_evaluations_per_solve": 4 * steps,
        "velocity_evaluations": sum(row["velocity_evaluations"] for row in rows),
        "latency_seconds": sum(row["latency_seconds"] for row in rows),
        "passed": len(rows) == EXPECTED_CONDITIONS
        and all(row["passed"] for row in rows),
        "conditions": rows,
    }
    for key in (
        "full_internal_reconstruction",
        "controller_reconstruction",
        "known_noise_recovery",
        "known_noise_action_channels",
        "known_noise_padding_channels",
        "reference_sampler_parity",
    ):
        result[key] = (
            {
                metric: max(row[key][metric] for row in rows)
                for metric in ("rmse", "max_abs")
            }
            if rows
            else None
        )
    result["aggregate_error_definition"] = (
        "Maximum of per-condition errors; RMSE is worst-condition RMSE, not pooled RMSE"
    )
    return result


def run_probe(args, manifest, records, provenance):
    import torch

    from astra_reversal.policy_adapter import load_policy

    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(
            "Runtime numerical validation requires an allocated CUDA GPU"
        )
    device = torch.device(args.device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    gpu = torch.cuda.get_device_name(device)
    if args.expected_gpu not in gpu:
        raise RuntimeError(f"Expected {args.expected_gpu}, found {gpu}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    openpi_client = PACKAGE_ROOT.parent / "external/openpi/packages/openpi-client/src"
    if openpi_client.is_dir():
        sys.path.insert(0, str(openpi_client))
    inventory = json.loads(
        (PACKAGE_ROOT / "checkpoints/lerobot_pi05_libero_base.json").read_text()
    )
    checkpoint = Path(args.checkpoint).expanduser()
    if not all((checkpoint / item["path"]).is_file() for item in inventory["files"]):
        from huggingface_hub import snapshot_download

        snapshot_download(
            inventory["repo_id"],
            revision=inventory["revision"],
            local_dir=checkpoint,
            token=False,
        )
    for item in inventory["files"]:
        if file_sha256(checkpoint / item["path"]) != item["sha256"]:
            raise ValueError(f"Pinned checkpoint hash mismatch: {item['path']}")
    config = manifest["config"]["policy"]
    policy = load_policy(
        args.checkpoint,
        config["config_name"],
        args.device,
        config["checkpoint_provenance"],
        config["training_overlap"],
        tokenizer_path=args.tokenizer_path,
        input_profile=config["input_profile"],
        reference_assets=args.reference_assets,
    )
    identity = _check_checkpoint(policy, manifest)
    ordered = sorted(
        records,
        key=lambda row: (
            -row["source"]["recorded_known_noise_recovery"]["max_abs"],
            row["source"]["observation_step"],
        ),
    )
    screening = ordered[: args.screen_count]
    remaining = ordered[args.screen_count :]
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite numerical results: {output}")
    report = {
        "schema_version": "1.0",
        "purpose": "Development-only numerical solver selection on all recorded reference conditions",
        "status": "running",
        "solver": "rk4",
        "solver_options": SOLVER_OPTIONS,
        "checkpoint": policy.metadata,
        "checkpoint_matches_recorded_source": identity,
        "action_spec_id": manifest["action_spec"]["action_spec_id"],
        "source": provenance,
        "tolerances": TOLERANCES,
        "proposal_source": "policy_generated_from_recorded_full_known_noise",
        "selection_rule": "Minimum velocity evaluations per solve among candidates passing every one of the 14 recorded development conditions; no OOD or control-success selection",
        "screening_rule": "Descending prior RK4/50 maximum known-noise error; reject a candidate after the screen if any screened condition fails",
        "screening_condition_ids": [r["source"]["condition_id"] for r in screening],
        "evaluate_all_candidates_on_all_conditions": args.evaluate_all,
        "full_condition_order": [r["source"] for r in ordered],
        "runtime": {
            "gpu": gpu,
            "visible_gpu_count": torch.cuda.device_count(),
            "device": args.device,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tf32": False,
            "workflow": os.environ.get("ASTRA_RUN_ID"),
            "payload_sha256": os.environ.get("PAYLOAD_SHA256"),
            "probe_source_sha256": file_sha256(__file__),
        },
        "results": [],
        "native_parity": {},
        "selected_solver": None,
        "experiments_run": False,
        "astra_calls_run": False,
        "interpretation": "Only matched-condition numerical reconstruction on one recorded development trajectory; no Astra-proposal, OOD, or control-success claim",
    }
    archive = None
    if os.environ.get("ASTRA_RUN_ID") and os.environ.get("R2_ENDPOINT_URL"):
        from astra_reversal.osmo.experiment import RESULTS, Archive

        if not output.resolve().is_relative_to(RESULTS.resolve()):
            raise ValueError(
                "OSMO archival requires output beneath its run results directory"
            )
        archive = Archive()

    def save():
        write_json(output, report)
        if archive is not None:
            archive.sync()

    candidate_rows = {steps: [] for steps in args.steps}
    candidate_status = {steps: "pending" for steps in args.steps}
    began = time.perf_counter()

    def refresh():
        report["results"] = [
            summarize_candidate(n, candidate_rows[n], candidate_status[n])
            for n in args.steps
        ]
        report["velocity_evaluations"] = sum(
            r["velocity_evaluations"] for r in report["results"]
        ) + sum(r["velocity_evaluations"] for r in report["native_parity"].values())
        report["latency_seconds"] = time.perf_counter() - began
        save()

    def evaluate(record, candidates, phase):
        source = record["source"]
        print(
            json.dumps(
                {
                    "phase": "runtime_condition",
                    "observation_step": source["observation_step"],
                    "candidates": candidates,
                    "stage": phase,
                }
            ),
            flush=True,
        )
        condition = policy.prepare(
            record["observation"], source["observation_id"], source["prompt"]
        )
        if condition.condition_id != source["condition_id"]:
            raise ValueError(
                "Reconstructed policy condition differs from recorded condition"
            )
        parity = _parity(policy, condition, policy.tensor(record["known_noise"]))
        parity["preparation_seconds"] = condition.preparation_seconds
        report["native_parity"][source["condition_id"]] = parity
        for steps in candidates:
            row = probe_condition(
                policy,
                condition,
                record,
                steps,
                parity,
                output.parent / "runtime_arrays",
                phase,
            )
            candidate_rows[steps].append(row)
            candidate_status[steps] = f"running_{phase}"
            refresh()
            print(
                json.dumps(
                    {
                        "phase": "runtime_result",
                        "steps": steps,
                        "observation_step": source["observation_step"],
                        "noise": row["known_noise_recovery"],
                        "action": row["controller_reconstruction"],
                        "passed": row["passed"],
                        "velocity_evaluations": row["velocity_evaluations"],
                        "latency_seconds": row["latency_seconds"],
                    }
                ),
                flush=True,
            )
        del condition
        gc.collect()

    try:
        save()
        for record in screening:
            evaluate(record, args.steps, "screening")
        candidates = [
            steps
            for steps in args.steps
            if args.evaluate_all or all(row["passed"] for row in candidate_rows[steps])
        ]
        for steps in args.steps:
            candidate_status[steps] = (
                "screen_passed" if steps in candidates else "rejected_screening"
            )
        refresh()
        for record in remaining:
            if candidates:
                evaluate(record, candidates, "all_conditions")
        for steps in candidates:
            candidate_status[steps] = "complete"
        refresh()
        passing = [row for row in report["results"] if row["passed"]]
        if passing:
            selected = min(
                passing,
                key=lambda row: (row["velocity_evaluations_per_solve"], row["steps"]),
            )
            report["selected_solver"] = {
                "solver": "rk4",
                "steps": selected["steps"],
                "solver_options": SOLVER_OPTIONS,
                "velocity_evaluations_per_solve": selected[
                    "velocity_evaluations_per_solve"
                ],
                "conditions_passed": EXPECTED_CONDITIONS,
                "diagnostics": str(output),
            }
            report["status"] = "complete_passing_solver"
        else:
            report["status"] = "complete_no_passing_solver"
        refresh()
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["selected_solver"] = None
        # An interrupted/incomplete run cannot produce an accepted gate.
        for row in report["results"]:
            row["passed"] = False
        save()
        raise
    finally:
        if archive is not None:
            archive.sync(include_arrays=True)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--record-root", default="astra_reversal/.deps/runtime-probe-input"
    )
    result.add_argument(
        "--checkpoint", default="astra_reversal/.deps/checkpoints/pi05_libero_base"
    )
    result.add_argument(
        "--tokenizer-path",
        default="astra_reversal/.deps/tokenizers/paligemma-3b-pt-224",
    )
    result.add_argument(
        "--reference-assets", default="astra_reversal/.deps/reference/pi05_libero"
    )
    result.add_argument("--device", default="cuda")
    result.add_argument("--expected-gpu", default="L40S")
    result.add_argument(
        "--output", default="astra_reversal/artifacts/osmo/runtime_diagnostics.json"
    )
    result.add_argument("--steps", nargs="+", type=int, default=[100, 200, 500])
    result.add_argument("--screen-count", type=int, default=3)
    result.add_argument(
        "--evaluate-all",
        action="store_true",
        help="Evaluate even screening failures on all 14 conditions",
    )
    result.add_argument("--verify-records-only", action="store_true")
    result.add_argument(
        "--pack-input",
        metavar="DESTINATION",
        help="Create a reduced input bundle without loading a policy",
    )
    return result


def main():
    args = parser().parse_args()
    if args.steps != sorted(set(args.steps)) or any(n < 1 for n in args.steps):
        raise ValueError("Declare distinct positive step counts in increasing order")
    if not 0 <= args.screen_count <= EXPECTED_CONDITIONS:
        raise ValueError(f"screen-count must be between 0 and {EXPECTED_CONDITIONS}")
    if args.pack_input:
        bundle = pack_records(args.record_root, args.pack_input)
        print(
            json.dumps(
                {
                    "input_bundle": args.pack_input,
                    "conditions": bundle["conditions"],
                    "arrays": len(bundle["verified_arrays"]),
                    "array_bytes": sum(
                        item["bytes"] for item in bundle["verified_arrays"]
                    ),
                    "source_events_sha256": bundle["events_sha256"],
                }
            ),
            flush=True,
        )
        return
    manifest, records, provenance = load_records(args.record_root)
    if args.verify_records_only:
        print(
            json.dumps(
                {
                    key: value
                    for key, value in provenance.items()
                    if key != "verified_arrays"
                }
            ),
            flush=True,
        )
        return
    report = run_probe(args, manifest, records, provenance)
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_solver": report["selected_solver"],
                "output": args.output,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
