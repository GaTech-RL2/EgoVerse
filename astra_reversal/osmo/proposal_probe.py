"""GPU round-trip preflight for one genuine, recorded Astra development proposal.

This reads the accepted vision smoke response; it never calls Astra or creates a
proposal. The selected solver must first pass all 14 recorded development
conditions. OOD episodes and their outcomes are not inputs to this preflight.
"""

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from astra_reversal.action_adapter import ActionAdapter, ActionSpec
from astra_reversal.agent import CAMERAS, SCHEMA_VERSION, parse_proposal
from astra_reversal.diagnostics import diagnose, require_diagnostics
from astra_reversal.flow import error_metrics
from astra_reversal.records import digest, file_sha256, to_numpy

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ACTUAL_MODEL = "azure/openai/gpt-6-astra"
TOLERANCES = {"action_atol": 0.02, "noise_atol": 0.1, "parity_atol": 1e-5}
# These are the unmodified accepted development smoke files, not credentials.
INPUT_SHA256 = {
    "request.json": "1742730edff547cf932d7233785e31556dfcd4bbede22ab8b90bc8f2bddfd777",
    "response.json": "83148481657a8691c8a189f7b061e47906914f3692369157ba0a4db9282b6643",
    "provider.jsonl": "7cc876e14ae8e65d230b1d766aa9bb55589a0a2397790841e60163b26cfdf951",
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_proposal(directory):
    """Recover the exact request arrays and validate accepted provider output."""
    directory = Path(directory)
    hashes = {name: file_sha256(directory / name) for name in INPUT_SHA256}
    if hashes != INPUT_SHA256:
        raise ValueError(
            "Proposal input files differ from the genuine accepted smoke artifacts"
        )
    request = json.loads((directory / "request.json").read_text())
    observation = request["observation"]
    if set(observation) != {*CAMERAS, "observation/state"}:
        raise ValueError(
            "The genuine vision request must contain both cameras and robot state"
        )
    for key in CAMERAS:
        value = observation[key]
        if set(value) != {"encoding", "data"} or value["encoding"] != "base64_png":
            raise ValueError("Expected a lossless RGB PNG camera encoding")
        with Image.open(
            io.BytesIO(base64.b64decode(value["data"], validate=True))
        ) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (224, 224):
                raise ValueError(
                    "The development request must retain its original RGB camera pixels"
                )
            observation[key] = np.asarray(image, dtype=np.uint8).copy()
    # build_request hashed the original float32 ndarray, before wire conversion.
    observation["observation/state"] = np.asarray(
        observation["observation/state"], dtype=np.float32
    )
    if (
        observation["observation/state"].shape != (8,)
        or not np.isfinite(observation["observation/state"]).all()
    ):
        raise ValueError("Invalid original robot state")
    if digest(observation) != request["observation_id"]:
        raise ValueError("Reconstructed observation does not match its recorded digest")
    fingerprint = request["request_fingerprint"]
    if (
        digest({k: v for k, v in request.items() if k != "request_fingerprint"})
        != fingerprint
    ):
        raise ValueError("Reconstructed request fingerprint mismatch")
    if (
        request["schema_version"] != SCHEMA_VERSION
        or request["stage"] != 1
        or request["episode_id"] != "libero_10:task0:state0"
        or request["observation_step"] != 0
        or request["requested_model_version"] != ACTUAL_MODEL
    ):
        raise ValueError(
            "Expected the recorded Stage-1 LIBERO-10 development vision request"
        )
    spec = ActionSpec(**request["action_spec"])
    spec_payload = {
        key: request["action_spec"][key]
        for key in (
            "horizon",
            "model_action_dim",
            "timestep_seconds",
            "lower",
            "upper",
            "semantics",
        )
    }
    if digest(spec_payload) != spec.action_spec_id or (
        spec.horizon,
        spec.model_action_dim,
    ) != (10, 32):
        raise ValueError("Recorded controller action specification mismatch")
    provider_rows = [
        json.loads(line)
        for line in (directory / "provider.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(provider_rows) != 1:
        raise ValueError("Expected exactly one accepted provider record")
    provider = provider_rows[0]
    if (
        provider["accepted"] is not True
        or provider["http_status"] != 200
        or provider["requested_model"] != ACTUAL_MODEL
        or provider["response"]["model"] != ACTUAL_MODEL
        or provider["request_fingerprint"] != fingerprint
        or any(
            provider[key] != request[key]
            for key in ("episode_id", "observation_step", "stage", "sampling_settings")
        )
    ):
        raise ValueError(
            "Provider record does not establish the accepted request/model identity"
        )
    choices = provider["response"]["choices"]
    if (
        len(choices) != 1
        or choices[0]["finish_reason"] != "stop"
        or choices[0]["message"]["role"] != "assistant"
    ):
        raise ValueError("Expected one complete provider response")
    raw = (directory / "response.json").read_text().strip()
    if choices[0]["message"]["content"].strip() != raw:
        raise ValueError("Saved action proposal differs from the raw provider response")
    proposal = parse_proposal(
        raw,
        request,
        model_version=ACTUAL_MODEL,
        request_time=provider["request_time"],
        response_time=provider["response_time"],
        max_timeout=request["max_timeout_env_steps"],
        completion_types=request["supported_completion_types"],
    )
    provenance = {
        "file_sha256": hashes,
        "request_fingerprint": fingerprint,
        "observation_id": request["observation_id"],
        "condition_id": digest({**observation, "prompt": request["task_instruction"]}),
        "episode_id": request["episode_id"],
        "observation_step": request["observation_step"],
        "plan_id": proposal.plan_id,
        "actual_model": provider["response"]["model"],
        "requested_model": provider["requested_model"],
        "provider_response_id": provider["response"]["id"],
        "provider_accepted": True,
        "request_time": provider["request_time"],
        "response_time": provider["response_time"],
        "action_chunk_sha256": digest(proposal.action_chunk),
        "action_chunk_shape": list(proposal.action_chunk.shape),
        "observation_arrays": {
            key: {
                "sha256": digest(value),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
            for key, value in observation.items()
        },
    }
    return request, proposal, spec, provenance


def selected_runtime(path, provenance, spec):
    report = json.loads(Path(path).read_text())
    selected = report.get("selected_solver") or {}
    if (
        report.get("status") != "complete_passing_solver"
        or report.get("tolerances") != TOLERANCES
        or report.get("solver") != "rk4"
        or report.get("solver_options") != {"time_power": 3.0}
        or selected.get("solver") != report["solver"]
        or selected.get("solver_options") != report["solver_options"]
        or selected.get("conditions_passed") != 14
        or report["source"]["split"] != "development"
        or report["source"]["suite"] != "libero_10"
        or report["action_spec_id"] != spec.action_spec_id
    ):
        raise ValueError(
            "Proposal preflight requires a selected, passing 14-condition development solver"
        )
    rows = [row for row in report["results"] if row["steps"] == selected.get("steps")]
    if len(rows) != 1:
        raise ValueError(
            "The selected runtime solver must have exactly one diagnostic row"
        )
    row = rows[0]
    conditions = row.get("conditions", [])
    if (
        row.get("passed") is not True
        or row.get("conditions_tested") != 14
        or row.get("passing_conditions") != 14
        or len(conditions) != 14
        or len({item["condition_id"] for item in conditions}) != 14
        or any(item.get("passed") is not True for item in conditions)
    ):
        raise ValueError(
            "A screened or incomplete solver cannot authorize proposal preflight"
        )
    for condition in conditions:
        for metric, tolerance in (
            ("known_noise_recovery", 0.1),
            ("full_internal_reconstruction", 0.02),
            ("controller_reconstruction", 0.02),
            ("reference_sampler_parity", 1e-5),
        ):
            error = condition[metric]["max_abs"]
            if not np.isfinite(error) or error < 0 or error > tolerance:
                raise ValueError(
                    "Runtime diagnostic metrics contradict a passing condition"
                )
    source = next(
        (
            item
            for item in conditions
            if item["condition_id"] == provenance["condition_id"]
        ),
        None,
    )
    if source is None or any(
        source[key] != provenance[key]
        for key in ("observation_id", "episode_id", "observation_step")
    ):
        raise ValueError(
            "The genuine proposal observation must be one of the validated development conditions"
        )
    config = SimpleNamespace(
        flow=SimpleNamespace(
            integrator=report["solver"],
            inversion_steps=selected["steps"],
            generation_steps=selected["steps"],
            solver_options=report["solver_options"],
        )
    )
    return report, config


class CapturedPolicy:
    """Observe existing diagnose calls; do not change or add policy solves."""

    def __init__(self, policy):
        self.policy, self.samples, self.inverses, self.condition = policy, [], [], None

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def prepare(self, *args, **kwargs):
        self.condition = self.policy.prepare(*args, **kwargs)
        return self.condition

    def sample(self, condition, initial, **kwargs):
        result = self.policy.sample(condition, initial, **kwargs)
        self.samples.append((to_numpy(initial).copy(), result))
        return result

    def invert(self, condition, initial, **kwargs):
        result = self.policy.invert(condition, initial, **kwargs)
        self.inverses.append((to_numpy(initial).copy(), result))
        return result


def diagnose_proposal(policy, actions, request, proposal, steps, artifact_path):
    captured = CapturedPolicy(policy)
    report = diagnose(
        captured,
        actions,
        request["observation"],
        request["task_instruction"],
        seed=0,
        resolutions=(steps,),
        solver="rk4",
        solver_options={"time_power": 3.0},
        controller_actions=proposal.action_chunk,
        **TOLERANCES,
    )
    if len(captured.samples) != 3 or len(captured.inverses) != 2:
        raise RuntimeError(
            "The diagnostic solve sequence changed; recheck proposal capture"
        )
    encoded, inverse = captured.inverses[0]
    replay = captured.samples[1][1]
    expected = actions.encode(
        proposal.action_chunk,
        {**request["observation"], "prompt": request["task_instruction"]},
    )
    np.testing.assert_array_equal(encoded, expected)
    np.testing.assert_array_equal(captured.samples[1][0], to_numpy(inverse.value))
    state = to_numpy(captured.condition.state)[0].copy()
    decoded_endpoint = actions.output_transform(
        {"actions": encoded[0].copy(), "state": state}
    )["actions"]
    decoded_replay = actions.output_transform(
        {"actions": to_numpy(replay.value)[0].copy(), "state": state}
    )["actions"]
    actual = error_metrics(proposal.action_chunk, decoded_replay)
    encoding = error_metrics(proposal.action_chunk, decoded_endpoint)
    report["actual_controller_proposal"] = {
        "known_source_noise": None,
        "known_source_noise_explanation": "Astra supplied controller actions; no true generating policy noise exists for this proposal",
        "encoding_decoding_error": encoding,
        "reconstruction": actual,
        "action_atol": TOLERANCES["action_atol"],
        "clipping_applied_to_metric": False,
        "passed": actual["max_abs"] <= TOLERANCES["action_atol"]
        and encoding["max_abs"] <= TOLERANCES["action_atol"],
        "velocity_evaluations": inverse.velocity_evaluations
        + replay.velocity_evaluations,
        "latency_seconds": inverse.latency_seconds + replay.latency_seconds,
    }
    report["known_noise_recovery_scope"] = (
        "Separate policy-generated sample from seed 0 on the same observation; never a known-noise claim for Astra actions"
    )
    values = {
        "proposal_controller_actions": proposal.action_chunk,
        "proposal_encoded_model_actions": encoded,
        "proposal_recovered_noise": to_numpy(inverse.value),
        "proposal_replay_model_actions": to_numpy(replay.value),
        "proposal_replay_controller_actions": np.asarray(decoded_replay),
        "policy_reference_known_noise": captured.samples[0][0],
        "policy_reference_endpoint": to_numpy(captured.samples[0][1].value),
        "policy_reference_recovered_noise": to_numpy(captured.inverses[1][1].value),
    }
    np.savez_compressed(artifact_path, **values)
    report["roundtrip_artifacts"] = {
        "path": str(artifact_path),
        "file_sha256": file_sha256(artifact_path),
        "arrays": {
            name: {
                "sha256": digest(value),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in values.items()
        },
    }
    row = report["results"][0]
    row["diagnose_passed"] = row["passed"]
    row["passed"] = row["passed"] and report["actual_controller_proposal"]["passed"]
    return report


def run_gpu(args, request, proposal, spec, provenance, runtime, config):
    import torch

    from astra_reversal.policy_adapter import load_policy

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(
            "Actual proposal numerical validation requires an allocated CUDA GPU"
        )
    device = torch.device(args.device)
    torch.cuda.set_device(
        device if device.index is not None else torch.cuda.current_device()
    )
    gpu = torch.cuda.get_device_name()
    if args.expected_gpu not in gpu:
        raise RuntimeError(f"Expected {args.expected_gpu}, found {gpu}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    client = PACKAGE_ROOT.parent / "external/openpi/packages/openpi-client/src"
    if client.is_dir():
        sys.path.insert(0, str(client))
    inventory = json.loads(
        (PACKAGE_ROOT / "checkpoints/lerobot_pi05_libero_base.json").read_text()
    )
    for item in inventory["files"]:
        if file_sha256(Path(args.checkpoint) / item["path"]) != item["sha256"]:
            raise ValueError(f"Pinned checkpoint file mismatch: {item['path']}")
    policy = load_policy(
        args.checkpoint,
        "pi05_libero",
        args.device,
        runtime["checkpoint"]["provenance"],
        runtime["checkpoint"]["training_overlap"],
        tokenizer_path=args.tokenizer_path,
        input_profile="openpi_libero",
        reference_assets=args.reference_assets,
    )
    if any(p.requires_grad for p in policy.policy.parameters()):
        raise ValueError(
            "The genuine-proposal preflight requires frozen policy weights"
        )
    actions = ActionAdapter(spec, policy.input_transform, policy.output_transform)
    require_diagnostics(runtime, policy, actions, config)
    print(
        json.dumps(
            {
                "phase": "genuine_proposal_diagnostics",
                "gpu": gpu,
                "steps": config.flow.inversion_steps,
                "model": ACTUAL_MODEL,
                "request_fingerprint": provenance["request_fingerprint"],
            }
        ),
        flush=True,
    )
    report = diagnose_proposal(
        policy,
        actions,
        request,
        proposal,
        config.flow.inversion_steps,
        Path(args.output).with_suffix(".npz"),
    )
    report["runtime"] = {
        "gpu": gpu,
        "visible_gpu_count": torch.cuda.device_count(),
        "device": args.device,
        "tf32": False,
    }
    return report, policy, actions


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--runtime-diagnostics",
        default="astra_reversal/.deps/ood-inputs/runtime_diagnostics.json",
    )
    result.add_argument(
        "--proposal-input", default="astra_reversal/.deps/astra-proposal-input"
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
        "--output",
        default="astra_reversal/artifacts/osmo/astra_proposal_diagnostics.json",
    )
    result.add_argument("--verify-input-only", action="store_true")
    return result


def main():
    args = parser().parse_args()
    if args.verify_input_only:
        print(json.dumps(load_proposal(args.proposal_input)[3]), flush=True)
        return
    output = Path(args.output)
    if output.exists() or output.with_suffix(".npz").exists():
        raise FileExistsError(
            "Refusing to overwrite a genuine-proposal numerical report"
        )
    report = {
        "status": "running",
        "actual_model": ACTUAL_MODEL,
        "tolerances": TOLERANCES,
        "results": [],
        "astra_calls_run": False,
        "probe_source_sha256": file_sha256(__file__),
        "interpretation": "One recorded genuine Astra proposal on LIBERO-10 development; no OOD or control-success result",
    }
    began = time.perf_counter()
    try:
        write_json(output, report)
        request, proposal, spec, provenance = load_proposal(args.proposal_input)
        report["proposal_provenance"] = provenance
        runtime, config = selected_runtime(args.runtime_diagnostics, provenance, spec)
        report["runtime_diagnostics_sha256"] = file_sha256(args.runtime_diagnostics)
        report["selected_solver"] = runtime["selected_solver"]
        write_json(output, report)
        numerical, policy, actions = run_gpu(
            args, request, proposal, spec, provenance, runtime, config
        )
        report.update(numerical)
        report["interpretation"] = (
            "Round-trip numerical preflight of one recorded genuine Astra development proposal, plus a separate same-observation known-noise check; no OOD or control-success claim"
        )
        write_json(output, report)
        require_diagnostics(report, policy, actions, config)
        if not report["actual_controller_proposal"]["passed"]:
            raise ValueError(
                "The actual Astra controller proposal failed round-trip tolerance"
            )
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - began
        write_json(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "actual_model": ACTUAL_MODEL,
                "output": str(output),
                "actual_controller_proposal": report["actual_controller_proposal"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
