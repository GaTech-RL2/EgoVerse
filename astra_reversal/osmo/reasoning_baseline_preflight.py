"""Zero-environment-action, full-weight RLinf conversion and numerical audit."""

import json
import os
import time

import numpy as np
import torch

from astra_reversal.intervention_search import write_json
from astra_reversal.osmo.experiment import RESULTS, ROOT
from astra_reversal.osmo.interpolation import load_frozen_policy
from astra_reversal.osmo.ood_distributed import WorkerArchive
from astra_reversal.osmo.reasoning_policy_learning import load_protocol
from astra_reversal.reasoning_learning import rlinf_bridge as bridge
from astra_reversal.reasoning_learning.guidance import GuidanceConfig, generate


def guidance_sweep(native, observation, prompt):
    from astra_reversal.lerobot_policy import prepare_velocity

    batch = native._preprocess({**observation, "prompt": prompt})
    velocity = prepare_velocity(native.policy, batch, differentiable=True)
    noise = native.noise(np.random.default_rng(173))
    zero = torch.zeros_like(noise)
    reference, _ = generate(velocity, noise, zero, zero, GuidanceConfig(strength=0))
    target = reference.clone()
    target[:, :5, 2] += 0.1
    mask = zero.clone()
    mask[:, :5, 2] = 1
    initial_error = float(((reference - target) * mask).norm())
    rows = []
    for strength in (0.25, 1, 5, 10):
        candidate, trace = generate(
            velocity, noise, target, mask, GuidanceConfig(strength=strength)
        )
        error = float(((candidate - target) * mask).norm())
        rows.append(
            {
                "strength": strength,
                "initial_error": initial_error,
                "endpoint_error": error,
                "fraction_error_reduction": 1 - error / initial_error,
                "max_action_change": float((candidate - reference).abs().max()),
                "off_mask_change": float(
                    ((candidate - reference) * (1 - mask)).abs().max()
                ),
                "trace": trace,
            }
        )
    return {
        "environment_actions": 0,
        "target": "native normalized z + 0.1 on first five actions",
        "cases": rows,
    }


def main():
    protocol = load_protocol()
    allocation = float(os.environ["ASTRA_WORKER_GPU_HOURS"])
    if not 0 < allocation <= protocol["compute"]["authorized_gpu_hours"]:
        raise ValueError("Worker allocation exceeds study budget")
    if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Allocate one OSMO L40S")
    RESULTS.mkdir(parents=True, exist_ok=False)
    archive = WorkerArchive(0)
    start = time.monotonic()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    write_json(
        RESULTS / "runtime.json",
        {
            "workflow": os.environ["ASTRA_RUN_ID"],
            "source_revision": os.environ["ASTRA_SOURCE_REVISION"],
            "payload_sha256": os.environ["PAYLOAD_SHA256"],
            "rlinf_revision": bridge.REVISION,
            "environment_actions": 0,
            "purpose": "conversion compatibility, not an RL result",
        },
    )
    try:
        native = load_frozen_policy()
        inventory = json.loads(
            (
                ROOT / "astra_reversal/checkpoints/lerobot_pi05_libero_base.json"
            ).read_text()
        )
        with np.load(
            ROOT / "astra_reversal/.deps/reference/cpu_libero_probe.npz",
            allow_pickle=False,
        ) as data:
            observation = {
                key: data[key]
                for key in (
                    "observation/image",
                    "observation/wrist_image",
                    "observation/state",
                )
            }
            prompt = str(data["prompt"].item())
        write_json(
            RESULTS / "guidance_sweep.json", guidance_sweep(native, observation, prompt)
        )
        archive.sync()
        core = bridge.import_core(ROOT / "astra_reversal/.deps/RLinf")
        converted, receipt = bridge.load_converted(
            core,
            ROOT / inventory["local_directory"],
            horizon=native.horizon,
            device="cuda",
        )
        receipt["unmodified_rlinf"] = bridge.measure(
            core, converted, native, observation, prompt
        )
        write_json(RESULTS / "parity.json", receipt)
        archive.sync()
        with bridge.match_native_gelu(core, converted, native.policy) as compatibility:
            receipt["activation_compatibility"] = compatibility
            receipt["gelu_matched_rlinf"] = bridge.measure(
                core, converted, native, observation, prompt
            )
        receipt["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
        write_json(RESULTS / "parity.json", receipt)
        write_json(
            RESULTS / "completion.json",
            {
                "status": "parity_audit_complete",
                "passed": receipt["gelu_matched_rlinf"]["passed"],
                "research_objective_met": False,
            },
        )
    finally:
        write_json(
            RESULTS / "cost.json",
            {
                "worker_gpu_hours": (time.monotonic() - start) / 3600,
                "bootstrap_gpu_hours_excluded": True,
            },
        )
        archive.sync()


if __name__ == "__main__":
    main()
