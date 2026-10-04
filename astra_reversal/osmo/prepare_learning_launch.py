"""Prepare one immutable, time-limited OSMO worker; never submit on import."""

import argparse
import json
import math
import secrets
import shlex
import subprocess
from pathlib import Path

import yaml

from astra_reversal.osmo.reasoning_policy_learning import load_protocol
from astra_reversal.reasoning_learning.rl_recipes import collection_screen, settings
from astra_reversal.records import file_sha256


def connection_script(repo, bundle, destination, *, phase, gpu_hours, port):
    """Upload before opening the tunnel: rsync waits for the worker to exist."""
    token = destination / "relay.token"
    script = '#!/usr/bin/env bash\nset -euo pipefail\ntest "$#" = 1\nworkflow="$1"\ncase "$workflow" in astra-pi05-reasoning-learning-20261003-*) ;; *) exit 2;; esac\n'
    script += "cd " + shlex.quote(str(repo)) + "\n"
    script += f'osmo workflow rsync "$workflow" worker0 {shlex.quote(str(bundle / "payload.tar.gz") + ":/osmo/run/workspace")} --once --timeout 180\n'
    if phase == "pilot":
        script += f'osmo workflow port-forward "$workflow" worker0 --port {port}:8769 --connect-timeout 60 > {shlex.quote(str(destination / "port-forward.log"))} 2>&1 &\nforward_pid=$!\n'
        script += 'relay_pid=""\ntrap \'kill "$forward_pid" ${relay_pid:+"$relay_pid"} 2>/dev/null || true\' EXIT\n'
        script += f"python -m astra_reversal.codex_relay --url http://127.0.0.1:{port} --token-file {shlex.quote(str(token))} --directory {shlex.quote(str(destination / 'jobs'))} --idle-timeout {math.ceil(gpu_hours * 3600)} &\nrelay_pid=$!\n"
        script += 'while kill -0 "$relay_pid" 2>/dev/null; do\n  if ! kill -0 "$forward_pid" 2>/dev/null; then\n    echo "OSMO tunnel exited; stopping the local relay. Inspect port-forward.log." >&2\n    exit 1\n  fi\n  sleep 1\ndone\nwait "$relay_pid"\n'
    return script


def prepare(
    bundle,
    destination,
    *,
    phase,
    gpu_hours,
    task_index=0,
    seed_index=0,
    port=19943,
    protocol_version="v1",
    rl_recipe="standard",
    collection_limit=None,
):
    protocol = load_protocol(version=protocol_version)
    if (
        phase in ("replay-learning", "replay-masked", "replay-credit")
        and protocol.get("learner", {}).get("loss_action_dimensions") != 7
    ):
        raise ValueError("Executed replay ablation requires the corrected native loss")
    if rl_recipe not in ("standard", "more_reuse") or (
        rl_recipe != "standard" and phase not in ("dsrl", "ppo")
    ):
        raise ValueError("RL tuning applies only to a recorded baseline recipe")
    if (
        phase
        not in (
            "preflight",
            "pilot",
            "baseline-preflight",
            "dsrl",
            "guidance-probe",
            "ppo",
            "replay-learning",
            "replay-masked",
            "replay-credit",
            "checkpoint-evaluation",
        )
        or not math.isfinite(gpu_hours)
        or not (1 / 60) <= gpu_hours <= protocol["compute"]["authorized_gpu_hours"]
    ):
        raise ValueError("Choose a valid phase within the recorded study budget")
    if not 0 <= task_index < len(
        protocol["pilot"]["development_tasks"]
    ) or not 0 <= seed_index < len(protocol["pilot"]["seeds"]):
        raise ValueError("Task/seed outside the preregistered pilot")
    if not 1024 <= port <= 65535:
        raise ValueError("Unprivileged local relay port required")
    if collection_limit is not None:
        if phase not in ("pilot", "dsrl", "ppo"):
            raise ValueError("Collection limits require a collecting study worker")
        period = (
            1
            if phase == "pilot"
            else settings(
                rl_recipe,
                phase,
                collection_rollouts=len(protocol["pilot"]["collection_reset_indices"]),
                evaluation_schedule=protocol["pilot"][
                    "evaluation_after_collection_rollouts"
                ],
            ).get("collection_rollouts_per_update", 1)
        )
        collection_screen(
            protocol["pilot"]["collection_reset_indices"],
            protocol["pilot"]["evaluation_after_collection_rollouts"],
            collection_limit,
            update_period=period,
        )
    repo = Path(__file__).resolve().parents[2]
    bundle, destination = Path(bundle).resolve(), Path(destination).resolve()
    identity = json.loads((bundle / "source_identity.json").read_text())
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if revision != identity["source_revision"]:
        raise ValueError("Bundle source revision differs from this checkout")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", "astra_reversal", "tests"],
        cwd=repo,
        text=True,
    ).strip():
        raise ValueError("Commit source and tests before preparing an experiment")
    for filename, field in (
        ("payload.tar.gz", "payload_sha256"),
        ("bootstrap.sh", "bootstrap_sha256"),
    ):
        if file_sha256(bundle / filename) != identity[field]:
            raise ValueError("Immutable bundle checksum differs")
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    token = destination / "relay.token"
    token.write_text(secrets.token_urlsafe(32))
    token.chmod(0o600)
    spec = yaml.safe_load(
        (repo / "astra_reversal/osmo/demo_skill_library_l40s.yaml").read_text()
    )
    workflow = spec["workflow"]
    workflow["name"] = f"astra-pi05-reasoning-learning-20261003-{phase}"
    workflow["timeout"]["exec_timeout"] = f"{max(1, math.floor(gpu_hours * 60))}m"
    workflow["tasks"] = workflow["tasks"][:1]
    task = workflow["tasks"][0]
    task["environment"].pop("ASTRA_DEMO_PILOT", None)
    task["environment"].update(
        PAYLOAD_SHA256=identity["payload_sha256"],
        ASTRA_SOURCE_REVISION=revision,
        ASTRA_ENTRY_MODULE="astra_reversal.osmo.reasoning_policy_learning",
        ASTRA_LEARNING_PHASE=phase,
        ASTRA_WORKER_GPU_HOURS=str(gpu_hours),
        ASTRA_LEARNING_TASK_INDEX=str(task_index),
        ASTRA_LEARNING_SEED_INDEX=str(seed_index),
        ASTRA_LEARNING_PROTOCOL_VERSION=protocol_version,
        ASTRA_RL_RECIPE=rl_recipe,
    )
    if phase == "baseline-preflight":
        task["environment"].update(
            ASTRA_INCLUDE_RLINF="1",
            ASTRA_ENTRY_MODULE="astra_reversal.osmo.reasoning_baseline_preflight",
        )
    if collection_limit is not None:
        task["environment"]["ASTRA_COLLECTION_LIMIT"] = str(collection_limit)
    if phase in ("dsrl", "ppo"):
        task["environment"].update(
            ASTRA_INCLUDE_RLINF="1",
            ASTRA_ENTRY_MODULE="astra_reversal.osmo.reasoning_rl_baseline",
            ASTRA_LEARNING_METHOD=phase,
        )
    if phase == "guidance-probe":
        task["environment"].update(
            ASTRA_GUIDANCE_PROBE="1",
            ASTRA_ENTRY_MODULE="astra_reversal.osmo.reasoning_baseline_preflight",
        )
    if phase in ("replay-learning", "replay-masked"):
        task["environment"]["ASTRA_ENTRY_MODULE"] = (
            "astra_reversal.osmo.reasoning_replay_learning"
        )
    if phase == "replay-credit":
        task["environment"]["ASTRA_ENTRY_MODULE"] = (
            "astra_reversal.osmo.reasoning_credit_learning"
        )
    if phase == "checkpoint-evaluation":
        task["environment"]["ASTRA_ENTRY_MODULE"] = (
            "astra_reversal.osmo.reasoning_checkpoint_evaluation"
        )
    task["files"] = [
        {"localpath": str(bundle / "bootstrap.sh"), "path": "/tmp/astra-bootstrap.sh"},
        {"localpath": str(token), "path": "/tmp/astra-relay.token"},
    ]
    (destination / "workflow.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    plan = {
        "status": "prepared_not_submitted",
        "phase": phase,
        "pool": "groot-l40s-01",
        "gpu_count": 1,
        "allocated_gpu_hours": gpu_hours,
        "task_index": task_index,
        "seed_index": seed_index,
        "protocol_version": protocol_version,
        "source_revision": revision,
        "payload_sha256": identity["payload_sha256"],
        "automatic_experiment_retries": False,
        "baselines_included": phase in ("dsrl", "ppo"),
        "rl_recipe": rl_recipe if phase in ("dsrl", "ppo") else None,
        "collection_limit": collection_limit,
    }
    (destination / "launch_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (destination / "connect.sh").write_text(
        connection_script(
            repo, bundle, destination, phase=phase, gpu_hours=gpu_hours, port=port
        )
    )
    (destination / "connect.sh").chmod(0o700)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--phase",
        choices=(
            "preflight",
            "pilot",
            "baseline-preflight",
            "dsrl",
            "guidance-probe",
            "ppo",
            "replay-learning",
            "replay-masked",
            "replay-credit",
            "checkpoint-evaluation",
        ),
        default="preflight",
    )
    parser.add_argument("--gpu-hours", type=float, required=True)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument(
        "--protocol-version",
        choices=("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"),
        default="v1",
    )
    parser.add_argument("--port", type=int, default=19943)
    parser.add_argument("--collection-limit", type=int)
    parser.add_argument(
        "--rl-recipe", choices=("standard", "more_reuse"), default="standard"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.bundle,
                args.destination,
                phase=args.phase,
                gpu_hours=args.gpu_hours,
                task_index=args.task_index,
                seed_index=args.seed_index,
                protocol_version=args.protocol_version,
                port=args.port,
                rl_recipe=args.rl_recipe,
                collection_limit=args.collection_limit,
            )
        )
    )


if __name__ == "__main__":
    main()
