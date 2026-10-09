"""Reconstruct reset XML on CPU, with no policy calls or control actions."""

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from astra_reversal.complex_manipulation.worker import Publisher, archive_client, write_json
from astra_reversal.complex_manipulation.xiaomi_worker import restore
from astra_reversal.records import digest


def inspect(output):
    import gymnasium as gym
    import numpy as np
    import robocasa  # noqa: F401

    rows = []
    for task in ("LoadPreparedFood", "PackIdenticalLunches"):
        for seed in (0, 1, 2):
            np.random.seed(seed)
            env = gym.make(f"robocasa/{task}", split="pretrain", seed=seed, disable_env_checker=True)
            try:
                obs, _ = env.reset(seed=seed)
                raw = env.unwrapped.env
                xml = raw.sim.model.get_xml().encode()
                with gzip.open(output / f"{task}_seed{seed}.xml.gz", "wb") as stream:
                    stream.write(xml)
                rows.append({"task": task, "seed": seed, "model_sha256": hashlib.sha256(xml).hexdigest(),
                             "state_sha256": digest(raw.sim.get_state().flatten()),
                             "instruction": obs["annotation.human.task_description"],
                             "setup_resets": 1, "explicit_audit_resets": 1,
                             "policy_calls": 0, "control_actions": 0, "policy_rollouts": 0})
                write_json(output / "reset_audit.json", rows)
                print(json.dumps(rows[-1]), flush=True)
            finally:
                env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inspect", action="store_true")
    a = p.parse_args()
    output = Path("/opt/astra-audit-results")
    if a.inspect:
        inspect(output)
        return
    workflow = os.environ["ASTRA_RUN_ID"]
    if not workflow.startswith("astra-complex-20261006-robocasa-xiaomi-reset-audit-"):
        raise ValueError("Unexpected reset-audit identity")
    output.mkdir(parents=True, exist_ok=False)
    client = archive_client()
    prefix = f"experiments/astra-complex-20261006/{workflow}/results"
    if client.list_objects_v2(Bucket="rldb", Prefix=prefix + "/", MaxKeys=1).get("KeyCount"):
        raise FileExistsError("Audit archive already exists")
    publisher = Publisher(client, output, prefix)
    write_json(output / "scope.json", {"gpu_count": 0, "policy_rollouts": 0,
        "reason": "Reconstruct six initial model XML documents to diagnose reset fingerprint differences",
        "source_revision": os.environ["ASTRA_SOURCE_REVISION"]})
    try:
        restore(client, "astra-complex-20261006-robocasa-xiaomi-stage-1",
                Path("/tmp/xiaomi_manifest.json"), Path("/opt/astra-xiaomi"), include_weights=False)
        env = {k: v for k, v in os.environ.items() if not k.startswith("R2_")}
        with (output / "inspection.log").open("w") as log:
            subprocess.run(["/opt/astra-xiaomi/runtime/bin/python", "-u", "-m",
                "astra_reversal.complex_manipulation.xiaomi_reset_audit", "--inspect"],
                env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
    finally:
        publisher.publish()


if __name__ == "__main__":
    main()
