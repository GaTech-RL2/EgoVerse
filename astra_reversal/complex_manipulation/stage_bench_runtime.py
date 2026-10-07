"""Build Bench2Dex's frozen policy environment on a GPU-zero OSMO worker."""

import argparse
import json
import os
import re
import subprocess
import tarfile
import time
from pathlib import Path

from stage_robocasa import archive_client, clone, sha256


def run(command, *, cwd=None, environment=None):
    print(
        json.dumps({"command": list(map(str, command)), "time": time.time()}),
        flush=True,
    )
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    workflow = os.environ["ASTRA_RUN_ID"]
    if not re.fullmatch(r"astra-complex-20261006-bench-runtime-stage-[0-9]+", workflow):
        raise ValueError("Unexpected runtime staging identity")
    manifest = json.loads(args.manifest.read_text())
    root = Path("/opt/astra-bench-runtime")
    root.mkdir(exist_ok=False)
    prefix = f"experiments/astra-complex-20261006/{workflow}"
    client = archive_client()
    if client.list_objects_v2(Bucket="rldb", Prefix=prefix + "/", MaxKeys=1).get(
        "KeyCount"
    ):
        raise FileExistsError("Do not overwrite an existing runtime stage")
    source = root / "Bench2Dex"
    clone(manifest["sources"]["bench2dex"], source)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("R2_")}
    environment.pop("VIRTUAL_ENV", None)
    environment.update(
        UV_PYTHON_INSTALL_DIR=str(root / "python"),
        UV_LINK_MODE="copy",
        UV_NO_PROGRESS="1",
        GIT_LFS_SKIP_SMUDGE="1",
    )
    project = source / "policy/pi05"
    run(["uv", "python", "install", "3.11"], environment=environment)
    run(
        ["uv", "sync", "--frozen", "--no-dev", "--python", "3.11"],
        cwd=project,
        environment=environment,
    )
    python = project / ".venv/bin/python"
    probe = """
import importlib.metadata as m,json,sys
import jax
from openpi.training import config
from openpi.policies import policy_config
c=config.get_config('pi05_base_dex2bench_full')
print(json.dumps({'python':sys.version,'versions':{n:m.version(n) for n in ['jax','jaxlib','numpy','flax','orbax-checkpoint','torch','lerobot']},'model_type':str(c.model.model_type),'devices':[str(d) for d in jax.devices()],'note':'Imports only; no checkpoint loaded and no inference or simulation run.'}))
"""
    observed = subprocess.check_output(
        [str(python), "-c", probe], cwd=project, env=environment, text=True
    )
    (root / "import_probe.json").write_text(observed)
    frozen = subprocess.check_output(
        ["uv", "pip", "freeze", "--python", str(python)], env=environment, text=True
    )
    (root / "requirements-frozen.txt").write_text(frozen)
    receipt = {
        "workflow": workflow,
        "status": "policy_runtime_imports_validated_gpu_inference_not_run",
        "gpu_count": 0,
        "source": manifest["sources"]["bench2dex"],
        "root": str(root),
        "python": str(python),
        "uv_lock_sha256": sha256(project / "uv.lock"),
        "requirements_sha256": sha256(root / "requirements-frozen.txt"),
        "probe": json.loads(observed.strip().splitlines()[-1]),
        "script_sha256": sha256(Path(__file__)),
    }
    archive = root.parent / "astra-bench-policy-runtime.tar.gz"

    def included(entry):
        return (
            None
            if any(p in {".git", ".cache"} for p in Path(entry.name).parts)
            else entry
        )

    with tarfile.open(archive, "w:gz", compresslevel=1) as bundle:
        for name in (
            "Bench2Dex",
            "python",
            "import_probe.json",
            "requirements-frozen.txt",
        ):
            bundle.add(root / name, arcname=name, filter=included)
    receipt["archive"] = {
        "key": prefix + "/runtime.tar.gz",
        "bytes": archive.stat().st_size,
        "sha256": sha256(archive),
    }
    client.upload_file(str(archive), "rldb", receipt["archive"]["key"])
    client.put_object(
        Bucket="rldb",
        Key=prefix + "/receipt.json",
        Body=json.dumps(receipt, indent=2).encode(),
        ContentType="application/json",
    )
    print(
        json.dumps(
            {"status": receipt["status"], "archive_bytes": archive.stat().st_size}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
