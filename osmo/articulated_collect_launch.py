"""Package an isolated simulator snapshot and submit CPU-only OSMO collection.

No Git push, shared checkout changes, training task, or global runtime install.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import subprocess
import tarfile
import textwrap
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--embodiment",
        required=True,
        choices=[
            "u_socket",
            "gripper",
            "chain_gripper",
            "umi",
            "suction",
            "triangle",
            "scoop",
            "spring",
            "flipper",
        ],
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--run-id", default="articulated-20260909")
    ap.add_argument("--job-tag", default="")
    ap.add_argument("--pool", default="groot-l40s-03")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--shard-stop", type=int, default=24)
    ap.add_argument("--checkpoint-prefix")
    ap.add_argument("--fast-search", action="store_true")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    if not 0 <= a.shard_start < a.shard_stop <= 24:
        ap.error("invalid shard interval")
    if not re.fullmatch(r"[a-z0-9-]+", a.run_id):
        ap.error("invalid run ID")
    if a.job_tag and not re.fullmatch(r"[a-z0-9-]+", a.job_tag):
        ap.error("invalid job tag")
    if a.checkpoint_prefix and not re.fullmatch(
        r"[A-Za-z0-9_/-]+", a.checkpoint_prefix
    ):
        ap.error("invalid checkpoint prefix")
    repo = Path(__file__).resolve().parents[1]
    filenames = [
        "Tsimulation/__init__.py",
        "Tsimulation/sim_v2/__init__.py",
        "Tsimulation/sim_v2/collect/__init__.py",
        "Tsimulation/sim_v2/collect/contact_controller.py",
        "egomimic/__init__.py",
        "egomimic/rldb/__init__.py",
        "egomimic/rldb/zarr/__init__.py",
        "egomimic/rldb/zarr/zarr_writer.py",
    ]
    filenames += [
        str(p.relative_to(repo))
        for p in (repo / "Tsimulation/sim_v2/pushshapes").glob("*.py")
    ]
    filenames += [
        str(p.relative_to(repo))
        for p in (repo / "Tsimulation/sim_v2/collect").glob("articulation_*.py")
    ]
    capsule = io.BytesIO()
    with tarfile.open(fileobj=capsule, mode="w:gz") as tar:
        for name in sorted(filenames):
            tar.add(repo / name, arcname=name, recursive=False)
    payload = capsule.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    job = a.run_id + "-" + a.embodiment.replace("_", "-")
    if a.shard_start != 0 or a.shard_stop != 24:
        job += f"-s{a.shard_start:03d}-s{a.shard_stop:03d}"
    if a.job_tag:
        job += "-" + a.job_tag
    prefix = "staged/pushshapes_articulated/" + a.run_id + "/" + a.embodiment
    provenance = prefix + "/runs/" + job
    expected = (a.shard_stop - a.shard_start) * 6 * 125
    restore = (
        f"python -m Tsimulation.sim_v2.collect.articulation_restore --prefix {a.checkpoint_prefix} --out /workspace/demos --shard-start {a.shard_start} --shard-stop {a.shard_stop}"
        if a.checkpoint_prefix
        else ""
    )
    entry = textwrap.dedent(f"""\
        #!/bin/bash
        set -euo pipefail
        mkdir -p /workspace/EgoVerse
        base64 -d /tmp/source.b64 > /workspace/source.tar.gz
        echo '{digest}  /workspace/source.tar.gz' | sha256sum -c -
        tar xzf /workspace/source.tar.gz -C /workspace/EgoVerse
        cd /workspace/EgoVerse
        python -m venv emimic
        source emimic/bin/activate
        pip install --disable-pip-version-check --quiet numpy==2.2.6 pymunk==7.3.0 pygame-ce==2.5.5 gymnasium==1.2.0 shapely==2.1.2 zarr==3.1.3 simplejpeg==1.8.2 opencv-python-headless==4.12.0.88 boto3
        export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
        export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
        export ARTICULATED_SOURCE_SHA256={digest}
        export ARTICULATED_FAST_SEARCH={int(a.fast_search)}
        {restore}
        python -u -m Tsimulation.sim_v2.collect.articulation_probe --embodiments {a.embodiment} --seeds 10 --max-steps {a.max_steps} --output /workspace/preflight
        python - <<'PREFLIGHT'
        import json
        from pathlib import Path
        from Tsimulation.sim_v2.collect.articulation_batch import r2_client
        client=r2_client()
        client.put_object(Bucket='rldb',Key='{provenance}/run_manifest.json',
                          Body=json.dumps(dict(source_sha256='{digest}',expected_episodes={expected},shard_start={a.shard_start},shard_stop={a.shard_stop})).encode())
        client.upload_file('/workspace/source.tar.gz','rldb','{provenance}/source.tar.gz')
        receipt=Path('/workspace/demos/restored_checkpoint.json')
        if receipt.exists(): client.upload_file(str(receipt),'rldb','{provenance}/restored_checkpoint.json')
        rows=[json.loads(p.read_text()) for p in Path('/workspace/preflight').glob('*.json')]
        assert len(rows)==10
        assert sum(r['engaged'] for r in rows)>=8, rows
        good=[r for r in rows if r['success']]
        assert len(good)>=1 and all(r['jerk_speed']<.5 for r in good), rows
        print('PREFLIGHT PASSED',len(good),'successful seeds; source {digest}',flush=True)
        PREFLIGHT
        python -u -m Tsimulation.sim_v2.collect.articulation_batch --out /workspace/demos --embodiments {a.embodiment} --target-per-cell 3000 --shards 24 --shard-start {a.shard_start} --shard-stop {a.shard_stop} --workers {a.workers} --max-attempts 20000 --max-steps {a.max_steps} --s3-prefix {prefix}
        python - <<'PERSIST'
        from Tsimulation.sim_v2.collect.articulation_batch import r2_client
        from pathlib import Path
        c=r2_client()
        c.upload_file('/workspace/source.tar.gz','rldb','{provenance}/source.tar.gz')
        for p in Path('/workspace/preflight').glob('*.json'):
            c.upload_file(str(p),'rldb','{provenance}/preflight/'+p.name)
        print('VERIFIED COLLECTION PERSISTED: s3://rldb/{prefix}',flush=True)
        PERSIST
    """)
    workflow = dict(
        workflow=dict(
            name=job,
            tasks=[
                dict(
                    name="collect",
                    image="python:3.12-slim",
                    credentials={
                        "egoverse-aws": {
                            "AWS_ACCESS_KEY_ID": "aws_access_key_id",
                            "AWS_SECRET_ACCESS_KEY": "aws_secret_access_key",
                            "AWS_DEFAULT_REGION": "aws_region",
                        }
                    },
                    command=["bash"],
                    args=["/tmp/entry.sh"],
                    files=[
                        dict(
                            path="/tmp/source.b64",
                            contents=base64.b64encode(payload).decode(),
                        ),
                        dict(path="/tmp/entry.sh", contents=entry),
                    ],
                )
            ],
            resources=dict(
                default=dict(
                    cpu=a.workers + 4,
                    gpu=0,
                    memory="32Gi",
                    storage="200Gi" if a.max_steps > 1200 else "50Gi",
                )
            ),
            timeout=dict(queue_timeout="1d", exec_timeout="2d"),
        )
    )
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output / (job + ".yaml")
    path.write_text(yaml.safe_dump(workflow, sort_keys=False, width=120))
    (a.output / (job + ".source.json")).write_text(
        json.dumps(
            dict(
                sha256=digest,
                files=filenames,
                base_commit="f952ca0d",
                destination="s3://rldb/" + prefix,
                provenance="s3://rldb/" + provenance,
                expected_episodes=expected,
                shard_start=a.shard_start,
                shard_stop=a.shard_stop,
                checkpoint_prefix=a.checkpoint_prefix,
                fast_search=a.fast_search,
            ),
            indent=2,
        )
        + "\n"
    )
    if a.submit:
        result = subprocess.run(
            [
                "osmo",
                "workflow",
                "submit",
                str(path),
                "--pool",
                a.pool,
                "--format-type",
                "json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        (a.output / (job + ".submission.json")).write_text(result.stdout)
        print(result.stdout)
    else:
        print(path)


if __name__ == "__main__":
    main()
