from __future__ import annotations

import json
import os
from pathlib import Path

import modal

app = modal.App("egoverse-foldclothes-hpt")

volume = modal.Volume.from_name("egoverse-hackathon-data", create_if_missing=True)

REPO = "/root/egoverse"
DATA_ROOT = "/data/foldclothes-zarr"
OUT_ROOT = "/data/output/foldclothes-hpt"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "wget", "ca-certificates")
    .run_commands(
        "wget -q https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz",
        "tar -xzf s5cmd_2.2.2_Linux-64bit.tar.gz",
        "mv s5cmd /usr/local/bin/s5cmd",
        "rm -f s5cmd_2.2.2_Linux-64bit.tar.gz",
    )
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "lightning",
        "hydra-core",
        "omegaconf",
        "zarr==3.1.5",
        "pandas==2.2.3",
        "pyarrow==18.1.0",
        "numpy==1.26.4",
        "boto3",
        "botocore",
        "cloudpathlib",
        "einops",
        "timm",
        "transformers==4.57.3",
        "opencv-python-headless==4.10.0.84",
        "termcolor",
        "tabulate",
        "rich",
        "pyyaml",
        "simplejpeg",
        "positional-encodings[pytorch]",
        "scipy",
        "scikit-learn",
        "tqdm",
        "geomloss",
        "overrides",
        "tslearn",
        "sqlalchemy",
        "huggingface_hub",
        "scaleapi",
        "matplotlib",
        "requests",
        "projectaria-tools==2.0.0",
    )
    .add_local_dir("egomimic", remote_path=f"{REPO}/egomimic")
    .add_local_dir(
        "artifacts/foldclothes-v1/manifests",
        remote_path=f"{REPO}/artifacts/foldclothes-v1/manifests",
    )
)

RUNS = ("random-774", "duration-balanced-774", "diversity-774")
MANIFEST_ROOT = Path(REPO) / "artifacts/foldclothes-v1/manifests"


def _manifest_for(run_id: str, smoke: bool) -> Path:
    if smoke:
        return MANIFEST_ROOT / "smoke_train.csv"
    return MANIFEST_ROOT / "curation_variants" / f"{run_id}.csv"


def _val_manifest(smoke: bool) -> Path:
    return MANIFEST_ROOT / ("smoke_val.csv" if smoke else "val.csv")


def _test_manifest(smoke: bool) -> Path:
    return MANIFEST_ROOT / ("smoke_val.csv" if smoke else "test.csv")


def _union_manifests(smoke: bool) -> list[tuple[str, str]]:
    import pandas as pd

    paths = [_val_manifest(smoke), _test_manifest(smoke)]
    if smoke:
        paths.append(MANIFEST_ROOT / "smoke_train.csv")
    else:
        paths.extend(MANIFEST_ROOT / "curation_variants" / f"{name}.csv" for name in RUNS)

    frames = [pd.read_csv(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("episode_hash")
    return list(
        zip(
            df["zarr_processed_path"].astype(str).tolist(),
            df["episode_hash"].astype(str).tolist(),
        )
    )


def _run_hydra(overrides: list[str]) -> None:
    import subprocess
    import sys

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}"
    cmd = [
        sys.executable,
        f"{REPO}/egomimic/trainHydra.py",
        "--config-name=train_foldclothes_hpt",
        *overrides,
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


def _last_metric(output_dir: str, column: str):
    import pandas as pd

    json_path = Path(output_dir) / "val_metrics.json"
    if json_path.is_file():
        payload = json.loads(json_path.read_text())
        if column in payload:
            return float(payload[column])

    matches = list(Path(output_dir).rglob("metrics.csv"))
    if not matches:
        return None
    frame = pd.read_csv(matches[0])
    if column not in frame.columns:
        return None
    series = frame[column].dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def _best_checkpoint(output_dir: str) -> str:
    ckpt_dir = Path(output_dir) / "checkpoints"
    last = ckpt_dir / "last.ckpt"
    if last.is_file():
        return str(last)
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        return ""
    return str(ckpts[-1])


def _sync_zarrs(s3_paths: list[tuple[str, str]], local_dir: Path) -> int:
    import subprocess
    import tempfile

    local_dir.mkdir(parents=True, exist_ok=True)
    to_sync = []
    for processed_path, episode_hash in s3_paths:
        dest = local_dir / episode_hash
        if dest.is_dir() and any(dest.iterdir()):
            continue
        src = processed_path.rstrip("/") + "/*"
        to_sync.append(f'sync "{src}" "{dest}/"')

    if not to_sync:
        print("All requested zarrs already present.", flush=True)
        return 0
    print(f"Downloading {len(to_sync)} zarrs with s5cmd.", flush=True)

    endpoint = os.environ["R2_ENDPOINT_URL"]
    env = os.environ.copy()
    env["AWS_ACCESS_KEY_ID"] = os.environ["R2_ACCESS_KEY_ID"]
    env["AWS_SECRET_ACCESS_KEY"] = os.environ["R2_SECRET_ACCESS_KEY"]
    env["AWS_DEFAULT_REGION"] = "auto"
    env["AWS_REGION"] = "auto"

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("\n".join(to_sync) + "\n")
        batch_path = handle.name
    try:
        subprocess.run(
            [
                "s5cmd",
                "--log",
                "error",
                "--endpoint-url",
                endpoint,
                "--numworkers",
                "16",
                "run",
                batch_path,
            ],
            check=True,
            env=env,
        )
    finally:
        os.remove(batch_path)
    return len(to_sync)


@app.function(
    image=image,
    timeout=4 * 60 * 60,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("egoverse-r2")],
)
def sync_foldclothes_zarrs(smoke: bool = False) -> dict:
    os.makedirs(DATA_ROOT, exist_ok=True)
    paths = _union_manifests(smoke)
    downloaded = _sync_zarrs(paths, Path(DATA_ROOT))
    volume.commit()
    return {
        "status": "synced",
        "episodes": len(paths),
        "downloaded": downloaded,
        "smoke": smoke,
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=6 * 60 * 60,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("egoverse-r2")],
)
def train_foldclothes_run(run_id: str, smoke: bool = False) -> dict:
    import sys
    import time

    sys.path.insert(0, REPO)
    os.chdir(REPO)

    os.environ["FOLDCLOTHES_TRAIN_MANIFEST"] = str(_manifest_for(run_id, smoke))
    os.environ["FOLDCLOTHES_VAL_MANIFEST"] = str(_val_manifest(smoke))
    run_dir = f"{OUT_ROOT}/{'smoke' if smoke else 'v1'}/{run_id}"
    os.environ["FOLDCLOTHES_OUTPUT_DIR"] = run_dir
    os.makedirs(run_dir, exist_ok=True)

    overrides = [
        f"run_id={run_id}",
        f"paths.dataset_dir={DATA_ROOT}",
        f"hydra.run.dir={run_dir}/train",
    ]
    if smoke:
        overrides.extend(
            [
                "trainer.max_steps=4",
                "trainer.val_check_interval=2",
                "trainer.limit_val_batches=2",
                "trainer.limit_train_batches=2",
                "callbacks.model_checkpoint.every_n_train_steps=2",
                "norm_stats.sample_frac=1.0",
                "model.scheduler.T_max=4",
            ]
        )

    started = time.time()
    _run_hydra(overrides)
    train_dir = f"{run_dir}/train"
    best_path = _best_checkpoint(train_dir)
    matches = list(Path(train_dir).rglob("norm_stats.json"))
    norm_path = str(matches[0]) if matches else ""

    os.environ["FOLDCLOTHES_VAL_MANIFEST"] = str(_test_manifest(smoke))
    eval_overrides = [
        f"run_id={run_id}",
        "mode=eval",
        f"paths.dataset_dir={DATA_ROOT}",
        f"hydra.run.dir={run_dir}/test",
        f"ckpt_path={best_path}",
        "evaluator.limit_val_batches=80" if not smoke else "evaluator.limit_val_batches=2",
    ]
    if norm_path:
        eval_overrides.append(f"norm_stats.precomputed_norm_path={norm_path}")
    _run_hydra(eval_overrides)

    result = {
        "run_id": run_id,
        "smoke": smoke,
        "val_action_loss": _last_metric(train_dir, "Valid/action_loss"),
        "test_action_loss": _last_metric(f"{run_dir}/test", "Valid/action_loss"),
        "selected_checkpoint": best_path,
        "wall_clock_s": round(time.time() - started, 1),
        "output_dir": run_dir,
    }
    result_path = Path(run_dir) / "run_result.json"
    result_path.write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(
    image=image,
    timeout=20 * 60 * 60,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("egoverse-r2")],
)
def run_experiment(smoke: bool = False) -> dict:
    print(f"Syncing foldclothes zarrs (smoke={smoke})", flush=True)
    sync_info = sync_foldclothes_zarrs.remote(smoke=smoke)
    print(sync_info, flush=True)

    results = []
    run_ids = ["random-774"] if smoke else list(RUNS)
    for run_id in run_ids:
        print(f"Training {run_id}", flush=True)
        result = train_foldclothes_run.remote(run_id, smoke)
        print(result, flush=True)
        results.append(result)

    ledger = {
        "experiment": "foldclothes-v1-curation",
        "model": "hpt_bc_flow_human",
        "smoke": smoke,
        "sync": sync_info,
        "runs": results,
    }
    out = Path(OUT_ROOT) / ("smoke" if smoke else "v1") / "training_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ledger, indent=2))
    volume.commit()
    return ledger


@app.local_entrypoint()
def main(smoke: bool = False, detach: bool = True):
    if smoke or not detach:
        ledger = run_experiment.remote(smoke=smoke)
        out = Path("artifacts/foldclothes-v1/training_results.json")
        out.write_text(json.dumps(ledger, indent=2))
        print(f"Wrote {out}")
        print(json.dumps(ledger, indent=2))
        return

    call = run_experiment.spawn(smoke=False)
    print(f"Detached foldclothes experiment: {call.object_id}")
    print("It will keep running on Modal if this local process exits.")
    print("Poll later with: python -c \"import modal; print(modal.FunctionCall.from_id('%s').get())\"" % call.object_id)
