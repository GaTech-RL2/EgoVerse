"""Offline normalization-statistics computation on Modal CPUs — sharded edition.

Architecture
------------
* A coordinator container (run_norm_stats) clones the repo, loads the data
  config, queries the SQL episode table, splits episodes into N_SHARDS chunks,
  and fans out to compute_shard_stats via .map().
* Each shard container reads its episodes directly from the zarr volume,
  accumulates Welford online mean/std (Chan's parallel formula), running
  min/max, and per-dimension t-digest sketches for quantiles.
* The coordinator merges all shard results (exact mean/std via Chan's,
  exact min/max, approximate quantiles via merged t-digests) and writes
  norm_stats.json.

~300 containers × own FUSE daemon → ~300× throughput vs a single container.
Each shard reads only SAMPLES_PER_SHARD frames per key then exits early.

Usage:
    modal run --detach --env robotics egomimic/modal/offline_norm_stats.py \\
        -- mecka_all_zarr [--n_shards 300] [--samples_per_shard 700]
                          [--exclude_hashes_file /path/to/failures.jsonl]

In training, point at the result with:
    norm_stats.precomputed_norm_path=precomputed_norm_stats/<data_config>
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import modal

os.environ.setdefault("MODAL_ENVIRONMENT", "robotics")

# ---------------------------------------------------------------------------
# Inline config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class _Config:
    remote_repo_dir: str = "/root/EgoVerse"
    zarr_volume_name: str = field(
        default_factory=lambda: os.environ.get("MODAL_ZARR_VOLUME", "mecka_data_v2")
    )
    volume_mount_path: str = "/mnt/zarr-data"
    output_mount_path: str = "/root/EgoVerse/logs"
    secret_names: list[str] = field(
        default_factory=lambda: ["egoverse-r2", "egoverse-mongodb", "egoverse-db", "egoverse-sql"]
    )


CFG = _Config()

# ---------------------------------------------------------------------------
# Image — add tdigest for mergeable quantile sketches
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime",
        add_python="3.10",
    )
    .apt_install("git", "curl", "build-essential")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:$PATH"})
    .pip_install(
        "lightning",
        "hydra-core",
        "omegaconf",
        "wandb",
        "boto3",
        "cloudpathlib",
        "zarr==3.1.5",
        "pyarrow",
        "simplejpeg",
        "h5py",
        "av==12.0.0",
        "mediapy",
        "datasets==4.0.0",
        "transformers==4.57.3",
        "timm",
        "einops",
        "positional-encodings[pytorch]",
        "pytorch-kinematics",
        "arm-pytorch-utilities",
        "geomloss",
        "tslearn",
        "scipy",
        "hydra-submitit-launcher==1.2.0",
        "submitit",
        "opencv-python-headless",
        "projectaria-tools",
        "pyquaternion",
        "sqlalchemy",
        "psycopg[binary]",
        "pandas",
        "rich",
        "tabulate",
        "prettytable",
        "packaging",
        "overrides",
        "typing_extensions",
        "pyyaml",
        "matplotlib",
        "termcolor",
        "tqdm",
        "filelock",
        "imageio",
        "imageio-ffmpeg",
        "safetensors",
        "huggingface-hub",
        "scaleapi",
        "openai",
        "pyzmq",
        "torchvision==0.21.0",
        "s5cmd",
        "tdigest==0.5.2.1",
    )
)

zarr_volume = modal.Volume.from_name(CFG.zarr_volume_name)
training_outputs_volume = modal.Volume.from_name(
    "egoverse-training-outputs", create_if_missing=True
)
app = modal.App("egomimic-norm-stats", image=image)

_TIMEOUT = 3 * 3600
_NORM_SUBDIR = "precomputed_norm_stats"

# ---------------------------------------------------------------------------
# Container helpers
# ---------------------------------------------------------------------------


def _ssh_to_https(url: str) -> str:
    if url.startswith("git@github.com:"):
        path = url[len("git@github.com:"):]
        return f"https://github.com/{path}"
    return url


def _prepare_repo(git_remote: str, git_commit: str) -> None:
    clone_url = _ssh_to_https(git_remote)
    repo_dir = Path(CFG.remote_repo_dir)

    if (repo_dir / ".git").exists():
        subprocess.run(
            ["git", "-C", CFG.remote_repo_dir, "fetch", "--all", "--tags"], check=True
        )
    elif repo_dir.exists():
        subprocess.run(["git", "init", CFG.remote_repo_dir], check=True)
        subprocess.run(
            ["git", "-C", CFG.remote_repo_dir, "remote", "add", "origin", clone_url],
            check=True,
        )
        subprocess.run(
            ["git", "-C", CFG.remote_repo_dir, "fetch", "origin", "--tags"], check=True
        )
    else:
        subprocess.run(["git", "clone", clone_url, CFG.remote_repo_dir], check=True)

    subprocess.run(
        ["git", "-C", CFG.remote_repo_dir, "checkout", git_commit], check=True
    )
    subprocess.run(
        ["git", "-C", CFG.remote_repo_dir, "submodule", "update", "--init", "--recursive"],
        check=True,
    )
    subprocess.run(
        ["uv", "pip", "install", "--system", "-e", ".", "--no-deps"],
        cwd=CFG.remote_repo_dir,
        check=True,
    )


# ---------------------------------------------------------------------------
# Shard worker — no repo clone, reads zarr directly
# ---------------------------------------------------------------------------


@app.function(
    cpu=2,
    memory=4 * 1024,
    timeout=1800,
    volumes={CFG.volume_mount_path: zarr_volume},
)
def compute_shard_stats(
    shard_id: int,
    episodes: list[dict],
    norm_key_map: dict,
    samples_per_shard: int,
) -> dict:
    """Per-shard stats: Welford online mean/std, running min/max, t-digest quantiles.

    episodes: [{"episode_hash", "local_path", "num_frames"}, ...]
    norm_key_map: {key_name: {"zarr_key": str}}
    Returns serialized per-key stats dict.
    """
    import random
    import numpy as np
    import zarr
    from tdigest import TDigest

    accum: dict = {}
    samples_collected = {k: 0 for k in norm_key_map}

    episodes = list(episodes)
    random.shuffle(episodes)

    for ep in episodes:
        if all(c >= samples_per_shard for c in samples_collected.values()):
            break

        try:
            store = zarr.open_group(ep["local_path"], mode="r")
        except Exception as e:
            print(f"[Shard {shard_id}] Failed to open {ep['local_path']}: {e}")
            continue

        for key_name, key_info in norm_key_map.items():
            need = samples_per_shard - samples_collected[key_name]
            if need <= 0:
                continue

            zarr_key = key_info["zarr_key"]
            try:
                raw = store[zarr_key][:]  # (n_frames, *shape)
            except Exception:
                continue

            raw = np.asarray(raw, dtype=np.float64)
            n_frames = raw.shape[0]
            if n_frames == 0:
                continue
            raw = raw.reshape(n_frames, -1)
            n_take = min(need, n_frames)
            idx = (
                random.sample(range(n_frames), n_take)
                if n_take < n_frames
                else list(range(n_frames))
            )
            frames = raw[idx]  # (n_take, n_dims)
            n_b, n_dims = frames.shape

            if key_name not in accum:
                accum[key_name] = {
                    "n": 0,
                    "mean": np.zeros(n_dims),
                    "M2": np.zeros(n_dims),
                    "min": np.full(n_dims, np.inf),
                    "max": np.full(n_dims, -np.inf),
                    "digests": [TDigest() for _ in range(n_dims)],
                }

            a = accum[key_name]
            n_a = a["n"]
            batch_mean = frames.mean(axis=0)
            batch_M2 = ((frames - batch_mean) ** 2).sum(axis=0)

            # Chan's parallel formula
            if n_a == 0:
                a["mean"] = batch_mean.copy()
                a["M2"] = batch_M2.copy()
            else:
                n_c = n_a + n_b
                delta = batch_mean - a["mean"]
                a["mean"] = (n_a * a["mean"] + n_b * batch_mean) / n_c
                a["M2"] = a["M2"] + batch_M2 + delta ** 2 * n_a * n_b / n_c

            a["n"] += n_b
            np.minimum(a["min"], frames.min(axis=0), out=a["min"])
            np.maximum(a["max"], frames.max(axis=0), out=a["max"])

            for dim_i in range(n_dims):
                a["digests"][dim_i].batch_update(frames[:, dim_i].tolist())

            samples_collected[key_name] += n_b

    # Serialize
    out: dict = {}
    for key_name, a in accum.items():
        if a["n"] == 0:
            continue
        out[key_name] = {
            "n": int(a["n"]),
            "mean": a["mean"].tolist(),
            "M2": a["M2"].tolist(),
            "min": a["min"].tolist(),
            "max": a["max"].tolist(),
            # Per-dim: list of {"centroids": [{"m": float, "c": float}], "n": int}
            "tdigests": [a["digests"][i].to_dict() for i in range(len(a["digests"]))],
        }

    print(f"[Shard {shard_id}] collected={samples_collected}")
    return out


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------


def _chan_merge(acc: dict, shard: dict) -> dict:
    """Merge one shard's Welford stats into the running accumulator (in-place)."""
    import numpy as np

    n_a = acc["n"]
    n_b = shard["n"]
    if n_b == 0:
        return acc

    mean_b = np.asarray(shard["mean"])
    M2_b = np.asarray(shard["M2"])
    min_b = np.asarray(shard["min"])
    max_b = np.asarray(shard["max"])

    if n_a == 0:
        acc["mean"] = mean_b.copy()
        acc["M2"] = M2_b.copy()
        acc["min"] = min_b.copy()
        acc["max"] = max_b.copy()
        acc["n"] = n_b
        return acc

    n_c = n_a + n_b
    mean_a = np.asarray(acc["mean"])
    delta = mean_b - mean_a
    acc["mean"] = (n_a * mean_a + n_b * mean_b) / n_c
    acc["M2"] = np.asarray(acc["M2"]) + M2_b + delta ** 2 * n_a * n_b / n_c
    acc["min"] = np.minimum(acc["min"], min_b)
    acc["max"] = np.maximum(acc["max"], max_b)
    acc["n"] = n_c
    return acc


def _merge_tdigests(all_td_dicts: list[dict]):
    """Merge a list of t-digest dicts (from to_dict()) into one TDigest."""
    from tdigest import TDigest

    merged = TDigest()
    for td in all_td_dicts:
        for c in td.get("centroids", []):
            merged.update(c["m"], c["c"])
    return merged


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


@app.function(
    cpu=8,
    memory=32 * 1024,
    timeout=_TIMEOUT,
    allow_preemption=False,
    secrets=[modal.Secret.from_name(name) for name in CFG.secret_names],
    volumes={
        CFG.volume_mount_path: zarr_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)
def run_norm_stats(
    data_config: str,
    git_remote: str,
    git_commit: str,
    n_shards: int = 300,
    samples_per_shard: int = 700,
    exclude_hashes: list[str] | None = None,
) -> str:
    """Fan out to shard workers, merge results, write norm_stats.json."""
    import copy
    import json
    import math
    import sys
    import time

    import numpy as np

    _prepare_repo(git_remote=git_remote, git_commit=git_commit)
    zarr_volume.reload()
    sys.path.insert(0, CFG.remote_repo_dir)

    import hydra
    from omegaconf import OmegaConf, open_dict

    from egomimic.utils.aws.aws_data_utils import load_env
    from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df
    from egomimic.rldb.embodiment.embodiment import get_embodiment_id

    load_env()
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    data_cfg_path = (
        Path(CFG.remote_repo_dir) / "egomimic" / "hydra_configs" / "data" / f"{data_config}.yaml"
    )
    if not data_cfg_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_cfg_path}")
    data_cfg = OmegaConf.load(str(data_cfg_path))

    out_path = Path(CFG.output_mount_path) / _NORM_SUBDIR / data_config / "norm_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Build norm_key_map from config ----
    all_stats_out: dict = {}

    for dataset_name, ds_cfg_raw in data_cfg.train_datasets.items():
        print(f"[NormStats] Processing dataset <{dataset_name}>")

        ds_cfg = copy.deepcopy(ds_cfg_raw)
        with open_dict(ds_cfg):
            if OmegaConf.select(ds_cfg, "resolver.debug", default=None) is not None:
                ds_cfg.resolver.debug = False

        key_map = hydra.utils.instantiate(ds_cfg.resolver.key_map)
        norm_key_map = {
            k: {"zarr_key": v["zarr_key"]}
            for k, v in key_map.items()
            if isinstance(v, dict)
            and v.get("key_type") not in ("camera_keys", "annotation_keys")
            and "zarr_key" in v
        }
        print(f"[NormStats] norm_key_map keys: {list(norm_key_map.keys())}")

        # ---- Query SQL for episode list ----
        engine = create_default_engine()
        df = episode_table_to_df(engine)
        if df.empty:
            raise ValueError("SQL episode table is empty")
        df = df[df["is_deleted"] != True]  # noqa: E712
        if exclude_hashes:
            df = df[~df["episode_hash"].isin(set(exclude_hashes))]
            print(f"[NormStats] After excluding {len(exclude_hashes)} hashes: {len(df)} rows")

        # ---- Find episodes present on local volume ----
        # Single os.listdir() call is orders of magnitude faster than
        # 197K individual is_dir() FUSE stat calls.
        volume_path = Path(CFG.volume_mount_path)
        print(f"[NormStats] Listing volume directory {volume_path} ...")
        import os as _os
        local_names = set(_os.listdir(str(volume_path)))
        print(f"[NormStats] Volume has {len(local_names)} entries")

        episodes: list[dict] = []
        n_missing = 0
        for _, row in df.iterrows():
            h = row["episode_hash"]
            if h in local_names:
                local_path = str(volume_path / h)
            elif f"{h}.zarr" in local_names:
                local_path = str(volume_path / f"{h}.zarr")
            else:
                n_missing += 1
                continue
            episodes.append({
                "episode_hash": h,
                "local_path": local_path,
                "num_frames": int(row["num_frames"]),
            })

        print(f"[NormStats] {len(episodes)} episodes found locally, {n_missing} missing")

        if not episodes:
            raise ValueError("No episodes found on local volume.")

        # ---- Split into shards ----
        actual_shards = min(n_shards, len(episodes))
        shard_size = math.ceil(len(episodes) / actual_shards)
        shards = [
            episodes[i: i + shard_size]
            for i in range(0, len(episodes), shard_size)
        ]
        print(f"[NormStats] {len(shards)} shards × ~{shard_size} episodes each, "
              f"{samples_per_shard} samples/shard/key → "
              f"~{len(shards) * samples_per_shard} total samples/key")

        # ---- Fan out ----
        t_start = time.time()
        shard_inputs = [
            (i, shard, norm_key_map, samples_per_shard)
            for i, shard in enumerate(shards)
        ]
        shard_results = list(
            compute_shard_stats.starmap(shard_inputs)
        )
        elapsed = time.time() - t_start
        print(f"[NormStats] All shards complete in {elapsed:.1f}s")

        # ---- Merge ----
        merged: dict = {k: {"n": 0, "mean": None, "M2": None, "min": None, "max": None} for k in norm_key_map}
        # Per key per dim: collect all shard t-digest dicts
        per_key_dim_tdigests: dict = {k: None for k in norm_key_map}

        for shard_result in shard_results:
            for key_name in norm_key_map:
                if key_name not in shard_result:
                    continue
                sr = shard_result[key_name]
                merged[key_name] = _chan_merge(merged[key_name], sr)

                # Accumulate t-digest dicts per dimension
                if per_key_dim_tdigests[key_name] is None:
                    n_dims = len(sr["tdigests"])
                    per_key_dim_tdigests[key_name] = [[] for _ in range(n_dims)]
                for dim_i, td_dict in enumerate(sr["tdigests"]):
                    per_key_dim_tdigests[key_name][dim_i].append(td_dict)

        # ---- Compute final stats ----
        emb_id = str(get_embodiment_id(dataset_name))
        key_stats: dict = {}

        for key_name in norm_key_map:
            m = merged[key_name]
            if m["n"] == 0 or m["mean"] is None:
                print(f"[NormStats] No data for key {key_name}, skipping")
                continue

            mean = np.asarray(m["mean"], dtype=np.float32)
            std = np.sqrt(np.asarray(m["M2"]) / m["n"]).astype(np.float32)
            min_ = np.asarray(m["min"], dtype=np.float32)
            max_ = np.asarray(m["max"], dtype=np.float32)

            n_dims = len(mean)
            quantile_stats = {
                "median": np.zeros(n_dims, dtype=np.float32),
                "quantile_1": np.zeros(n_dims, dtype=np.float32),
                "quantile_99": np.zeros(n_dims, dtype=np.float32),
                "quantile_0_01": np.zeros(n_dims, dtype=np.float32),
                "quantile_99_99": np.zeros(n_dims, dtype=np.float32),
            }

            if per_key_dim_tdigests[key_name] is not None:
                for dim_i in range(n_dims):
                    td = _merge_tdigests(per_key_dim_tdigests[key_name][dim_i])
                    quantile_stats["median"][dim_i] = td.percentile(50)
                    quantile_stats["quantile_1"][dim_i] = td.percentile(1)
                    quantile_stats["quantile_99"][dim_i] = td.percentile(99)
                    quantile_stats["quantile_0_01"][dim_i] = td.percentile(0.01)
                    quantile_stats["quantile_99_99"][dim_i] = td.percentile(99.99)

            key_stats[key_name] = {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "min": min_.tolist(),
                "max": max_.tolist(),
                **{k: v.tolist() for k, v in quantile_stats.items()},
            }
            print(f"[NormStats] key={key_name} n={m['n']} shape={mean.shape}")

        if not key_stats:
            raise RuntimeError(f"No stats produced for dataset={dataset_name}")

        all_stats_out[emb_id] = key_stats

    payload = {
        "stats": all_stats_out,
        "loading_time": None,
        "computing_time": None,
        "frames": sum(m["n"] for key_name in norm_key_map for m in [merged.get(key_name, {"n": 0})]),
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=4)

    training_outputs_volume.commit()
    print(f"[NormStats] Saved → {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


def _resolve_git_state() -> tuple[str, str, bool]:
    def _git(args):
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()

    git_remote = _git(["git", "config", "--get", "remote.origin.url"])
    git_commit = _git(["git", "rev-parse", "HEAD"])
    is_dirty = bool(_git(["git", "status", "--porcelain"]))

    try:
        subprocess.run(
            ["git", "fetch", "--quiet", "origin"], cwd=REPO_ROOT, check=True, capture_output=True
        )
        result = subprocess.run(
            ["git", "branch", "-r", "--contains", git_commit],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if not result.stdout.strip():
            raise SystemExit(
                f"ERROR: commit {git_commit[:12]} has not been pushed.\n"
                "Push your branch first, then re-run."
            )
    except subprocess.CalledProcessError:
        pass

    return git_remote, git_commit, is_dirty


@app.local_entrypoint()
def main(*args: str) -> None:
    """Compute norm stats via sharded Modal containers with t-digest quantiles."""
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(prog="offline_norm_stats")
    parser.add_argument("data_config", help="Data config name, e.g. mecka_all_zarr")
    parser.add_argument("--n_shards", type=int, default=300, help="Number of parallel shard containers")
    parser.add_argument("--samples_per_shard", type=int, default=700, help="Frames per key per shard")
    parser.add_argument("--exclude_hashes_file", type=str, default=None,
                        help="JSONL file with episode_hash fields to exclude")
    parsed = parser.parse_args(list(args))

    exclude_hashes: list[str] = []
    if parsed.exclude_hashes_file:
        with open(parsed.exclude_hashes_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    exclude_hashes.append(_json.loads(line)["episode_hash"])
        print(f"Loaded {len(exclude_hashes)} hashes to exclude")

    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print("Warning: local repo has uncommitted changes. Modal runs the last committed state.")

    total_samples = parsed.n_shards * parsed.samples_per_shard
    print(
        f"Submitting norm-stats job: data={parsed.data_config!r} "
        f"n_shards={parsed.n_shards} samples_per_shard={parsed.samples_per_shard} "
        f"→ ~{total_samples:,} total samples/key"
        + (f"  exclude_hashes={len(exclude_hashes)}" if exclude_hashes else "")
    )

    out_path = run_norm_stats.remote(
        data_config=parsed.data_config,
        git_remote=git_remote,
        git_commit=git_commit,
        n_shards=parsed.n_shards,
        samples_per_shard=parsed.samples_per_shard,
        exclude_hashes=exclude_hashes or None,
    )

    print(f"\nDone. Volume path: {out_path}")
    print(f"\nTo use in training:\n  norm_stats.precomputed_norm_path={_NORM_SUBDIR}/{parsed.data_config}")
