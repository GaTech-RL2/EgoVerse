"""Offline normalization-statistics computation on Modal CPUs — sharded edition.

Architecture
------------
* A coordinator container (run_norm_stats) clones the repo, loads the data
  config, queries the SQL episode table, splits episodes into N_SHARDS chunks,
  and fans out to compute_shard_stats via .starmap().
* Each shard container reads episodes from the zarr volume, applies the same
  transform pipeline as training (head-frame coordinate transform + pause
  removal + pose interpolation + XYZWXYZ→XYZYPR + left/right concat), then
  accumulates Welford online mean/std (Chan's parallel formula), running
  min/max, and per-dimension t-digest sketches for quantiles on the
  post-transform keys.
* Stats are produced for the DataSchematic key names used by training:
    - "ee_pose"           → shape (12,)      (obs: left+right in head frame, XYZYPR)
    - "actions_cartesian" → mean/std/min/max shape (100, 12);
                            quantiles shape (12,) pooled over 100 steps
  under embodiment ID "9" (mecka_bimanual).
* The coordinator merges all shard results and writes norm_stats.json in the
  format expected by DataSchematic.infer_norm_from_dataset() and _check_bounds().

Output path on egoverse-training-outputs volume:
    precomputed_norm_stats/<data_config>/norm_stats.json

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
# Post-transform key names (must match DataSchematic default.yaml: mecka_bimanual)
# ---------------------------------------------------------------------------
_OBS_KEY = "observations.state.ee_pose"   # zarr_key for DataSchematic key "ee_pose"
_ACT_KEY = "actions_cartesian"             # zarr_key == key_name in DataSchematic

# Raw zarr keys read from each episode
_ZK_RIGHT_EE = "right.obs_ee_pose"
_ZK_LEFT_EE = "left.obs_ee_pose"
_ZK_HEAD = "obs_head_pose"

# Transform params (must match training config)
_ACTION_HORIZON = 30    # raw frames read per action chunk
_CHUNK_LEN = 100        # interpolated action steps after InterpolatePose
_OBS_DIM = 12           # left(6) + right(6) after XYZWXYZ→XYZYPR
_ACT_DIM = 12           # same
_PAUSE_EPSILON = 0.005  # PauseRemovalTransform epsilon for mecka

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


def _prepare_repo(git_remote: str, git_commit: str, recurse_submodules: bool = True) -> None:
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
        subprocess.run(
            ["git", "clone", "--no-recurse-submodules", clone_url, CFG.remote_repo_dir],
            check=True,
        )

    subprocess.run(
        ["git", "-C", CFG.remote_repo_dir, "checkout", git_commit], check=True
    )
    if recurse_submodules:
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
# Shard worker — applies transforms, accumulates post-transform stats
# ---------------------------------------------------------------------------


@app.function(
    cpu=2,
    memory=4 * 1024,
    timeout=7200,  # 2h — transform pipeline adds ~1h compute on top of setup
    volumes={CFG.volume_mount_path: zarr_volume},
)
def compute_shard_stats(
    shard_id: int,
    episodes: list[dict],
    samples_per_shard: int,
    git_remote: str = "",
    git_commit: str = "",
) -> dict:
    """Per-shard stats on post-transform data.

    Applies the mecka_bimanual cartesian transform pipeline to each sampled
    frame and accumulates Welford + t-digest stats for:
      - "ee_pose"           (12,)       obs in head frame, XYZYPR
      - "actions_cartesian" (100, 12)   action chunk in head frame, XYZYPR
                            (quantiles: (12,) pooled over 100 steps)

    episodes: [{"episode_hash", "local_path", "num_frames"}, ...]
    Returns serialized per-key stats dict.
    """
    import random
    import numpy as np
    import zarr
    from tdigest import TDigest

    # Install repo so egomimic is importable; skip submodules (openpi etc.) — not needed here
    if git_remote and git_commit:
        _prepare_repo(git_remote=git_remote, git_commit=git_commit, recurse_submodules=False)

    # Build transform pipeline matching training (mecka_bimanual, cartesian mode)
    import sys as _sys
    _sys.path.insert(0, CFG.remote_repo_dir)
    from egomimic.rldb.embodiment.human import Mecka
    transform_list = Mecka.get_transform_list(
        mode="cartesian",
        pause_removal_epsilon=_PAUSE_EPSILON,
    )

    # Per-key accumulators
    accum = {
        "ee_pose": {
            "n": 0,
            "mean": np.zeros(_OBS_DIM),
            "M2": np.zeros(_OBS_DIM),
            "min": np.full(_OBS_DIM, np.inf),
            "max": np.full(_OBS_DIM, -np.inf),
            "digests": [TDigest() for _ in range(_OBS_DIM)],
        },
        "actions_cartesian": {
            "n": 0,
            "mean": np.zeros((_CHUNK_LEN, _ACT_DIM)),
            "M2": np.zeros((_CHUNK_LEN, _ACT_DIM)),
            "min": np.full((_CHUNK_LEN, _ACT_DIM), np.inf),
            "max": np.full((_CHUNK_LEN, _ACT_DIM), -np.inf),
            # Quantile digests: 12 per-dim, pooled over all 100 steps
            "digests": [TDigest() for _ in range(_ACT_DIM)],
        },
    }
    n_collected = 0

    episodes = list(episodes)
    random.shuffle(episodes)

    for ep in episodes:
        if n_collected >= samples_per_shard:
            break

        try:
            store = zarr.open_group(ep["local_path"], mode="r")
        except Exception as e:
            print(f"[Shard {shard_id}] Failed to open {ep['local_path']}: {e}")
            continue

        try:
            right_ee = np.asarray(store[_ZK_RIGHT_EE][:], dtype=np.float64)
            left_ee = np.asarray(store[_ZK_LEFT_EE][:], dtype=np.float64)
            head = np.asarray(store[_ZK_HEAD][:], dtype=np.float64)
        except Exception as e:
            print(f"[Shard {shard_id}] Failed to read from {ep['local_path']}: {e}")
            continue

        T = right_ee.shape[0]
        if T <= _ACTION_HORIZON:
            continue

        valid_frames = list(range(T - _ACTION_HORIZON))
        need = samples_per_shard - n_collected
        sample_frames = random.sample(valid_frames, min(need, len(valid_frames)))

        for t in sample_frames:
            data = {
                "right.action_ee_pose": right_ee[t: t + _ACTION_HORIZON],  # (H, 7)
                "left.action_ee_pose":  left_ee[t: t + _ACTION_HORIZON],   # (H, 7)
                "right.obs_ee_pose":    right_ee[t],                        # (7,)
                "left.obs_ee_pose":     left_ee[t],                         # (7,)
                "obs_head_pose":        head[t],                            # (7,)
            }

            try:
                for tf in transform_list:
                    data = tf.transform(data)
            except Exception:
                continue

            ee = np.asarray(data[_OBS_KEY], dtype=np.float64)   # (12,)
            ac = np.asarray(data[_ACT_KEY], dtype=np.float64)    # (100, 12)

            if np.any(~np.isfinite(ee)) or np.any(~np.isfinite(ac)):
                continue

            # --- Welford update: ee_pose (1 observation = 12-dim vector) ---
            a = accum["ee_pose"]
            n = a["n"]
            if n == 0:
                a["mean"] = ee.copy()
                a["M2"][:] = 0.0
                a["min"] = ee.copy()
                a["max"] = ee.copy()
            else:
                delta = ee - a["mean"]
                a["mean"] += delta / (n + 1)
                a["M2"] += delta * (ee - a["mean"])
                np.minimum(a["min"], ee, out=a["min"])
                np.maximum(a["max"], ee, out=a["max"])
            a["n"] += 1
            for d in range(_OBS_DIM):
                a["digests"][d].update(ee[d])

            # --- Welford update: actions_cartesian (1 obs = (100,12) chunk) ---
            a = accum["actions_cartesian"]
            n = a["n"]
            if n == 0:
                a["mean"] = ac.copy()
                a["M2"] = np.zeros_like(ac)
                a["min"] = ac.copy()
                a["max"] = ac.copy()
            else:
                delta = ac - a["mean"]
                a["mean"] += delta / (n + 1)
                a["M2"] += delta * (ac - a["mean"])
                np.minimum(a["min"], ac, out=a["min"])
                np.maximum(a["max"], ac, out=a["max"])
            a["n"] += 1
            # Pooled quantile digests: feed all 100 steps into each dim's digest
            for d in range(_ACT_DIM):
                a["digests"][d].batch_update(ac[:, d].tolist())

            n_collected += 1

    print(f"[Shard {shard_id}] collected={n_collected}")

    # Serialize
    out: dict = {}
    for key, a in accum.items():
        if a["n"] == 0:
            continue
        out[key] = {
            "n": int(a["n"]),
            "mean": a["mean"].tolist(),
            "M2": a["M2"].tolist(),
            "min": a["min"].tolist(),
            "max": a["max"].tolist(),
            "tdigests": [a["digests"][i].to_dict() for i in range(len(a["digests"]))],
        }

    return out


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------


def _chan_merge(acc: dict, shard: dict) -> dict:
    """Merge one shard's Welford stats into the running accumulator (in-place).

    Works for any numpy-compatible shape of mean/M2/min/max.
    """
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
    """Fan out to shard workers, merge results, write norm_stats.json.

    Stats are keyed by DataSchematic key names ("ee_pose", "actions_cartesian")
    under the embodiment ID for mecka_bimanual.
    """
    import copy
    import json
    import math
    import sys
    import time

    import numpy as np

    _prepare_repo(git_remote=git_remote, git_commit=git_commit)
    zarr_volume.reload()
    sys.path.insert(0, CFG.remote_repo_dir)

    from egomimic.utils.aws.aws_data_utils import load_env
    from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df
    from egomimic.rldb.embodiment.embodiment import get_embodiment_id

    load_env()

    out_path = Path(CFG.output_mount_path) / _NORM_SUBDIR / data_config / "norm_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_stats_out: dict = {}

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
    print(
        f"[NormStats] {len(shards)} shards × ~{shard_size} episodes each, "
        f"{samples_per_shard} samples/shard → "
        f"~{len(shards) * samples_per_shard} total frames sampled"
    )
    print(
        f"[NormStats] Stats produced for: ee_pose ({_OBS_DIM},), "
        f"actions_cartesian ({_CHUNK_LEN}, {_ACT_DIM}) "
        f"[quantiles: ({_ACT_DIM},) pooled over {_CHUNK_LEN} steps]"
    )

    # ---- Fan out ----
    t_start = time.time()
    shard_inputs = [
        (i, shard, samples_per_shard, git_remote, git_commit)
        for i, shard in enumerate(shards)
    ]
    shard_results = list(
        compute_shard_stats.starmap(shard_inputs)
    )
    elapsed = time.time() - t_start
    print(f"[NormStats] All shards complete in {elapsed:.1f}s")

    # ---- Merge ----
    # Keys we track: "ee_pose" and "actions_cartesian"
    tracked_keys = ["ee_pose", "actions_cartesian"]
    merged: dict = {k: {"n": 0, "mean": None, "M2": None, "min": None, "max": None}
                    for k in tracked_keys}
    per_key_dim_tdigests: dict = {k: None for k in tracked_keys}

    for shard_result in shard_results:
        for key in tracked_keys:
            if key not in shard_result:
                continue
            sr = shard_result[key]
            merged[key] = _chan_merge(merged[key], sr)

            n_dims = len(sr["tdigests"])
            if per_key_dim_tdigests[key] is None:
                per_key_dim_tdigests[key] = [[] for _ in range(n_dims)]
            for dim_i, td_dict in enumerate(sr["tdigests"]):
                per_key_dim_tdigests[key][dim_i].append(td_dict)

    # ---- Compute final stats ----
    # Embodiment ID for mecka_bimanual (from the data config dataset name)
    # We hard-code the dataset name since this script targets mecka_all_zarr.
    emb_id = str(get_embodiment_id("mecka_bimanual"))
    key_stats: dict = {}

    for key in tracked_keys:
        m = merged[key]
        if m["n"] == 0 or m["mean"] is None:
            print(f"[NormStats] No data for key {key}, skipping")
            continue

        mean = np.asarray(m["mean"], dtype=np.float32)
        std = np.sqrt(np.maximum(np.asarray(m["M2"]) / m["n"], 0.0)).astype(np.float32)
        min_ = np.asarray(m["min"], dtype=np.float32)
        max_ = np.asarray(m["max"], dtype=np.float32)

        # Quantiles: always shape (_ACT_DIM,) = (12,), computed from pooled t-digests
        n_dims = _ACT_DIM  # 12 for both keys
        quantile_stats = {
            "median":        np.zeros(n_dims, dtype=np.float32),
            "quantile_1":    np.zeros(n_dims, dtype=np.float32),
            "quantile_99":   np.zeros(n_dims, dtype=np.float32),
            "quantile_0_01": np.zeros(n_dims, dtype=np.float32),
            "quantile_99_99": np.zeros(n_dims, dtype=np.float32),
        }

        if per_key_dim_tdigests[key] is not None:
            for dim_i in range(n_dims):
                td = _merge_tdigests(per_key_dim_tdigests[key][dim_i])
                quantile_stats["median"][dim_i]        = td.percentile(50)
                quantile_stats["quantile_1"][dim_i]    = td.percentile(1)
                quantile_stats["quantile_99"][dim_i]   = td.percentile(99)
                quantile_stats["quantile_0_01"][dim_i] = td.percentile(0.01)
                quantile_stats["quantile_99_99"][dim_i] = td.percentile(99.99)

        key_stats[key] = {
            "mean": mean.tolist(),   # (12,) or (100, 12)
            "std":  std.tolist(),
            "min":  min_.tolist(),
            "max":  max_.tolist(),
            **{k: v.tolist() for k, v in quantile_stats.items()},  # always (12,)
        }
        print(f"[NormStats] key={key} n={m['n']} mean_shape={mean.shape}")

    if not key_stats:
        raise RuntimeError("No stats produced — check that episodes have valid zarr data.")

    all_stats_out[emb_id] = key_stats

    payload = {
        "stats": all_stats_out,
        "loading_time": None,
        "computing_time": elapsed,
        "frames": sum(merged[k]["n"] for k in tracked_keys if merged[k]["n"] > 0),
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
    parser.add_argument("--n_shards", type=int, default=300,
                        help="Number of parallel shard containers")
    parser.add_argument("--samples_per_shard", type=int, default=700,
                        help="Frames per shard to sample")
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

    total_frames = parsed.n_shards * parsed.samples_per_shard
    print(
        f"Submitting norm-stats job: data={parsed.data_config!r} "
        f"n_shards={parsed.n_shards} samples_per_shard={parsed.samples_per_shard} "
        f"→ ~{total_frames:,} total frames sampled"
        + (f"  exclude_hashes={len(exclude_hashes)}" if exclude_hashes else "")
    )
    print(
        f"Stats will be keyed as: ee_pose ({_OBS_DIM},), "
        f"actions_cartesian ({_CHUNK_LEN},{_ACT_DIM}) "
        f"[quantiles: ({_ACT_DIM},)]"
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
