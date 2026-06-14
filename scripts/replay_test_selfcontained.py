"""Self-contained replay coverage test for pushshapes zarr datasets.

Runs ``PushShapesEnv`` replays against episode_*.zarr dirs and reports
per-episode replayed coverage. Two modes:

* ``--rechunked DIR`` (compare mode): for each shared episode name in
  both ``--src`` and ``--rechunked``, replay both copies and assert
  identical coverage + replay length.
* no ``--rechunked`` (scan mode): replay every episode in ``--src``
  and report coverage distribution + count of episodes below
  ``--min-coverage``.

Self-contained — only depends on ``Tsimulation.pushshapes.env``, not on
``Tsimulation.examples`` or ``Tsimulation.collect`` (which may not be
present on every checkout). Constants for zarr keys are inlined.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import zarr

ACTION_KEY = "actions"
GOAL_KEY = "goal_pose"
STATE_KEY = "observations.state"


def _replay_one(ep_path: Path) -> dict:
    """Replay ``ep_path``'s recorded actions in a fresh env, return coverage."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from Tsimulation.pushshapes.env import PushShapesEnv

    store = zarr.open_group(str(ep_path), mode="r")
    attrs = dict(store.attrs)
    total_frames = attrs.get("total_frames", None)
    actions = np.asarray(store[ACTION_KEY][:])
    states = np.asarray(store[STATE_KEY][:])
    goal_pose = np.asarray(store[GOAL_KEY][0])
    reward = np.asarray(store["reward"][:]).squeeze()
    env_args = json.loads(attrs["task_description"])["env_args"]

    if total_frames is not None and total_frames < len(actions):
        actions = actions[:total_frames]
        states = states[:total_frames]
        reward = reward[:total_frames]

    ep_init = json.loads(attrs["episode_init"]) if "episode_init" in attrs else None

    env = PushShapesEnv(
        object_shape=env_args["object_shape"],
        pusher_shape=env_args["pusher_shape"],
        obstacle_level=env_args.get("obstacle_level", 0),
        image_size=env_args.get("image_size", 96),
    )
    reset_seed = ep_init.get("reset_seed") if ep_init else None
    env.reset(seed=reset_seed)

    if ep_init is not None:
        ap = tuple(ep_init["agent_pos"])
        op = tuple(ep_init["object_pose"])
        gp = tuple(ep_init["goal_pose"])
    else:
        s0 = states[0]
        ap = (float(s0[0]), float(s0[1]))
        op = (float(s0[2]), float(s0[3]), float(s0[4]))
        gp = (float(goal_pose[0]), float(goal_pose[1]), float(goal_pose[2]))
    env.set_state(agent_pos=ap, object_pose=op, goal_pose=gp)

    max_cov = 0.0
    for i in range(len(actions)):
        _, _, term, _, info = env.step(actions[i])
        max_cov = max(max_cov, float(info["coverage"]))
        if term:
            break
    env.close()
    return {
        "name": ep_path.name,
        "T": int(len(actions)),
        "stored_cov": float(reward.max()),
        "replay_cov": float(max_cov),
    }


def _worker(ep_path_str: str) -> dict:
    try:
        return _replay_one(Path(ep_path_str))
    except Exception as e:
        return {"name": Path(ep_path_str).name, "error": f"{type(e).__name__}: {e}"}


def _scan(src: Path, n: int | None, workers: int, min_cov: float) -> int:
    episodes = sorted(p for p in src.glob("episode_*.zarr") if p.is_dir())
    if n is not None:
        episodes = episodes[:n]
    print(f"scanning {len(episodes)} episodes from {src} with {workers} workers")
    t0 = time.perf_counter()
    n_below = 0
    n_err = 0
    covs: list[float] = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, r in enumerate(
            pool.imap_unordered(_worker, [str(p) for p in episodes], chunksize=4),
            start=1,
        ):
            if "error" in r:
                n_err += 1
                print(f"  ERR {r['name']}: {r['error']}", flush=True)
                continue
            covs.append(r["replay_cov"])
            if r["replay_cov"] < min_cov:
                n_below += 1
                print(
                    f"  LOW {r['name']:55s} cov={r['replay_cov']:.4f} (stored={r['stored_cov']:.4f}) T={r['T']}",
                    flush=True,
                )
            if i % 50 == 0 or i == len(episodes):
                rate = i / max(time.perf_counter() - t0, 1e-9)
                eta = (len(episodes) - i) / max(rate, 1e-9)
                print(
                    f"  [{i}/{len(episodes)}] below={n_below} err={n_err} "
                    f"| {rate:.1f} ep/s | eta {eta/60:.1f} min",
                    flush=True,
                )
    arr = np.array(covs)
    if len(arr):
        print()
        print(
            f"coverage stats: n={len(arr)} mean={arr.mean():.4f} median={np.median(arr):.4f} "
            f"min={arr.min():.4f} max={arr.max():.4f} "
            f"p10={np.percentile(arr,10):.4f} p90={np.percentile(arr,90):.4f}"
        )
        print(f"below threshold ({min_cov}): {n_below}/{len(arr)} ({100*n_below/len(arr):.1f}%)")
    print(f"errors: {n_err}")
    return 0 if n_err == 0 else 1


def _compare_worker(args: tuple[str, str]) -> dict:
    src_path, dst_path = args
    return {
        "name": Path(src_path).name,
        "src": _worker(src_path),
        "dst": _worker(dst_path),
    }


def _compare(src: Path, rechunked: Path, n: int, workers: int, verbose: bool) -> int:
    src_eps = {p.name for p in src.glob("episode_*.zarr") if p.is_dir()}
    dst_eps = {p.name for p in rechunked.glob("episode_*.zarr") if p.is_dir()}
    common = sorted(src_eps & dst_eps)
    if n is not None:
        common = common[:n]
    if not common:
        print(f"no common episode names", file=sys.stderr)
        return 2
    print(f"comparing {len(common)} episodes (src vs rechunked) with {workers} workers\n", flush=True)
    if verbose:
        header = f"  {'episode':55s} {'src_cov':>10s} {'dst_cov':>10s} {'src_T':>6s} {'dst_T':>6s} {'ok':>4s}"
        print(header, flush=True)
        print("  " + "-" * (len(header) - 2), flush=True)
    n_mismatch = 0
    n_err = 0
    t0 = time.perf_counter()
    jobs = [(str(src / name), str(rechunked / name)) for name in common]
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_compare_worker, jobs, chunksize=2), start=1):
            name = r["name"]; s = r["src"]; d = r["dst"]
            if "error" in s or "error" in d:
                n_err += 1
                if verbose:
                    print(f"  ERR {name}: src={s.get('error','-')}  dst={d.get('error','-')}", flush=True)
            else:
                ok = (s["replay_cov"] == d["replay_cov"]) and (s["T"] == d["T"])
                if not ok:
                    n_mismatch += 1
                    print(
                        f"  MISMATCH {name:55s} src={s['replay_cov']:.6f} dst={d['replay_cov']:.6f} "
                        f"src_T={s['T']} dst_T={d['T']}",
                        flush=True,
                    )
                elif verbose:
                    print(
                        f"  {name:55s} {s['replay_cov']:10.6f} {d['replay_cov']:10.6f} "
                        f"{s['T']:6d} {d['T']:6d} {'YES':>4s}",
                        flush=True,
                    )
            if i % 25 == 0 or i == len(jobs):
                rate = i / max(time.perf_counter() - t0, 1e-9)
                eta = (len(jobs) - i) / max(rate, 1e-9)
                print(
                    f"  [{i}/{len(jobs)}] match={i - n_mismatch - n_err} mismatch={n_mismatch} err={n_err} "
                    f"| {rate:.1f} ep/s | eta {eta/60:.1f} min",
                    flush=True,
                )
    print(flush=True)
    n_match = len(common) - n_mismatch - n_err
    print(f"summary: {n_match}/{len(common)} match exactly  mismatch={n_mismatch}  err={n_err}")
    return 0 if n_mismatch == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--rechunked", type=Path, default=None,
                   help="If provided, compare src vs rechunked; else scan src.")
    p.add_argument("--n", type=int, default=None,
                   help="Limit to first N episodes (default: all in scan, 10 in compare).")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--min-coverage", type=float, default=0.95)
    p.add_argument("--verbose", action="store_true",
                   help="In compare mode, print one line per episode (default: only mismatches + progress).")
    args = p.parse_args()

    if args.rechunked is not None:
        return _compare(args.src, args.rechunked, args.n, args.workers, args.verbose)
    return _scan(args.src, args.n, args.workers, args.min_coverage)


if __name__ == "__main__":
    raise SystemExit(main())
