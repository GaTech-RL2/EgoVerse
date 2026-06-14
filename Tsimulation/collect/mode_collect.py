"""Generate DELIBERATELY SUBOPTIMAL PushShapes episodes by degrading good ones.

Why this exists
---------------
Preference learning only beats a single scalar reward when the dataset contains
behaviours that *disagree* across axes -- one trajectory fast but sloppy,
another slow but precise. A dataset where everything succeeds carries no
preference signal. Measured on circle_3000 (3040 episodes, all >=0.95
coverage): the middle 50% of final translation error spans 2.0-2.6 world units
and rotation error is under 1.5 degrees. Nothing an annotator could rank.

Why degradation rather than a scripted controller
-------------------------------------------------
Writing a heuristic that pushes a T into a target *pose* is a real planning
problem -- bang-bang lever pushes oscillate: the T spins past the target angle,
the sign flips, it overshoots the other way, and meanwhile the lever force
shoves it across the arena. That is why PushT is a benchmark rather than a
solved toy. ``scripted_collect.py`` sidesteps it by ignoring orientation
entirely, which is why it plateaus near zero coverage.

Replaying a known-good trajectory dodges the problem completely: every episode
here is anchored to a real solution, so the degraded versions stay on the task
manifold and differ only in *execution quality* -- exactly the axis contrast
FPL needs.

Modes
-----
    fastsloppy    actions subsampled (bigger, coarser jumps) + noise, cut short
                  -> fewer frames, worse final pose
    slowprecise   actions interpolated (smaller, finer steps), no noise
                  -> more frames, pose as good as the source

Note what is deliberately ABSENT: fast *and* precise. That mirrors the paper's
bimodal-square construction (fast-left and slow-right demos, never fast-right),
which is what makes the compositionality test meaningful -- you ask the policy
at test time for a combination the data never showed it.

Usage::

    python -m Tsimulation.collect.mode_collect \\
        --source /coc/flash7/paphiwetsa3/datasets/circle_3000 \\
        --output data/pusht_modes --num-source 300
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import zarr

from Tsimulation.collect.zarr_writer import ZarrDemoWriter
from Tsimulation.pushshapes.env import PushShapesEnv


def _subsample(a: np.ndarray, stride: int) -> np.ndarray:
    """Keep every `stride`-th action. The pusher then has to cover the same
    ground in fewer commands, so each step is a coarser lunge."""
    return a[::stride]


def _interpolate(a: np.ndarray, factor: int) -> np.ndarray:
    """Linearly upsample the action sequence: same path, smaller increments."""
    n = len(a)
    src = np.arange(n)
    dst = np.linspace(0, n - 1, n * factor)
    return np.stack([np.interp(dst, src, a[:, i]) for i in range(a.shape[1])], axis=1)


def _repeat(a: np.ndarray, r: int) -> np.ndarray:
    """Hold each command for `r` steps. Same path and same step sizes as the
    source -- only the dwell time changes. Isolates duration from smoothness,
    which interpolation cannot (interpolation also shrinks the steps)."""
    return np.repeat(a, r, axis=0)


def _hesitate(a: np.ndarray, rng: np.random.Generator, events: int, back: int,
              dwell: int) -> np.ndarray:
    """Splice in retreat-and-resume stalls: at a few points the pusher is sent
    back to a command it issued `back` steps ago, held there, then carries on.
    The object gets released and re-approached, so coverage sags and recovers --
    which is what the directness axis measures."""
    n = len(a)
    if n < 40:
        return a
    pts = sorted(rng.choice(np.arange(back + 1, n - 2), size=min(events, n // 20),
                            replace=False))
    out, prev = [], 0
    for p in pts:
        out.append(a[prev:p])
        out.append(np.repeat(a[p - back][None, :], dwell, axis=0))
        prev = p
    out.append(a[prev:])
    return np.concatenate(out, axis=0)


# Every mode targets a DIFFERENT axis. Modes that all degrade the same thing
# would multiply the dataset without adding independent signal, which is the
# correlation trap that makes multi-axis preference learning pointless.
#
# `frac` is a (lo, hi) range resampled per episode: a single fixed value makes
# every episode equally bad, reproducing the no-spread problem at a lower mean.
#
# Subsampling is deliberately absent. The recorded actions are closed-loop
# responses to observed state; dropping two thirds of them desynchronises the
# pusher from the object and every episode collapses to exactly 0.000 coverage.
MODES: dict[str, dict] = {
    # slow + accurate: finer steps along the identical path   -> duration up
    "slowprecise": dict(factor=2, repeat=1, noise=0.0, frac=(1.0, 1.0),
                        hesitate=0),
    # abandoned early + noisy: both errors bad                -> translation up
    "fastsloppy": dict(factor=1, repeat=1, noise=2.5, frac=(0.45, 0.70),
                       hesitate=0),
    # stops during final alignment: placed but crooked        -> rotation up
    "misaligned": dict(factor=1, repeat=1, noise=0.0, frac=(0.86, 0.97),
                       hesitate=0),
    # completes, but shaky throughout                         -> smoothness up
    "jittery": dict(factor=1, repeat=1, noise=6.0, frac=(1.0, 1.0), hesitate=0),
    # retreats and re-approaches: coverage sags and recovers  -> directness down
    "hesitant": dict(factor=1, repeat=1, noise=0.0, frac=(1.0, 1.0), hesitate=4),
    # same motion, just dawdles                               -> duration only
    "dawdling": dict(factor=1, repeat=3, noise=0.0, frac=(1.0, 1.0), hesitate=0),
}


def degrade(actions: np.ndarray, m: dict, rng: np.random.Generator) -> np.ndarray:
    a = actions
    if m.get("factor", 1) > 1:
        a = _interpolate(a, m["factor"])
    if m.get("repeat", 1) > 1:
        a = _repeat(a, m["repeat"])
    if m.get("hesitate", 0):
        a = _hesitate(a, rng, events=m["hesitate"], back=25, dwell=10)
    lo, hi = m["frac"]
    frac = float(rng.uniform(lo, hi))
    if frac < 1.0:
        a = a[: max(4, int(len(a) * frac))]
    if m["noise"] > 0:
        a = a + rng.normal(0.0, m["noise"], a.shape)
    return a


def read_source(path: str):
    """Return (episode_init, actions) for a source episode, or None if unusable."""
    g = zarr.open(path, mode="r")
    at = dict(g.attrs)
    n = int(at["total_frames"])
    if n < 20:
        return None
    init = at.get("episode_init")
    if isinstance(init, str):
        init = json.loads(init)
    if not init or "object_pose" not in init:
        return None
    return init, np.asarray(g["actions"][:n], dtype=np.float64)


def replay(env, writer, init, actions) -> tuple[float, int]:
    """Reset, force the source episode's exact start state, then run `actions`."""
    env.reset(seed=init.get("reset_seed"))
    env.set_state(
        agent_pos=init["agent_pos"],
        object_pose=init["object_pose"],
        goal_pose=init["goal_pose"],
    )
    obs = env._get_obs()
    writer.start_episode(init_state=env.get_episode_init())
    cov = 0.0
    for k, act in enumerate(actions):
        pre = obs
        act = np.clip(act, 0.0, env.WORLD_SIZE)
        obs, reward, terminated, _trunc, info = env.step(act)
        writer.add_step(
            image=pre["image"],
            pusher_obs_pose=pre["agent_pos"],
            object_obs_pose=pre["object_pose"],
            pusher_cmd_pose=act,
            action=act,
            reward=reward,
            goal_pose=pre["goal_pose"],
        )
        cov = float(info.get("coverage", 0.0))
        if terminated:
            return cov, k + 1
    return cov, len(actions)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", required=True, help="dir of good episode_*.zarr")
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-source", type=int, default=200,
                    help="how many source episodes to degrade (each yields one "
                         "episode per mode)")
    ap.add_argument("--modes", default="all",
                    help="comma-separated, or 'all'")
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    modes = (list(MODES) if args.modes.strip() == "all"
             else [m.strip() for m in args.modes.split(",") if m.strip()])
    for m in modes:
        if m not in MODES:
            print(f"unknown mode {m!r}, valid: {list(MODES)}", file=sys.stderr)
            return 2

    src = sorted(glob.glob(f"{args.source}/*.zarr"))[: args.num_source]
    if not src:
        print(f"no episodes under {args.source}", file=sys.stderr)
        return 2
    print(f"source episodes: {len(src)}", flush=True)

    first = read_source(src[0])
    if first is None:
        print("could not read a usable source episode", file=sys.stderr)
        return 2
    init0 = first[0]

    env = PushShapesEnv(
        object_shape=init0.get("object_shape", "T"),
        pusher_shape=init0.get("pusher_shape", "circle"),
        obstacle_level=int(init0.get("obstacle_level", 0)),
        image_size=args.image_size,
    )
    env_args = {
        "object_shape": init0.get("object_shape", "T"),
        "pusher_shape": init0.get("pusher_shape", "circle"),
        "obstacle_level": int(init0.get("obstacle_level", 0)),
    }
    rng = np.random.default_rng(args.seed)

    # One writer (and filename tag) per mode so episodes stay separable, and so
    # the mode itself is available later as a categorical preference axis.
    writers = {
        m: ZarrDemoWriter(path=Path(args.output), env_args=env_args,
                          image_size=args.image_size, fps=args.fps, tag=m)
        for m in modes
    }

    stats = {m: [] for m in modes}
    skipped = 0
    for i, p in enumerate(src):
        rec = read_source(p)
        if rec is None:
            skipped += 1
            continue
        init, actions = rec
        for m in modes:
            a = degrade(actions, MODES[m], rng)
            cov, steps = replay(env, writers[m], init, a)
            if writers[m].commit_episode() >= 0:
                stats[m].append((cov, steps))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(src)}", flush=True)

    print(f"\nskipped {skipped} unusable source episodes\n")
    print(f"{'mode':13}{'n':>6}{'cov med':>10}{'cov p10':>10}{'cov p90':>10}"
          f"{'steps med':>11}")
    for m in modes:
        if not stats[m]:
            continue
        c = np.array([s[0] for s in stats[m]])
        n = np.array([s[1] for s in stats[m]])
        print(f"{m:13}{len(c):6d}{np.median(c):10.3f}{np.percentile(c,10):10.3f}"
              f"{np.percentile(c,90):10.3f}{int(np.median(n)):11d}")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
