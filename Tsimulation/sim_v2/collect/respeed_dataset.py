"""Re-render an existing dataset at a different PUSHER speed, as a new dataset.

Takes a collected dataset (e.g. circle_4500_plus_gen_v2), replays each episode
with the pusher's kinematic speed cap scaled by ``--speed-factor``, and writes
the result as a NEW dataset. Used to build slow/fast pusher embodiment variants
from data that already exists, instead of re-teleoperating it.

Two things make this work where naive speed scaling fails:

1. **Time-scaled command stream.** The source ``actions`` are absolute target
   positions paced to the ORIGINAL speed. ``env.step`` clamps each substep to
   ``min(PUSHER_SPEED, dist/dt_sub)``, so at 0.5x the pusher falls progressively
   behind the recorded stream and the episode desynchronizes (measured: coverage
   0.000). Holding each waypoint ``1/speed_factor`` steps makes the pusher walk
   the SAME path at the new speed. Episode length scales accordingly.

2. **Actions are the ACHIEVED PUSHER POSE, not the cursor.** ``--action-space
   pusher`` (default) records ``action[t] = agent_pos AFTER step t`` -- a target
   that is reachable within one step by construction, so the action stream is
   self-consistent at the new speed. ``--action-space cursor`` keeps the legacy
   semantics (record the commanded cursor target) for eval compatibility.

``state[t]`` stays the PRE-step observation, matching the collector, so the
``(state[t], action[t])`` causal pairing is unchanged.

Usage:
  python -m Tsimulation.sim_v2.collect.respeed_dataset \
      --src /path/circle_4500_plus_gen_v2 --dst /path/circle_4500_gen_v2_p0.5x \
      --speed-factor 0.5 --limit 100 --sim-version v2
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Tsimulation.sim_v2.collect.zarr_writer import ZarrDemoWriter  # noqa: E402
from Tsimulation.sim_v2.pushshapes import get_env  # noqa: E402

ACTION_KEY = "actions"
STATE_KEY = "observations.state"


def _episode_paths(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.name.endswith(".zarr"))


def respeed_one(
    src_path: Path,
    get_writer,
    *,
    speed_factor: float,
    sim_version: str | None,
    action_space: str,
    compensate: bool,
    success_threshold: float,
    resample: str = "hold",
    track_gain: float = 0.0,
    track_clamp: float = 25.0,
) -> dict:
    store = zarr.open_group(str(src_path), mode="r")
    attrs = dict(store.attrs)
    actions = np.asarray(store[ACTION_KEY][:])
    src_states = np.asarray(store[STATE_KEY][:])
    total_frames = attrs.get("total_frames", None)
    if total_frames is not None and total_frames < len(actions):
        actions = actions[:total_frames]
        src_states = src_states[:total_frames]

    ep_init = json.loads(attrs["episode_init"]) if "episode_init" in attrs else None
    if ep_init is None:
        raise ValueError("%s has no episode_init; cannot re-render" % src_path.name)
    env_args = json.loads(attrs["task_description"])["env_args"]

    n_src = int(len(actions))

    # ---- time-scale the command stream to the new speed ----------------------
    if compensate and speed_factor != 1.0:
        ts = 1.0 / float(speed_factor)
        n_new = max(1, int(round(len(actions) * ts)))
        if resample == "interp":
            # Linearly interpolate the waypoint path. When speeding up (ts<1)
            # this preserves the geometry instead of dropping every 1/ts-th
            # waypoint, so the pusher does not cut corners.
            src_t = np.linspace(0.0, 1.0, len(actions))
            new_t = np.linspace(0.0, 1.0, n_new)
            actions = np.stack(
                [np.interp(new_t, src_t, actions[:, d])
                 for d in range(actions.shape[1])], axis=1)
        elif resample == "adaptive":
            # Compress the TRANSIT, preserve the MANIPULATION.
            #
            # Uniform compression removes the same fraction of every phase,
            # including the slow endgame where the human is making the fine
            # corrections that actually seat the object -- which is exactly
            # where the 1.5x failures happen. But during contact the human was
            # already moving well below the speed cap, so those frames can keep
            # their original pacing for free; all the speedup can come out of
            # the free-flight segments, where the pusher has headroom.
            #
            # Frames where the object is moving keep a 1:1 rate; the rest are
            # compressed by `speed_factor`. Episode length lands between 1.0x
            # and `speed_factor`, and contact dynamics stay at source pacing.
            dobj = np.linalg.norm(np.diff(src_states[:, 2:4], axis=0), axis=1)
            dth = np.abs(np.diff(src_states[:, 4])) if src_states.shape[1] > 4 \
                else np.zeros_like(dobj)
            motion = np.concatenate([[0.0], dobj + 30.0 * dth])
            moving = motion > 0.5                       # world units per step
            rate = np.where(moving, 1.0, 1.0 / float(speed_factor))
            cum = np.cumsum(rate)
            n_new = max(1, int(round(cum[-1])))
            src_idx = np.interp(np.arange(n_new) + 0.5, cum,
                                np.arange(len(rate), dtype=np.float64))
            base = np.arange(len(actions), dtype=np.float64)
            actions = np.stack(
                [np.interp(src_idx, base, actions[:, d])
                 for d in range(actions.shape[1])], axis=1)
        else:
            idx = np.minimum((np.arange(n_new) / ts).astype(int), len(actions) - 1)
            actions = actions[idx]

    env = get_env(sim_version)(
        object_shape=env_args["object_shape"],
        pusher_shape=env_args["pusher_shape"],
        obstacle_level=env_args.get("obstacle_level", 0),
        image_size=env_args.get("image_size", 96),
        render_mode="rgb_array",
    )
    # Scale off the CLASS attr so repeated episodes never compound the factor.
    env.PUSHER_SPEED = type(env).PUSHER_SPEED * float(speed_factor)
    env.STICK_TURN_RATE = type(env).STICK_TURN_RATE * float(speed_factor)
    env.SUCCESS_THRESHOLD = float(success_threshold)
    env._skip_obs_render = False

    env.reset(seed=ep_init.get("reset_seed"))
    env.set_state(
        agent_pos=tuple(ep_init["agent_pos"]),
        agent_angle=float(ep_init.get("agent_angle", 0.0)),
        object_pose=tuple(ep_init["object_pose"]),
        goal_pose=tuple(ep_init["goal_pose"]),
    )

    is_socket = env_args["pusher_shape"] == "u_socket"
    # One writer per env-args signature, so the episode FILENAME carries the true
    # object/pusher shape and obstacle level, and task_description records the
    # real env_args. The writer indexes each name-family independently, so
    # several of them can share one output directory without colliding.
    writer = get_writer(env_args)
    writer.start_episode(init_state=dict(ep_init))
    obs = env._get_obs()
    peak = 0.0
    try:
        for t in range(len(actions)):
            pre_obs = obs
            cmd = np.asarray(actions[t], dtype=np.float64)

            # ---- closed-loop correction on the OBJECT trajectory -------------
            # Open-loop replay reproduces the PUSHER path faithfully (measured:
            # 3.4/512 units) but the object still drifts, because momentum
            # transfer differs at another speed. Reproducing the pusher path is
            # not the goal -- reproducing the OBJECT's path is. So steer the
            # pusher target against the object's error w.r.t. the source object
            # pose at matched progress. Pushing the target opposite the error
            # backs the pusher off when the object has run ahead, and leans in
            # when it lags.
            if track_gain > 0.0:
                prog = t / max(1, len(actions) - 1)
                ref = src_states[min(int(round(prog * (len(src_states) - 1))),
                                     len(src_states) - 1)]
                err = np.asarray(pre_obs["object_pose"][:2], dtype=np.float64) - ref[2:4]
                corr = -track_gain * err
                mag = float(np.linalg.norm(corr))
                if mag > track_clamp:
                    corr *= track_clamp / mag
                cmd = cmd.copy()
                cmd[:2] = cmd[:2] + corr

            obs, reward, terminated, _trunc, info = env.step(cmd)
            peak = max(peak, float(info.get("coverage", 0.0)))

            pusher_obs_pose = pre_obs["agent_pos"]
            if is_socket:
                pusher_obs_pose = np.concatenate(
                    [pre_obs["agent_pos"], np.atleast_1d(pre_obs["agent_angle"])]
                )

            if action_space == "pusher":
                # Where the pusher ACTUALLY ended up -- reachable in one step.
                act = np.asarray(obs["agent_pos"], dtype=np.float64)
                if is_socket:
                    act = np.concatenate([act, np.atleast_1d(obs["agent_angle"])])
            else:
                act = cmd

            writer.add_step(
                image=pre_obs["image"],
                pusher_obs_pose=pusher_obs_pose,
                object_obs_pose=pre_obs["object_pose"],
                pusher_cmd_pose=cmd,
                action=act,
                reward=reward,
                goal_pose=pre_obs["goal_pose"],
            )
            if terminated:
                break
    except Exception:
        writer.abort_episode()
        env.close()
        raise

    steps = writer.steps_in_episode
    idx = writer.commit_episode()
    env.close()
    return {"src": src_path.name, "index": idx, "steps": steps,
            "peak_coverage": peak, "src_frames": n_src}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--speed-factor", required=True, type=float)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--include", default=None,
                   help="only convert episodes whose filename matches this regex "
                        "(e.g. '_obs([1-9]|[12][0-9]|30)_' for obstacle-gen only)")
    p.add_argument("--shard", type=int, default=0,
                   help="this worker's index (0-based) for a sharded array run")
    p.add_argument("--num-shards", type=int, default=1,
                   help="total workers; each takes episodes[shard::num_shards]")
    p.add_argument("--sample", choices=("head", "even"), default="head",
                   help="with --limit: 'head' takes the first N, 'even' spreads "
                        "the N across the whole matched list (use for obstacle "
                        "levels, so all levels are represented)")
    p.add_argument("--sim-version", default=None,
                   help="PIN to the version that collected the source data")
    p.add_argument("--action-space", choices=("pusher", "cursor"), default="pusher",
                   help="what to store in `actions` (default: achieved pusher pose)")
    p.add_argument("--resample", choices=("hold", "interp", "adaptive"), default="hold",
                   help="how to time-scale the command stream: 'hold' repeats/"
                        "drops waypoints, 'interp' linearly interpolates the "
                        "path (no corner cutting); 'adaptive' preserves contact "
                        "frames and compresses only free transit")
    p.add_argument("--track-gain", type=float, default=0.0,
                   help="closed-loop gain correcting the pusher target against "
                        "the OBJECT's deviation from the source trajectory at "
                        "matched progress. 0 = open loop (default). ~0.5 is a "
                        "reasonable starting point.")
    p.add_argument("--track-clamp", type=float, default=25.0,
                   help="max correction magnitude in world units (default 25)")
    p.add_argument("--no-compensate", action="store_true",
                   help="do NOT time-scale the command stream (reproduces the "
                        "broken naive behaviour; for A/B only)")
    p.add_argument("--success-threshold", type=float, default=1.01,
                   help="env success cutoff; >1 disables early termination so "
                        "peak coverage is unbiased (default 1.01)")
    p.add_argument("--sr-threshold", type=float, default=0.80)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    if not math.isfinite(args.speed_factor) or args.speed_factor <= 0:
        p.error("--speed-factor must be a finite positive number")

    episodes = _episode_paths(args.src)
    if args.include:
        rx = re.compile(args.include)
        episodes = [e for e in episodes if rx.search(e.name)]
    if args.limit and args.limit < len(episodes):
        if args.sample == "even":
            idx = np.linspace(0, len(episodes) - 1, args.limit).round().astype(int)
            episodes = [episodes[i] for i in sorted(set(idx.tolist()))]
        else:
            episodes = episodes[: args.limit]
    if not episodes:
        print("no .zarr episodes in %s" % args.src, file=sys.stderr)
        return 2

    if args.num_shards > 1:
        episodes = episodes[args.shard::args.num_shards]
        if not episodes:
            print("shard %d/%d: nothing to do" % (args.shard, args.num_shards))
            return 0

    args.dst.mkdir(parents=True, exist_ok=True)
    print("src  : %s (%d episodes)" % (args.src, len(episodes)))
    print("dst  : %s" % args.dst)
    print("speed: %gx   action_space=%s   compensate=%s   sim=%s\n"
          % (args.speed_factor, args.action_space,
             not args.no_compensate, args.sim_version or "current"))

    _writers: dict = {}
    _meta = {
            "speed_factor": float(args.speed_factor),
            "action_space": args.action_space,
            "embodiment_variant": "pusher_%gx" % args.speed_factor,
            "source_dataset": str(args.src),
        "time_compensated": not args.no_compensate,
        "resample_mode": args.resample,
        "track_gain": float(args.track_gain),
    }

    def get_writer(env_args: dict) -> ZarrDemoWriter:
        key = json.dumps({k: env_args.get(k) for k in
                          ("object_shape", "pusher_shape", "obstacle_level")},
                         sort_keys=True)
        w = _writers.get(key)
        if w is None:
            w = ZarrDemoWriter(
                path=args.dst, env_args=dict(env_args),
                image_size=env_args.get("image_size", 96), fps=30,
                tag="respeed", metadata_override=dict(_meta),
            )
            _writers[key] = w
        return w

    results, failed = [], 0
    for i, ep in enumerate(episodes):
        try:
            r = respeed_one(
                ep, get_writer,
                speed_factor=args.speed_factor,
                sim_version=args.sim_version,
                action_space=args.action_space,
                compensate=not args.no_compensate,
                success_threshold=args.success_threshold,
                resample=args.resample,
                track_gain=args.track_gain,
                track_clamp=args.track_clamp,
            )
        except Exception:
            failed += 1
            print("  [%4d] %-44s ERROR" % (i, ep.name))
            traceback.print_exc(limit=2, file=sys.stderr)
            continue
        results.append(r)
        if i < 10 or i % 25 == 0:
            print("  [%4d] %-44s T %4d->%-5d peak=%.4f"
                  % (i, r["src"], r["src_frames"], r["steps"], r["peak_coverage"]))
    for _w in _writers.values():
        _w.close()

    if not results:
        print("\nno episodes converted", file=sys.stderr)
        return 1
    peaks = [r["peak_coverage"] for r in results]
    sr = sum(1 for c in peaks if c >= args.sr_threshold) / len(peaks)
    mean_cov = sum(peaks) / len(peaks)
    print("\n%s" % ("=" * 62))
    print("converted   : %d / %d   (errors %d)" % (len(results), len(episodes), failed))
    print("mean peak   : %.4f" % mean_cov)
    print("SR@%.2f     : %.3f  (%d/%d)"
          % (args.sr_threshold, sr,
             sum(1 for c in peaks if c >= args.sr_threshold), len(peaks)))
    print("mean length : %.0f frames (source %.0f)"
          % (sum(r["steps"] for r in results) / len(results),
             sum(r["src_frames"] for r in results) / len(results)))

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"src": str(args.src), "dst": str(args.dst),
             "speed_factor": args.speed_factor, "action_space": args.action_space,
             "compensated": not args.no_compensate,
             "n": len(results), "errors": failed,
             "mean_peak": mean_cov, "sr": sr,
             "sr_threshold": args.sr_threshold,
             "episodes": results}, indent=2))
        print("wrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
