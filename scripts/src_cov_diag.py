"""Decisive data-gen feasibility diagnostic.

(1) SOURCE coverage: for a sample of circle_3000 episodes, compute final AND max
    per-episode coverage (intersection(obj,goal)/goal_area, the env's own metric)
    directly from stored object_pose + goal_pose (no set_state CoG gotcha). Tells
    us whether >=0.9 coverage is even PRESENT in the data the BC learned from.
(2) SCRIPTED collector yield: run the privileged approach+push heuristic (GT poses)
    over N fresh seeds and report its final-coverage distribution + SR>=0.9 / 0.85
    / 0.8. Tells us whether a NON-learned expert is a better data-gen tool than the
    capability-limited BC.
"""
import argparse, glob, json, os
import numpy as np
import zarr

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from Tsimulation.pushshapes.env import PushShapesEnv
from Tsimulation.collect.scripted_collect import scripted_action


def cov_from_poses(env, obj_pose, goal_pose):
    op = env._build_object_polygon((float(obj_pose[0]), float(obj_pose[1])), float(obj_pose[2]))
    gp = env._build_object_polygon((float(goal_pose[0]), float(goal_pose[1])), float(goal_pose[2]))
    if gp.area <= 0:
        return 0.0
    return float(op.intersection(gp).area / gp.area)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/coc/flash7/paphiwetsa3/datasets/circle_3000")
    p.add_argument("--n-src", type=int, default=200)
    p.add_argument("--n-scripted", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=400)
    args = p.parse_args()

    env = PushShapesEnv(object_shape="T", pusher_shape="circle", obstacle_level=0,
                        image_size=96, render_mode="rgb_array")

    # ---- (1) SOURCE coverage distribution ----
    eps = sorted(glob.glob(os.path.join(args.data, "episode_*.zarr")))[:args.n_src]
    fin, mx = [], []
    for ep in eps:
        z = zarr.open(ep, mode="r")
        st = np.asarray(z["observations.state"])      # (T,5)
        gp = np.asarray(z["goal_pose"])                # (T,3)
        covs = [cov_from_poses(env, st[t, 2:5], gp[t]) for t in range(0, len(st), 4)]
        fin.append(cov_from_poses(env, st[-1, 2:5], gp[-1]))
        mx.append(max(covs))
    fin, mx = np.array(fin), np.array(mx)
    def sr(a, t): return float((a >= t).mean())
    print(f"=== SOURCE circle_3000 (n={len(fin)}) ===", flush=True)
    print(f"FINAL  cov: mean={fin.mean():.3f} med={np.median(fin):.3f} "
          f"SR>=0.9={sr(fin,0.9):.2f} >=0.85={sr(fin,0.85):.2f} >=0.8={sr(fin,0.8):.2f} >=0.7={sr(fin,0.7):.2f}", flush=True)
    print(f"MAX    cov: mean={mx.mean():.3f} med={np.median(mx):.3f} "
          f"SR>=0.9={sr(mx,0.9):.2f} >=0.85={sr(mx,0.85):.2f} >=0.8={sr(mx,0.8):.2f} >=0.7={sr(mx,0.7):.2f}", flush=True)

    # ---- (2) SCRIPTED privileged collector yield (fresh seeds) ----
    world = float(env.WORLD_SIZE)
    sc_fin, sc_max = [], []
    for s in range(args.n_scripted):
        obs, info = env.reset(seed=s)
        cov = float(info.get("coverage", 0.0)); best = cov
        for _ in range(args.max_steps):
            a = scripted_action(agent_xy=np.asarray(obs["agent_pos"], np.float64),
                                object_xy=np.asarray(obs["object_pose"][:2], np.float64),
                                goal_xy=np.asarray(obs["goal_pose"][:2], np.float64),
                                world_size=world)
            obs, r, term, trunc, info = env.step(a)
            cov = float(info.get("coverage", 0.0)); best = max(best, cov)
            if term or trunc:
                break
        sc_fin.append(cov); sc_max.append(best)
    sc_fin, sc_max = np.array(sc_fin), np.array(sc_max)
    print(f"\n=== SCRIPTED collector (privileged, n={len(sc_fin)}, max_steps={args.max_steps}) ===", flush=True)
    print(f"FINAL  cov: mean={sc_fin.mean():.3f} med={np.median(sc_fin):.3f} "
          f"SR>=0.9={sr(sc_fin,0.9):.2f} >=0.85={sr(sc_fin,0.85):.2f} >=0.8={sr(sc_fin,0.8):.2f} >=0.7={sr(sc_fin,0.7):.2f}", flush=True)
    print(f"MAX    cov: mean={sc_max.mean():.3f} med={np.median(sc_max):.3f} "
          f"SR>=0.9={sr(sc_max,0.9):.2f} >=0.85={sr(sc_max,0.85):.2f} >=0.8={sr(sc_max,0.8):.2f} >=0.7={sr(sc_max,0.7):.2f}", flush=True)
    print("DIAG-DONE", flush=True)


if __name__ == "__main__":
    main()
