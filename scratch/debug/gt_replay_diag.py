"""Diagnostic: replay GROUND-TRUTH recorded actions through PushShapesEnv and
report coverage. Separates env/goal/action-semantics bugs from policy bugs.
If GT replay coverage is high -> env works, the 0.0 model coverage is a policy
issue. If GT replay is also ~0 -> env init / goal / action-semantics is wrong.
"""
import numpy as np
import zarr
from Tsimulation.pushshapes import PushShapesEnv

ROOT = "/coc/flash7/paphiwetsa3/datasets/new_circle_3"
import os
eps = sorted([d for d in os.listdir(ROOT) if d.endswith(".zarr")])[:3]

env = PushShapesEnv(object_shape="T", pusher_shape="circle", obstacle_level=0,
                    image_size=96, render_mode="rgb_array")

for epname in eps:
    ep = os.path.join(ROOT, epname)
    store = zarr.open_group(ep, mode="r")
    keys = list(store.keys())
    actions = np.asarray(store["actions"][:], dtype=np.float64)        # (T,2)
    state = np.asarray(store["observations.state"][:], dtype=np.float64)  # (T,5)
    T = actions.shape[0]
    goal = None
    if "goal_pose" in keys:
        g = np.asarray(store["goal_pose"][:], dtype=np.float64)
        goal = tuple(float(x) for x in g[0].reshape(-1)[:3])

    agent_pos = (float(state[0, 0]), float(state[0, 1]))
    object_pose = (float(state[0, 2]), float(state[0, 3]), float(state[0, 4]))

    print(f"\n=== {epname}  T={T}  keys={keys} ===")
    print(f"  state[0]={state[0]}  action[0..2]={actions[:2].tolist()}  goal={goal}")
    print(f"  action range: x[{actions[:,0].min():.1f},{actions[:,0].max():.1f}] "
          f"y[{actions[:,1].min():.1f},{actions[:,1].max():.1f}]")

    for label, gp in [("no_goal", None), ("with_goal", goal)]:
        try:
            env.reset(seed=0)
            env.set_state(agent_pos=agent_pos, object_pose=object_pose, goal_pose=gp)
        except Exception as e:
            print(f"  [{label}] set_state failed: {e}")
            continue
        covs = []
        for t in range(T):
            a = np.array([float(actions[t, 0]), float(actions[t, 1])], dtype=np.float64)
            obs, r, term, trunc, info = env.step(a)
            covs.append(float(info.get("coverage", 0.0)))
            if term:
                break
        print(f"  [{label}] GT-replay final_cov={covs[-1]:.3f} "
              f"max_cov={max(covs):.3f} mean_cov={np.mean(covs):.3f} steps={len(covs)}")
