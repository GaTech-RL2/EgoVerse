"""Test: does replay with re-init per step achieve perfect coverage?
If re-setting state every step works but free-running doesn't, the issue
is physics divergence (floating point chaos in contact solver)."""
import numpy as np
import zarr
from pathlib import Path
from Tsimulation.pushshapes import PushShapesEnv

data_dir = Path("/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle")
eps = sorted([p for p in data_dir.iterdir() if p.name.endswith(".zarr")])[:20]

env_kw = dict(object_shape="T", pusher_shape="circle", obstacle_level=0, image_size=96)

results_free = []
results_reinit = []

for ep in eps:
    z = zarr.open_group(str(ep), mode="r")
    state = np.asarray(z["observations.state"][:])
    actions = np.asarray(z["actions"][:])
    goal = np.asarray(z["goal_pose"][:])
    reward = np.asarray(z["reward"][:]).squeeze()
    
    # Find real episode length (before zero padding)
    real_T = len(actions)
    for t in range(len(actions)):
        if np.allclose(actions[t], 0) and np.allclose(state[t], 0):
            real_T = t
            break
    
    if real_T < 10:
        continue
    
    goal_pose = tuple(float(x) for x in goal[0].reshape(-1)[:3])
    
    # Free-running replay (single init, then step with stored actions)
    env = PushShapesEnv(**env_kw)
    env.reset(seed=0)
    env.set_state(
        agent_pos=(float(state[0,0]), float(state[0,1])),
        object_pose=(float(state[0,2]), float(state[0,3]), float(state[0,4])),
        goal_pose=goal_pose,
    )
    max_cov_free = 0.0
    for t in range(real_T):
        _, _, term, _, info = env.step(actions[t])
        max_cov_free = max(max_cov_free, info["coverage"])
        if term:
            break
    
    # Re-init replay (reset state from GT every step, then step)
    env2 = PushShapesEnv(**env_kw)
    env2.reset(seed=0)
    max_cov_reinit = 0.0
    for t in range(real_T):
        env2.set_state(
            agent_pos=(float(state[t,0]), float(state[t,1])),
            object_pose=(float(state[t,2]), float(state[t,3]), float(state[t,4])),
            goal_pose=goal_pose,
        )
        _, _, term, _, info = env2.step(actions[t])
        max_cov_reinit = max(max_cov_reinit, info["coverage"])
        if term:
            break
    
    stored_max = float(reward.max())
    results_free.append(max_cov_free)
    results_reinit.append(max_cov_reinit)
    
    if max_cov_free < 0.9:
        print(f"{ep.name}: T={real_T} stored={stored_max:.3f} "
              f"free={max_cov_free:.3f} reinit={max_cov_reinit:.3f} ** DIVERGED")
    else:
        print(f"{ep.name}: T={real_T} stored={stored_max:.3f} "
              f"free={max_cov_free:.3f} reinit={max_cov_reinit:.3f}")

free = np.array(results_free)
reinit = np.array(results_reinit)
print(f"\nFree-run:  mean={free.mean():.3f} min={free.min():.3f} >=0.9: {(free>=0.9).sum()}/{len(free)}")
print(f"Re-init:   mean={reinit.mean():.3f} min={reinit.min():.3f} >=0.9: {(reinit>=0.9).sum()}/{len(reinit)}")
