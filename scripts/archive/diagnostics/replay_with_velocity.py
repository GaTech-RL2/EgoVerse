"""Test: if we capture and restore object velocity every step, does replay become perfect?"""
import numpy as np
import zarr
from pathlib import Path
from Tsimulation.pushshapes import PushShapesEnv

data_dir = Path("/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle")
eps = sorted([p for p in data_dir.iterdir() if p.name.endswith(".zarr")])

env_kw = dict(object_shape="T", pusher_shape="circle", obstacle_level=0, image_size=96)

# First: collect ground truth with velocities from a fresh run
print("=== PHASE 1: Record GT with velocities ===")
ep = eps[10]  # ep10 diverged badly (0.390)
z = zarr.open_group(str(ep), mode="r")
state = np.asarray(z["observations.state"][:])
actions = np.asarray(z["actions"][:])
goal = np.asarray(z["goal_pose"][:])
reward = np.asarray(z["reward"][:]).squeeze()

real_T = len(actions)
for t in range(len(actions)):
    if np.allclose(actions[t], 0) and np.allclose(state[t], 0):
        real_T = t
        break

goal_pose = tuple(float(x) for x in goal[0].reshape(-1)[:3])

# Record velocities from re-init replay (which matches GT perfectly)
env = PushShapesEnv(**env_kw)
env.reset(seed=0)
obj_velocities = []
obj_ang_velocities = []
pusher_velocities = []

for t in range(real_T):
    env.set_state(
        agent_pos=(float(state[t,0]), float(state[t,1])),
        object_pose=(float(state[t,2]), float(state[t,3]), float(state[t,4])),
        goal_pose=goal_pose,
    )
    # Record velocities BEFORE zeroing (set_state zeros them)
    # Actually set_state already zeroed them. Let's step and record AFTER step
    _, _, term, _, info = env.step(actions[t])
    obj_v = env._object_body.velocity
    obj_w = env._object_body.angular_velocity
    push_v = env._pusher_body.velocity
    obj_velocities.append((float(obj_v.x), float(obj_v.y)))
    obj_ang_velocities.append(float(obj_w))
    pusher_velocities.append((float(push_v.x), float(push_v.y)))

print(f"Recorded {len(obj_velocities)} velocity snapshots")
print(f"Obj velocity magnitudes: mean={np.linalg.norm(obj_velocities, axis=1).mean():.1f} "
      f"max={np.linalg.norm(obj_velocities, axis=1).max():.1f}")

# Phase 2: Free-run replay but restore object velocity from step t-1 before step t
print("\n=== PHASE 2: Free-run with velocity restore ===")
env2 = PushShapesEnv(**env_kw)
env2.reset(seed=0)
env2.set_state(
    agent_pos=(float(state[0,0]), float(state[0,1])),
    object_pose=(float(state[0,2]), float(state[0,3]), float(state[0,4])),
    goal_pose=goal_pose,
)

max_cov_velfix = 0.0
for t in range(real_T):
    if t > 0:
        # Restore the object velocity from the previous step's end
        env2._object_body.velocity = obj_velocities[t-1]
        env2._object_body.angular_velocity = obj_ang_velocities[t-1]
    _, _, term, _, info = env2.step(actions[t])
    max_cov_velfix = max(max_cov_velfix, info["coverage"])
    if term:
        break

# Phase 3: Plain free-run (no velocity restore)
env3 = PushShapesEnv(**env_kw)
env3.reset(seed=0)
env3.set_state(
    agent_pos=(float(state[0,0]), float(state[0,1])),
    object_pose=(float(state[0,2]), float(state[0,3]), float(state[0,4])),
    goal_pose=goal_pose,
)
max_cov_plain = 0.0
for t in range(real_T):
    _, _, term, _, info = env3.step(actions[t])
    max_cov_plain = max(max_cov_plain, info["coverage"])
    if term:
        break

stored = float(reward.max())
print(f"\nEp10 results:")
print(f"  Stored coverage:     {stored:.3f}")
print(f"  Free-run (plain):    {max_cov_plain:.3f}")
print(f"  Free-run (vel fix):  {max_cov_velfix:.3f}")
