"""Trace exactly when and where free-run diverges from GT state."""
import numpy as np
import zarr
from pathlib import Path
from Tsimulation.pushshapes import PushShapesEnv

data_dir = Path("/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle")
eps = sorted([p for p in data_dir.iterdir() if p.name.endswith(".zarr")])
ep = eps[10]  # the worst diverger

z = zarr.open_group(str(ep), mode="r")
state = np.asarray(z["observations.state"][:])
actions = np.asarray(z["actions"][:])
goal = np.asarray(z["goal_pose"][:])

real_T = len(actions)
for t in range(len(actions)):
    if np.allclose(actions[t], 0) and np.allclose(state[t], 0):
        real_T = t
        break

goal_pose = tuple(float(x) for x in goal[0].reshape(-1)[:3])

env = PushShapesEnv(object_shape="T", pusher_shape="circle", obstacle_level=0, image_size=96)
env.reset(seed=0)
env.set_state(
    agent_pos=(float(state[0,0]), float(state[0,1])),
    object_pose=(float(state[0,2]), float(state[0,3]), float(state[0,4])),
    goal_pose=goal_pose,
)

# Check initial state match
sim_pos = env._object_body.position
print(f"t=0 INIT: GT=({state[0,2]:.4f},{state[0,3]:.4f},{state[0,4]:.6f}) "
      f"sim=({sim_pos.x:.4f},{sim_pos.y:.4f},{env._object_body.angle:.6f})")

first_big_diff = None
for t in range(min(real_T, 100)):
    obs, _, term, _, info = env.step(actions[t])
    
    gt_next = state[t+1] if t+1 < len(state) else state[t]
    sim_agent = np.array([obs["agent_pos"][0], obs["agent_pos"][1]])
    sim_obj = np.array([env._object_body.position.x, env._object_body.position.y, 
                        env._object_body.angle])
    
    agent_diff = np.abs(sim_agent - gt_next[:2]).max()
    obj_diff = np.abs(sim_obj[:2] - gt_next[2:4]).max()
    angle_diff = abs(sim_obj[2] - gt_next[4])
    
    if t < 10 or obj_diff > 0.5 or (first_big_diff is None and obj_diff > 0.01):
        print(f"t={t:3d}: agent_err={agent_diff:.4f} obj_pos_err={obj_diff:.4f} "
              f"angle_err={angle_diff:.6f} cov={info['coverage']:.3f}")
        if first_big_diff is None and obj_diff > 0.01:
            first_big_diff = t
            print(f"  ^^^ FIRST SIGNIFICANT DRIFT at t={t}")
            print(f"  GT obj: ({gt_next[2]:.4f}, {gt_next[3]:.4f}, {gt_next[4]:.6f})")
            print(f"  Sim obj: ({sim_obj[0]:.4f}, {sim_obj[1]:.4f}, {sim_obj[2]:.6f})")
            print(f"  Obj velocity: ({env._object_body.velocity.x:.2f}, {env._object_body.velocity.y:.2f})")
            print(f"  Obj angular_vel: {env._object_body.angular_velocity:.4f}")
    if term:
        break

if first_big_diff is None:
    print("No significant drift in first 100 steps!")
