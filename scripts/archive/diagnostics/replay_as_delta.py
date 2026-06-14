import numpy as np
import zarr
from pathlib import Path
from Tsimulation.pushshapes import PushShapesEnv

data_dir = Path("/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle")
eps = sorted([p for p in data_dir.iterdir() if p.name.endswith(".zarr")])[:5]

for ep in eps:
    z = zarr.open_group(str(ep), mode="r")
    state = np.asarray(z["observations.state"][:])
    actions = np.asarray(z["actions"][:])
    goal = np.asarray(z["goal_pose"][:])

    env = PushShapesEnv(object_shape="T", pusher_shape="circle", obstacle_level=0, image_size=96)
    env.reset(seed=0)
    frame0 = state[0]
    agent_pos = (float(frame0[0]), float(frame0[1]))
    object_pose = (float(frame0[2]), float(frame0[3]), float(frame0[4]))
    goal_pose = tuple(float(x) for x in goal[0].reshape(-1)[:3])
    env.set_state(agent_pos=agent_pos, object_pose=object_pose, goal_pose=goal_pose)

    # Try: action = next agent position from STATE (ignore stored actions)
    final_cov = 0.0
    for t in range(min(state.shape[0] - 1, actions.shape[0])):
        target = state[t + 1, :2]  # use next state's agent pos as action
        obs, rew, term, trunc, info = env.step(target)
        final_cov = info.get("coverage", 0.0)
        if term:
            break

    print(f"{ep.name}: T={actions.shape[0]}, cov_using_next_state={final_cov:.3f}")
