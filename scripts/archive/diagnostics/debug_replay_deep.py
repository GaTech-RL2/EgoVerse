import numpy as np
import zarr
from pathlib import Path
from Tsimulation.pushshapes import PushShapesEnv
import inspect

# Check what step() expects
env = PushShapesEnv(object_shape="T", pusher_shape="circle", obstacle_level=0, image_size=96)
print("=== ENV STEP SOURCE ===")
print(inspect.getsource(env.step)[:2000])
print()

print("=== ENV CONSTANTS ===")
print(f"WORLD_SIZE={env.WORLD_SIZE}, DT={env.DT}, PUSHER_SPEED={env.PUSHER_SPEED}")
print(f"action_space={env.action_space}")
print()

# Load episode and inspect action/state relationship
ep = sorted([p for p in Path("/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle").iterdir() if p.name.endswith(".zarr")])[0]
z = zarr.open_group(str(ep), mode="r")
state = np.asarray(z["observations.state"][:])
actions = np.asarray(z["actions"][:])
goal = np.asarray(z["goal_pose"][:])

print(f"=== EPISODE {ep.name} ===")
print(f"state.shape={state.shape} actions.shape={actions.shape}")
print(f"state range: [{state.min(0).round(1)}, {state.max(0).round(1)}]")
print(f"actions range: [{actions.min(0).round(1)}, {actions.max(0).round(1)}]")
print()

# Check if actions look like absolute positions or deltas
print("=== FIRST 5 STEPS ===")
for t in range(5):
    agent_pos = state[t, :2]
    next_agent = state[t+1, :2] if t+1 < len(state) else state[t, :2]
    delta = next_agent - agent_pos
    print(f"t={t}: agent={agent_pos.round(1)} action={actions[t].round(1)} "
          f"next_agent={next_agent.round(1)} delta={delta.round(2)}")

# Check if actions ARE the next agent position
print()
print("=== ARE ACTIONS = NEXT AGENT POS? ===")
for t in range(5):
    next_agent = state[t+1, :2] if t+1 < len(state) else state[t, :2]
    print(f"t={t}: action={actions[t].round(2)} next_state_agent={next_agent.round(2)} "
          f"match={np.allclose(actions[t], next_agent, atol=1.0)}")
