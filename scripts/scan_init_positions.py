"""Scan episode_init across a dataset: distribution of pusher start
(agent_pos), object start pose, and goal pose. Usage: <folder> [N]"""
import glob, json, os, sys
import numpy as np
import zarr

folder = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
eps = sorted(glob.glob(os.path.join(folder, "episode_*.zarr")))
step = max(1, len(eps) // N)
samp = eps[::step][:N]

agent, obj, goal = [], [], []
for e in samp:
    r = json.load(open(os.path.join(e, "zarr.json")))
    a = r.get("attributes", r)
    ei = json.loads(a.get("episode_init", "{}"))
    if "agent_pos" in ei:
        agent.append(ei["agent_pos"])
        obj.append(ei["object_pose"])
        goal.append(ei["goal_pose"])

agent = np.array(agent); obj = np.array(obj); goal = np.array(goal)
print(f"folder = {folder}")
print(f"sampled {len(agent)} episodes with episode_init\n")

def describe(name, arr, labels):
    print(f"{name}:")
    for i, lab in enumerate(labels):
        c = arr[:, i]
        print(f"   {lab:8s} min={c.min():7.1f}  max={c.max():7.1f}  "
              f"mean={c.mean():7.1f}  std={c.std():6.1f}")
    print()

describe("PUSHER start (agent_pos)", agent, ["x", "y"])
describe("OBJECT start (object_pose)", obj, ["x", "y", "theta"])
describe("GOAL (goal_pose)", goal, ["x", "y", "theta"])

# Is pusher start clustered or spread over the whole 512 workspace?
span_x = agent[:,0].max() - agent[:,0].min()
span_y = agent[:,1].max() - agent[:,1].min()
print(f"pusher-start bounding box: {span_x:.0f} x {span_y:.0f} px "
      f"(workspace ~512x512)")
print(f"pusher-start centroid: ({agent[:,0].mean():.0f}, {agent[:,1].mean():.0f})")
