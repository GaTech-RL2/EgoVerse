"""Filter new_circle_clean_padded to only keep episodes that don't drift on replay.
Replays each episode from initial state with recorded actions, checks final
coverage > threshold. Copies passing episodes to new_circle_no_drift."""
import sys
import json
import shutil
import numpy as np
import zarr
from pathlib import Path

sys.path.insert(0, "/coc/flash7/paphiwetsa3/projects/EgoVerse2")
from Tsimulation.pushshapes import PushShapesEnv

SRC_DIR = Path("/coc/cedarp-dxu345-0/Tsim_datasets2/new_circle_clean_padded")
DST_DIR = Path("/coc/cedarp-dxu345-0/Tsim_datasets2/new_circle_no_drift")
COV_THRESHOLD = 0.5

episodes = sorted([p for p in SRC_DIR.iterdir() if p.name.endswith(".zarr")])
print(f"Found {len(episodes)} episodes in {SRC_DIR}")

env = PushShapesEnv(
    object_shape="T",
    pusher_shape="circle",
    obstacle_level=0,
    image_size=96,
    render_mode="rgb_array",
)

DST_DIR.mkdir(parents=True, exist_ok=True)

passed = 0
failed = 0
errors = 0
coverages = []

for i, ep_path in enumerate(episodes):
    try:
        z = zarr.open_group(str(ep_path), mode="r")

        if "episode_init" in z.attrs:
            init = json.loads(z.attrs["episode_init"])
            agent_pos = tuple(init["agent_pos"])
            object_pose = tuple(init["object_pose"])
            goal_pose = tuple(init["goal_pose"])
        else:
            state = np.asarray(z["observations.state"][0])
            agent_pos = (float(state[0]), float(state[1]))
            object_pose = (float(state[2]), float(state[3]), float(state[4]))
            goal = np.asarray(z["goal_pose"][0])
            goal_pose = (float(goal[0]), float(goal[1]), float(goal[2]))

        actions = np.asarray(z["actions"][:])

        env.reset()
        env.set_state(agent_pos=agent_pos, object_pose=object_pose, goal_pose=goal_pose)

        final_cov = 0.0
        for t in range(actions.shape[0]):
            action = actions[t]
            if np.allclose(action, 0.0):
                continue
            obs, rew, term, trunc, info = env.step(action)
            final_cov = info.get("coverage", 0.0)
            if term:
                break

        coverages.append(final_cov)

        if final_cov >= COV_THRESHOLD:
            dst_path = DST_DIR / ep_path.name
            if not dst_path.exists():
                shutil.copytree(str(ep_path), str(dst_path))
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        if (i + 1) % 50 == 0 or i == len(episodes) - 1:
            print(f"[{i+1}/{len(episodes)}] passed={passed} failed={failed} errors={errors} last={ep_path.name} cov={final_cov:.3f} [{status}]")

    except Exception as e:
        errors += 1
        print(f"[{i+1}/{len(episodes)}] {ep_path.name}: ERROR - {e}")

coverages = np.array(coverages)
print(f"\n=== SUMMARY ===")
print(f"Total: {len(episodes)}")
print(f"Passed (cov >= {COV_THRESHOLD}): {passed}")
print(f"Failed (cov < {COV_THRESHOLD}): {failed}")
print(f"Errors: {errors}")
if len(coverages) > 0:
    print(f"Mean coverage: {coverages.mean():.3f}")
    print(f"Median coverage: {np.median(coverages):.3f}")
print(f"No-drift dataset: {DST_DIR} ({passed} episodes)")
