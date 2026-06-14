"""Replay one GT episode through the sim and save a video."""
import numpy as np
import zarr
import torch
import torchvision.io as tvio
from pathlib import Path
from Tsimulation.pushshapes import PushShapesEnv

data_dir = Path("/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle")
ep = sorted([p for p in data_dir.iterdir() if p.name.endswith(".zarr")])[1]

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

frames = []
max_cov = 0.0
for t in range(actions.shape[0]):
    obs, rew, term, trunc, info = env.step(actions[t])
    cov = info.get("coverage", 0.0)
    max_cov = max(max_cov, cov)
    img = obs["image"]
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    # Upscale from 96x96 to 384x384 for visibility
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    img_up = torch.nn.functional.interpolate(img_t, size=(384, 384), mode="nearest")
    frames.append(img_up.squeeze(0).permute(1, 2, 0).byte())
    if term:
        break

video = torch.stack(frames)
out = "/coc/flash7/paphiwetsa3/projects/EgoVerse2/scripts/replay_circle_ep1.mp4"
tvio.write_video(out, video, fps=30, video_codec="h264")
print(f"Saved {out}: {video.shape[0]} frames, max_cov={max_cov:.3f}")
