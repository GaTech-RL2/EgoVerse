"""
Sanity-check the sim-init code by REPLAYING a recorded episode through
``PushShapesEnv``: read frame-0 state + goal_pose + the full action
sequence directly from the zarr, ``env.set_state`` to the recorded
initial conditions, then step the env with the recorded actions and
render frames. Save as .mp4.

If this replay video looks like the original dataset image sequence,
then ``SimRolloutEval._init_env`` is initialising the env correctly
and any drift in closed-loop rollouts is the model's fault. If the
replay diverges, the init code or the state-encoding contract is wrong.

Run:
    python scripts/replay_episode_in_sim.py \\
        --episode /coc/cedarp-dxu345-0/Tsim_datasets2/circle/episode_T_circle_obs0_000000.zarr \\
        --max-steps 200
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import simplejpeg
import torch
import torchvision.io as tvio
import zarr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True, help="Path to a .zarr episode dir")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out-dir", default="replay_smoke_out")
    parser.add_argument("--object-shape", default="T")
    parser.add_argument("--pusher-shape", default="circle")
    parser.add_argument("--obstacle-level", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from Tsimulation.pushshapes import PushShapesEnv

    z = zarr.open(args.episode, mode="r")
    print(f"[zarr] keys: {list(z)}")
    state = np.asarray(z["observations.state"])  # (T, 5)
    actions = np.asarray(z["actions"])  # (T, 2)
    goal_pose = np.asarray(z["goal_pose"])  # (T, 3) or (3,)
    if goal_pose.ndim == 1:
        goal_pose = goal_pose[None].repeat(state.shape[0], axis=0)
    # ``z[...]`` is a zarr Array of object-dtype; ``np.asarray`` materialises
    # it into a 1-D numpy array whose entries are real ``bytes`` objects
    # (simplejpeg can decode them directly).
    img_arr = np.asarray(z["observations.images.front_img_1"])  # (T,) bytes
    # Trim trailing zero-state padding to the real episode length.
    real_T = int(np.flatnonzero(np.any(state != 0, axis=1)).max() + 1)
    print(f"[zarr] padded T={state.shape[0]}, real episode length={real_T}")
    T_total = real_T
    T = min(args.max_steps or T_total, T_total)
    print(f"[zarr] T_total={T_total}  T_replay={T}")
    print(f"[zarr] state[0]={state[0]}  goal_pose[0]={goal_pose[0]}")
    print(f"[zarr] actions[0]={actions[0]}  actions[-1]={actions[-1]}")

    env = PushShapesEnv(
        object_shape=args.object_shape,
        pusher_shape=args.pusher_shape,
        obstacle_level=args.obstacle_level,
        render_mode="rgb_array",
    )

    env.reset(seed=0)
    # state = [pusher_x, pusher_y, obj_x, obj_y, obj_angle]
    agent_pos = (float(state[0, 0]), float(state[0, 1]))
    object_pose = (float(state[0, 2]), float(state[0, 3]), float(state[0, 4]))
    goal_tuple = tuple(float(x) for x in goal_pose[0, :3])
    env.set_state(
        agent_pos=agent_pos,
        object_pose=object_pose,
        goal_pose=goal_tuple,
    )
    print(
        f"[env] after set_state: agent_pos={env.agent_pos}  "
        f"object_pose={env.object_pose}  goal_pose={env.goal_pose}"
    )

    # Decode recorded JPEG frames so we can compare side-by-side.
    def _decode(buf):
        # zarr returns a 0-d numpy object array. ``.item()`` is np.bytes_
        # which simplejpeg can't handle directly — convert to plain bytes.
        if isinstance(buf, np.ndarray):
            buf = buf.item()
        if isinstance(buf, np.bytes_):
            buf = bytes(buf)
        return simplejpeg.decode_jpeg(buf, colorspace="RGB")

    env_frames = []
    recorded_frames = []
    coverages = []
    # Frame 0 (after set_state but before step).
    env_frames.append(env.render())
    recorded_frames.append(_decode(img_arr[0]))
    for t in range(T):
        act = actions[t].astype(np.float32)
        _, _, terminated, _, info = env.step(act)
        coverages.append(float(info["coverage"]))
        env_frames.append(env.render())
        # Recorded frame after taking action[t].
        rec_idx = min(t + 1, T_total - 1)
        recorded_frames.append(_decode(img_arr[rec_idx]))
        if terminated:
            print(f"[env] terminated at t={t}  coverage={coverages[-1]:.3f}")
            break

    mean_cov = float(np.mean(coverages)) if coverages else 0.0
    final_cov = coverages[-1] if coverages else 0.0
    print(
        f"[env] frames={len(env_frames)}  mean_coverage={mean_cov:.3f}  "
        f"final_coverage={final_cov:.3f}"
    )

    # Side-by-side: resize recorded (96×96) to match env render (512×512) and
    # concatenate horizontally. Labels can be added later if needed.
    H, W, _ = env_frames[0].shape
    combined = []
    for env_f, rec_f in zip(env_frames, recorded_frames):
        rec_up = cv2.resize(rec_f, (W, H), interpolation=cv2.INTER_NEAREST)
        combined.append(
            np.concatenate([rec_up, env_f], axis=1)
        )  # left=recorded, right=env

    out_path = out_dir / f"{Path(args.episode).stem}_sidebyside.mp4"
    tvio.write_video(
        str(out_path),
        torch.from_numpy(np.stack(combined, axis=0)),
        fps=30,
        video_codec="h264",
    )
    print(f"[saved] {out_path}  (left=recorded@96x96 upscaled, right=env render)")


if __name__ == "__main__":
    main()
