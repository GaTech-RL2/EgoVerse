# Human colour-rect policies — deployment note (h_rect / a3r_human, 2026-08-05)

The new human-corpus round: policies trained on `datasets/human_fullpp_rgbd` — COLOUR
images in the slam-rect frame (see `rgbd_data_handoff.md` for the dataset itself).

> CHECKPOINT RULE (user decision 2026-08-05): deploy the LATEST checkpoint
> (`last.ckpt` / `epoch=1999`) — mild IL overfit is acceptable. Applies here and to
> every other deployment guide.

## Status

| policy | what it is | val (27-ep holdout) | deployable? |
|---|---|---|---|
| **h_rect** | ResNet on colour-rect 224, no depth | **0.0651** — best human-corpus policy measured | **YES — via the warp LUT below** |
| a3r_human | Adapt3R colour+depth | 0.0895 (−0.024 vs h_rect) | NO — needs live depth (no real-time stereo exists); also loses offline |

Checkpoint: `logs/RBY1_human_rect/human_rect_resnet_2k/checkpoints/last.ckpt`
(run finishes ep1999 tonight; latest on disk is valid at any time — resume-safe).

## The live image path (the one new deploy component)

Training images = colour in the rectified-left-SLAM frame, built via depth. Live, we
reproduce that frame **depth-free** with a precomputed remap through the solved robot
RGB model (rotation-only; the 9.9 mm rect↔rgb translation is neglected):

**LUT file:** `ai_docs/assets_rect_lut/robot_rect224_lut.npz` (`map_x`, `map_y`)

```python
import numpy as np, cv2
L = np.load("ai_docs/assets_rect_lut/robot_rect224_lut.npz")
MX, MY = L["map_x"], L["map_y"]

def live_image(raw_bgr_640):                      # robot's raw 640x640 fisheye, BGR
    rect = cv2.remap(raw_bgr_640, MX, MY, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rect[..., ::-1].copy()                 # BGR -> RGB. REQUIRED (human-trained!)
```

**Validated** (job 3655240): LUT output vs the depth-exact training-pipeline images —
median 1.44 px, p90 1.52 px at 224 (n=6 frames across sessions); montage
`tmp/lut_vs_exact.png`. Out-of-coverage border = 9.5% of pixels (black, bottom/edges).

### Honest deltas vs the training distribution

1. **Device intrinsics:** trained on the human Aria's rect (f=308.05@512), deployed on
   the robot's (f=307.34) — 0.2%, ≤0.5 px at the image edge. Negligible.
2. **Black border:** training images carried a 4.1% bottom arc; the robot LUT gives
   9.5%. The extra dead margin never touches rows 0–168 (table/hands region) but is a
   mild distribution shift at the border. If rollouts look border-sensitive, this is
   the first suspect.
3. **Parallax:** rotation-only warp ≈1.4 px measured at table distance. Below the
   policy's 20–40 px crop augmentation scale.
4. It is still a **human-data policy** — the perspective/embodiment gap that affected
   all human-corpus policies applies; near-table close-up views remain OOD (HD-round
   finding). Expectation-set accordingly: this A/Bs against hd_wam3/hd_resnet on the
   full task, not against the teleop policies near-table.

## Everything else (unchanged contracts)

- Serve: `python egomimic/scripts/serve_policy.py --checkpoint logs/RBY1_human_rect/human_rect_resnet_2k/checkpoints/last.ckpt --port 8000`
- Obs: `front_img_1` = the (224,224,3) uint8 **RGB** image from `live_image()` above;
  `robot0_joint_pos` 22-D no-wheel `[torso 6, r_arm 7, l_arm 7, head 2]`;
  `hand_left_qpos`/`hand_right_qpos` 12 each. (Trained with proprio dropout 0.9 —
  robust to imperfect proprio, but send real values.)
- Actions: (1, 32, 49) @10 Hz, base = deltas → cumsum in frame-0 heading.
  ⚠ Human-data timing: chunks replay human motion at ~1/6 speed (60 Hz rows labeled
  10 Hz — handoff doc §6). Same as every prior human-corpus deployment.
- Dry run first: `python egomimic/scripts/test_serve_policy_client.py
  --dataset-folder datasets/human_fullpp_rgbd --episode-idx 0 --max-steps 30 --trajectory`
