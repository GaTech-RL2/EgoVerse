# real_mug rig: conventions + calibration (2026-10-07)

Raw demos: `/cluster/scratch/agavryushin/ben_icra/demos/*.h5` (Euler), copied to
`/iopsstor/scratch/cscs/jiaqchen/data/real_mug_30hz/raw`. Single Franka Panda + ORCA right
hand, 30 Hz, fixed overhead front cam + palm wrist cam (432x240), aligned depth.

Scripts ran on Euler (`python3` stack has h5py/scipy/cv2):
- `fk_check.py`, `rot_check.py`, `fk_final.py` — Panda modified-DH FK vs `arm/ee_target`:
  tool offset `[0.1325, -0.0049, 0.0735]` m in flange frame, rotation offset identity,
  quaternion order **xyzw**; residual 7.8 mm / 2.5 deg = controller lag (~3 frames).
- `seam_check.py` — recorded tool frame puts ZYX yaw on the +-180 seam for 15% of frames;
  post-multiplying by Rx(180 deg) gives 0%. Applied in `raw_to_processed.py` to proprio
  and actions (`tool_frame_flip_xyzw` attr). Deployment: post-multiply predicted
  orientation by Rx(180 deg) to recover the recorded tool frame.
- `tag_probe.py`, `handeye_calib.py` — AprilTag 36h11 id 5 on the table; wrist-cam
  hand-eye was too poorly excited to anchor the front camera (kept for reference).
- `front_calib_mug.py` — **used**: front camera pose from mug-in-hand correspondences
  (yellow-mug centroid via RGB+depth vs FK tool point), Kabsch + outlier rejection,
  24 mm RMS. Result registered as `mug_front` in `egomimic/utils/egomimicUtils.py`
  (`EXTRINSICS["mug_front"]["right_cam"] = T_base_cam`, `INTRINSICS["mug_front"]`).
- `raw_to_processed.py` / `validate_processed.py` — raw recorder H5 -> processed FAIVE layout
  consumed by `faive2lerobot/convert_faive_to_lerobot.py` (copies live in the iopsstor
  `code/faive2lerobot` dir). Validator emulates the converter's key parsing.
