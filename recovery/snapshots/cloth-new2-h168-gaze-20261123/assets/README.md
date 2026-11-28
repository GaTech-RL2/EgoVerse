# Cloth new2 1.68m rollout assets
Uses the exact policy 55 train / 5 validation recording split. Gaze: six independent ImageNet-initialized Heat7 members (seeds42–47,30epochs), same standard RGB/255 pipeline. Gen2 RGB is calibrated to the deployed camera KH before training; actual eye-gaze hits only. The five validation recordings select checkpoints and are not an independent test set.
Virtual camera: q22 ridge fit and clamps from training data only.
Reset: native circular mean of first-row torso/arm/head joints; wheel slots zero; arithmetic hand means,55training episodes only; includes native skipped metadata.
No robot motion or rollout has been executed.

Set these alongside the existing cloth policy/depth/crop client setup:
export GAZE_ENS=1
export GAZE_ENS_NAME=cloth_new2
export GAZE_ENS_DIR="<absolute assets path>/gaze"
export VV_MODEL="<absolute assets path>/virtual_cam_cloth_new2.npz"
export RESET_CACHE="<absolute assets path>/reset_pose_cloth_new2.npz"

Run sha256sum -c SHA256SUMS from this folder.
Human validation is not robot-camera validation; inspect geometry-audit.json and gaze-validation.json for measured errors. Do not substitute these files into the 1.60m recipe without evaluating the mismatch.
