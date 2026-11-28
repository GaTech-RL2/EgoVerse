# Cloth new2 1.68 m: gaze and rollout assets

Completed Nov23 10:37 EST. Matches the policy's frozen 55 train / 5 validation recording split: 25,077 training rows and 2,267 validation rows.

| Asset | Result |
| --- | --- |
| RGB-to-gaze | Six Heat7 members, seeds 42–47, 30 epochs each; median 1.6554°, P90 3.6405° on all 2,267 validation frames |
| Center-pixel baseline | Median 11.2329° on the same frames |
| Virtual camera | Train-only q22 ridge fit; held-out median 1.5053 cm / 1.4113° |
| Reset pose | First row of 55 training recordings; native circular joint means, wheel entries zero, and arithmetic hand means |

All six GPU jobs completed with exit 0. The first member passed separate-process full-state resume. Independent audits verified finite learned weights/optimizer state, changed backbone weights, all cache/source label hashes, exact split and native client loading. Native ensemble output matched within 0.000023 pixels. Reset was loaded through the native reader. All 14 packaged files passed SHA checks.

## Skynet paths

Assets directory:
`/coc/flash7/cxin34/cloth-new2-h168-gaze-20261123/assets`

Archive:
`/coc/flash7/cxin34/cloth-new2-h168-gaze-20261123/cloth-new2-h168-rollout-assets-20261123.tar.gz`

Archive SHA256:
`93ca529058d4d6cb134cd90874233f65e77529602e8a273bf4fbdbb8b43bcd1a`

Included: gaze/gaze_heat7_cloth_new2_m0.pt through m5.pt, virtual_cam_cloth_new2.npz, reset_pose_cloth_new2.npz, use-assets.sh, README.md, SHA256SUMS, and validation/compatibility audits. This is the auxiliary rollout asset bundle; policy checkpoints remain in their existing training run.

`source /absolute/path/to/assets/use-assets.sh` sets the asset variables and does not start hardware.

Validation recordings were also used for gaze checkpoint selection, so this is not an independent test set. Errors are measured against recorded gaze/camera labels; robot-camera performance remains unmeasured. No real-robot rollout or reset was executed. Existing 1.60 m / 1.68 m policy checkpoints and unrelated jobs were not modified.
