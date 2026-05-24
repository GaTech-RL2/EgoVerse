---
name: dexmimicgen-batch-train
description: Process DexMimicGen no-mobile RBY1 robomimic HDF5 files into LeRobot format and launch EgoVerse policy training, including vanilla and hierarchical joint-action configs. Use when the user asks to run or automate DexMimicGen training from HDF5 datasets.
---

# DexMimicGen Batch Process + Train

Use this workflow for DexMimicGen no-mobile RBY1 HDF5 files like:

`/coc/flash7/zhenyang/datasets/demo_first_100_policy_training/can_sort_mink_eef_100_final.hdf5`

## Pipeline

1. Convert HDF5 to LeRobot with:
   `egomimic/rldb/scripts/robomimic_hd5.py`
2. Use conversion config:
   `egomimic/rldb/configs/RBY1_dexmimicgen_no_mobile_HDF5_config.json`
3. Skip `egomimic/scripts/egoengine_process/egoengine_lerobot_extract_arm_hand.py`.
   The policy consumes raw `actions.joint`, so no SEW/EgoEngine derived-action step is required.
4. Train with one of:
   - `experiments/dexmimicgen/train_dexmimicgen_no_mobile_joint_act32`
   - `experiments/dexmimicgen/train_dexmimicgen_no_mobile_hierarchical_p1_masked_attn`
   - `experiments/dexmimicgen/train_dexmimicgen_no_mobile_hierarchical_p2_block_heads`

## Run

From `/coc/flash7/zhenyang/EgoVerse`:

```bash
./dexmimicgen_batch_workflow.sh --monitor /path/to/data.hdf5
```

Choose another training config:

```bash
./dexmimicgen_batch_workflow.sh \
  --train-config experiments/dexmimicgen/train_dexmimicgen_no_mobile_hierarchical_p1_masked_attn \
  --monitor \
  /path/to/data.hdf5
```

## Verified Schema Facts

From dataset inspection and `verify_replay_from_collected.py`:

- `actions.joint` is 47-D.
- Order is `left_arm(7), right_arm(7), torso(6), head(2), base_delta(3), left_hand(11), right_hand(11)`.
- Obs keys used for training are:
  - `obs.frontview_image`
  - `obs.robot0_joint_pos`
  - `obs.robot0_left_gripper_qpos`
  - `obs.robot0_right_gripper_qpos`
- The dataset is no-mobile/fixed-base; `base_delta` remains in `actions.joint` because replay ignores it but the raw action contains it.

## Assumptions To State Back

When using or modifying this pipeline, explicitly mention these assumptions:

- Training should predict raw `actions.joint` rather than a reordered derived action.
- `frontview_image` is the only camera input.
- `robot0_joint_pos` is the 22-D robot proprio input.
- Left/right gripper qpos are 11-D each and correspond to the 11-D hand command blocks.
- The LeRobot train and valid loaders both point at the same converted dataset, following the existing EgoVerse convention.
- FPS is 10 unless the user overrides `--fps`.
- Hierarchical configs preserve raw action order. P1 can mask attention across all blocks; P2 decodes sequentially, so its parents only point to earlier blocks.

## Files

- `dexmimicgen_batch_workflow.sh` submits one job per HDF5.
- `submit_dexmimicgen_training.sbatch` converts and trains inside the SLURM job.
- `egomimic/rldb/configs/RBY1_dexmimicgen_no_mobile_HDF5_config.json` defines HDF5 keys.
- Hydra configs live under:
  - `egomimic/hydra_configs/experiments/dexmimicgen/`
  - `egomimic/hydra_configs/model/experiments/dexmimicgen/`
  - `egomimic/hydra_configs/data/experiments/dexmimicgen/`

## Gotchas

- Do not use `RBY1_SEW_lowdim_HDF5_config.json`; it expects SEW keys that are absent here.
- Do not use `egoengine_lerobot_extract_arm_hand.py`; it expects a 49-D SEW action vector and 12-D hands.
- If a job is cancelled during conversion, resubmitting is safe: the sbatch script checks `total_episodes` and rebuilds partial outputs.
