---
name: egoengine-batch-train
description: Process one or more raw RBY1 teleop HDF5 files through the EgoEngine LeRobot pipeline and launch a SLURM training job per dataset on Skynet. Use when the user provides HDF5 paths and asks to "process and train" / "run egoengine on these" / "kick off training for these teleop files".
---

# EgoEngine Batch Process + Train

For each input HDF5, this pipeline:

1. **HDF5 → LeRobot raw** via `egomimic/rldb/scripts/robomimic_hd5.py` using
   `egomimic/rldb/configs/RBY1_egoengine_HDF5_config.json`
   (obs: `images`, `hand_right_qpos`, `eef_proprio`; actions: `eef`, `hand_right_cmd_qpos`).
2. **Combine actions** via `egomimic/scripts/egoengine_process/egoengine_lerobot_combine_action.py`
   to produce `actions.right_arm_eef_hand` (18-D).
3. **Train** with config `train_rby1_egoengine_eef_hand_img_act32`.

All three steps run **inside a single SLURM job per dataset** (overcap, 1× A40, 24h). The driver script just submits N jobs; data processing happens on the GPU node, not the head node.

## Naming convention

For raw file `/.../teleop_<obj>_<n>.hdf5`:
- LeRobot dataset dir: `datasets/RBY1_egoengine_teleop_<obj>_<n>/`
- Intermediate raw dir: `datasets/RBY1_egoengine_teleop_<obj>_<n>_lerobot_raw/`
- Training run `name`: `RBY1_egoengine_teleop_<obj>_<n>` (same as dataset)

The dataset name always mirrors the HDF5 basename (without `.hdf5`).

## How to run

From repo root `/coc/flash7/zhenyang/EgoVerse`:

```bash
# Default 10-file batch (drawer/flower/hammer/mustard):
./egoengine_batch_workflow.sh

# Or pass explicit HDF5 paths:
./egoengine_batch_workflow.sh /path/to/a.hdf5 /path/to/b.hdf5
```

Steps 1 and 2 are idempotent — they skip if the output dir already exists, so re-running only re-submits training.

## Files

- `egoengine_batch_workflow.sh` — driver loop (data processing + sbatch submit).
- `submit_egoengine_training.sbatch` — single-job SLURM script; expects `DATASET` env var (set by the driver via `--export=ALL,DATASET=...`).

## When invoked

1. Confirm the list of HDF5 paths with the user (or use the defaults baked into the script).
2. Run `./egoengine_batch_workflow.sh <paths...>` from the repo root.
3. After submission, report back the squeue job IDs and where logs land
   (`./logs/slurm/<jobid>/task_0/{out,err}.log`).

## Episode-count sanity check

The sbatch script verifies that LeRobot `total_episodes` (in `meta/info.json`) equals the HDF5 demo count after Step 1 and again after Step 2. Mismatch → job exits non-zero, no training. Skip-if-exists logic also reads `total_episodes` and rebuilds the dir if it's partial — this prevents a previous-run truncation (e.g. from a `scancel`) from being silently inherited. Whenever you cancel jobs mid-Step-1, the script self-heals on resubmit; you should *not* need to manually `rm -rf` raw dirs.

After all jobs finish, audit with:

```bash
for f in /coc/flash7/zhenyang/datasets/teleop_*.hdf5; do
  base=$(basename "$f" .hdf5)
  fin=datasets/RBY1_egoengine_${base}/meta/info.json
  exp=$(python -c "import h5py;print(len(h5py.File('$f','r')['data'].keys()))")
  got=$(python -c "import json;print(json.load(open('$fin'))['total_episodes'])" 2>/dev/null || echo MISSING)
  printf "%-26s expected=%-3s got=%s\n" "$base" "$exp" "$got"
done
```

## Gotchas

- The egoengine pipeline is distinct from the SEW pipeline in `SEW_data_workflow.sh` —
  do **not** use `RBY1_SEW_lowdim_HDF5_config.json` or `egoengine_lerobot_extract_arm_hand.py` here.
- Training reads the dataset folder as both train and valid split (per existing convention in `submit_training.sbatch`).
- The `.ckpt` bakes in the `DataSchematic` + norm stats — editing configs after training is launched does not change inference behavior.
