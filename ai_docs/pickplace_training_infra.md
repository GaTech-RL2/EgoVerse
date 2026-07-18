# Pickplace training infrastructure (validation / diagnostics / eval workflow)

> Built 2026-07-18, BEFORE the final nav_pick_and_place data lands. Nothing here
> changes the optimized training loss — the recipe stays exactly crop100
> (2000 ep, batch 32, RandomResizedCrop 0-100px, proprio dropout 0.9,
> noise_std_raw 0.03, proprio_clamp 3.0). Everything below is measurement.

## 1. What was built

### 1.1 Real-robot validation set (the 5 teleop demos)
`datasets/rby1_teleop_pickplace_val` — LeRobot conversion of
`mobile_generalist/data/0717_unified/fix_head_base_0717_merged.hdf5`
(the 5 verified teleop demos, images resized 640→224 to match the training
contract, standard Step1+2 pipeline). Used as `valid_datasets` during training —
**never trained on**. Why it's the right val set: real robot images from the
robot's head + REAL measured proprio. On human-retargeted train data
proprio ≡ action, so a proprio-leaking policy looks falsely perfect;
the teleop set breaks that degeneracy.

Design decision: **all 5 episodes go to validation** (`mode: train,
valid_ratio: 0.0` on that folder). Norm stats come from the TRAIN dataset only
(trainHydra only calls `infer_norm_from_dataset` on train), so val is normalized
with train stats — exactly like serving.

### 1.2 Loss decomposition (WandB)
- **Train** (`Train/rby1_loss/<block>`): per-block flow-matching MSE (normalized
  space), blocks = base/torso/head/l_arm/r_arm/l_hand/r_hand. Diagnostic only —
  detached, the optimized loss is unchanged. Implemented via
  `loss_block_slices` on the FM head (`denoising_policy.py`) → stashed →
  logged through the existing losses dict.
- **Valid** (`Valid/rby1_actions_..._<block>_mae_avg` + `_mse_avg`): per-block
  errors in UNNORMALIZED units (rad; base in m) — directly comparable to the
  offline gate numbers in `policy_training_status.md`. Full-49D MAE:
  `Valid/..._mae_avg`.
- Existing metrics kept: paired/final MSE, Fréchet, reverse-KL (samples 8→4 to
  pay for 4× more frequent validation).

### 1.3 Nav/manip phase split (validation)
A val sample is **manip phase** iff its GT chunk's total base |dx|+|dy| over the
32-step horizon is < `manip_base_disp_thresh` (default **0.05 m** per 3.2 s —
i.e., the base is essentially parked at the table). Logged:
- `Valid/..._manip_frac` — fraction of val samples in manip phase
- `Valid/..._manip_mae_avg` + per-block (`_manip_<block>_mae_avg`) — watch
  `_manip_r_arm/_r_hand` (the acting arm) and `_manip_l_hand` (should stay put)
- `Valid/..._nav_mae_avg` + per-block — navigation quality
Design decision: phase split is a METRIC, not a loss re-weighting — keeps runs
comparable with all previous rounds. If we later want to up-weight manip, add it
as an explicit experiment.

### 1.4 Aug-image debugging (quota-frugal)
`AugImageLogger` callback (`egomimic/pl_utils/aug_image_logger.py`): every 100
epochs it saves ONE PNG (top row: raw batch images; bottom row: the same images
through the exact `train_image_augs` Compose, de-normalized) to
`<run_dir>/aug_debug/epoch_XXXX_rby1.png`, and mirrors that single small image
to WandB (`AugDebug/rby1`). 2000-epoch run → 20 images total (~4 MB disk).
The PNGs on disk exist even if WandB quota runs out.

### 1.5 Policy-comparison workflow
`egomimic/scripts/eval_on_teleop.py` — the standard "which checkpoint is
better" comparator on the teleop set (replaces the ad-hoc /tmp eval scripts):
```
python egomimic/scripts/eval_on_teleop.py \
  --ckpt crop100=logs/aria_egoposer_firm/crop100_2k/checkpoints/last.ckpt \
  --ckpt pickplace_v1=logs/<...>/checkpoints/last.ckpt \
  --proprio real          # or zero for the vision-only probe
```
Reports overall / nav / manip MAE, per-block, per-episode; writes
`results.json`, `blocks_chart.png` (all-frames + manip-only panels), and a
pred-vs-GT block-mean overlay along one episode.

## 2. Files touched
- `egomimic/models/denoising_policy.py` — `loss_block_slices` kwarg + per-block stash
- `egomimic/algo/hpt.py` — stash plumb (HPTModel.compute_loss → forward_training →
  compute_losses), `RBY1_JOINT49_BLOCKS`, `rby1_joint49_block_metrics`,
  `rby1_manip_phase_mask`, phase metrics in `forward_eval_logging`,
  `manip_base_disp_thresh` kwarg (old ckpts safe: getattr defaults)
- `egomimic/pl_utils/aug_image_logger.py` — new callback
- `egomimic/hydra_configs/model/experiments/wholebody_image/rby1_wb_img_proprio_act32_pickplace.yaml`
- `egomimic/hydra_configs/experiments/wholebody_image/wb_img_proprio_pickplace_v1.yaml`
- `submit_wb_img_training.sbatch` — optional `VAL_DATASET_DIR` env (default
  unchanged: val = split of train data, so all old configs behave identically)
- `egomimic/scripts/eval_on_teleop.py` — new comparator

## 3. Launch (when the final data arrives — ONE command)
```
cd /coc/flash7/czhang883/Documents/EgoVerse
sbatch --job-name=wbimg_pickplace_v1 \
  --exclude=puma,deebot,qt-1,sonny,cyborg,crushinator,ig-88 \
  --export=ALL,DATASET_NAME=aria_pickplace,RAW_DATA_PATH=<FINAL_CUT_HDF5>,TRAIN_CONFIG=experiments/wholebody_image/wb_img_proprio_pickplace_v1,DESCRIPTION=pickplace_v1_2k,VAL_DATASET_DIR=/coc/flash7/czhang883/Documents/EgoVerse/datasets/rby1_teleop_pickplace_val \
  submit_wb_img_training.sbatch
```
The sbatch converts the raw HDF5 (Step 1+2), then trains with the teleop val set.
Auto-resumes across the 24 h wall / preemption. WandB: personal account,
project `sew_policy`.

## 4. What "good" looks like on the val curves
- `Valid/..._mae_avg` decreasing and settling ~0.02-0.05: the policy transfers
  to robot-view images. (It will NOT reach the human-data in-dist ~0.013 — the
  val set is a different embodiment view + real proprio; watch the TREND.)
- `_manip_r_arm/_r_hand` decreasing = manipulation actually being learned;
  `_manip_l_hand` small and flat = left hand staying put, as in the demos.
- `_nav_base` decreasing = navigation transfer.
- If `Valid` plateaus early while `Train` keeps dropping → appearance gap
  dominating; that's the signal to consider DINOv3/conv-neck encoders (R7) or
  WAM co-training for v2.
