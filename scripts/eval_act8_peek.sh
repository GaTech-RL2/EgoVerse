#!/bin/bash
#SBATCH --job-name=evalAct8Peek
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/evalpeek_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/evalpeek_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200

# Env-passed: TAG, NAME, MODELCFG, DS, GOAL(random|paper)
L=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs
ROOT=$L/$NAME
# highest-epoch checkpoint across all of this run's (resume-fragmented) dirs
CKPTF=$(ls $ROOT/*/checkpoints/epoch_epoch=*.ckpt 2>/dev/null | sed -E "s/.*epoch=([0-9]+)\.ckpt/\1 &/" | sort -n | tail -1 | cut -d" " -f2-)
NORMF=$(ls -t $ROOT/*/norm_stats/norm_stats.json 2>/dev/null | head -1)
EP=$(echo "$CKPTF" | sed -E "s/.*epoch=([0-9]+)\.ckpt/\1/")
RUNDIR=$(dirname $(dirname $CKPTF))
export ROLLOUT_VIDEO_DIR=$RUNDIR/eval/video_peek_k8_randinit_thr95_ep${EP}
mkdir -p "$ROLLOUT_VIDEO_DIR"
# Hydra parses '=' in the ckpt path (epoch_epoch=NNN.ckpt) as a 2nd override ->
# symlink to a '='-free name so ckpt_path= passes cleanly.
SAFECKPT=$ROLLOUT_VIDEO_DIR/_peekckpt.ckpt
ln -sf "$CKPTF" "$SAFECKPT"

FG=""
if [ "$GOAL" = "paper" ]; then FG="++evaluator.fixed_goal=[256.0,256.0,0.7853981633974483]"; fi

srun python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=evalPeek${TAG} description=peek_act8_k8_ep${EP} mode=eval \
  data=tsimulation model=${MODELCFG} evaluator=eval_hpt_standard \
  ++model.robomimic_model.head_specs.pushshapes_sim.action_horizon=8 \
  ++model.robomimic_model.head_specs.pushshapes_sim.model.act_seq=8 \
  ckpt_path=$SAFECKPT \
  evaluator.init_mode=seeds \
  '+evaluator.init_seeds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]' \
  ${FG} \
  '+evaluator.rollout_mode=chunk_openloop' '+evaluator.chunk_k=8' '+evaluator.temporal_ensemble=false' \
  evaluator.max_steps=500 evaluator.coverage_threshold=0.95 \
  evaluator.limit_val_batches=1 trainer.limit_val_batches=1 \
  trainer=debug trainer.profiler=null logger=csv \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DS \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DS \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_causal \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=egomimic.rldb.embodiment.pushshapes.get_keymap_causal_eval \
  data.train_datasets.pushshapes_sim.resolver.key_map.action_horizon=8 \
  data.valid_datasets.pushshapes_sim.resolver.key_map.action_horizon=8 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=20 \
  norm_stats.norm_mode=quantile \
  norm_stats.precomputed_norm_path=$NORMF
echo "EVAL_EXIT=$?"
