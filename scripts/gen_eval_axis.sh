#!/bin/bash
# Generate + submit one ep-399 sim eval sbatch per run. Args: NAME MODELCFG RUNDIR
set -euo pipefail
NAME="$1"; MODELCFG="$2"; RUNDIR="$3"
EV2=/coc/flash7/paphiwetsa3/projects/EgoVerse2
SB=$EV2/scripts/eval_axis_${NAME}.sbatch
CKPT="${RUNDIR}checkpoints/${NAME}_ep400.ckpt"
cat > "$SB" <<EOF
#!/bin/bash
#SBATCH --job-name=ev_${NAME}
#SBATCH --time=01:10:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=${EV2}/logs/sbatch/ev_${NAME}_%j.out
#SBATCH --error=${EV2}/logs/sbatch/ev_${NAME}_%j.err
set -uxo pipefail
cd ${EV2}
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
RUNDIR=${RUNDIR}
cp -n "\${RUNDIR}checkpoints/epoch_epoch=399.ckpt" "${CKPT}"
srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \\
  name=${NAME}_ep399_eval description=${NAME}_axis_rollout mode=eval data=tsimulation \\
  ckpt_path=${CKPT} \\
  model=${MODELCFG} \\
  evaluator=eval_hnet_sim evaluator.rollout_mode=ar \\
  evaluator.init_mode=seeds evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \\
  evaluator.limit_val_batches=1 trainer.limit_val_batches=1 \\
  trainer=debug trainer.precision=32 trainer.profiler=null logger=wandb \\
  data.train_datasets.pushshapes_sim.resolver.folder_path=\$NC3 \\
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=\$KM \\
  data.valid_datasets.pushshapes_sim.resolver.folder_path=\$NC3 \\
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=\$KM \\
  norm_stats.norm_mode=minmax \\
  norm_stats.precomputed_norm_path=\${RUNDIR}norm_stats/norm_stats.json
echo "${NAME}_EXIT=\$?"
EOF
JID=$(sbatch --parsable "$SB")
echo "${NAME} ${JID}"
