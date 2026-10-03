#!/bin/bash
# Pack several small-DDP main-EgoVerse runs on ONE exclusive Lambda loaner node
# (8 x H100), one srun step per run inside the pytorch image, each on its own
# GPUs / DDP port, resuming from its last.ckpt on requeue. The dataset dir is
# copied to node-local disk first (/tmp; the /ephemeral array is root-only on
# the loaner nodes): zarr reads straight off the NFS /workspace stalled nodes
# and left cancelled jobs undead (Ryan, 2026-09-26).
#
#   RUNS="k1 k2 k4" ARGS_k1="--config-name=train_zarr_fold_ladder_rdt1b data=mecka_fold_freeform_opscale1_6d" ... \
#     sbatch --qos=loaner-normal egomimic/scripts/abc_sim/lambda_pack.sbatch
#   GPUS_PER_RUN=2       must match launch_params.gpus_per_node of the configs
#   DATA_SRC=...         NFS dataset dir (default datasets/fold_ladder), staged to node-local /tmp
#   TABLE=...            EGOVERSE_EPISODE_TABLE pickle (default: the one in DATA_SRC)
#   RUN_ROOT=...         default /workspace/users/$USER/runs_abc
#   EXTRA="..."          appended to every run (smokes: trainer.max_epochs=1 ...)
# Onto a RUNNING job's spare GPUs: SLURM_JOB_ID=<job> bash egomimic/scripts/abc_sim/lambda_pack.sbatch
# (steps --overlap its CPUs; they die with that job, so queue an afterany follow-up to resume).
#SBATCH --job-name=abc_pack
#SBATCH --account=loaner
#SBATCH --partition=loaner
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:8
#SBATCH --mem-per-gpu=225G  # the partition's DefMemPerGPU (256000 MB) x 8 exceeds a node
#SBATCH --cpus-per-task=224
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/workspace/users/%u/logs/%x-%j.out
set -euo pipefail
SRC=${SRC:-/workspace/users/$USER/EgoVerse}
IMAGE=${IMAGE:-/workspace/users/$USER/images/nvidia-pytorch-25.06-py3.sqsh}
RUN_ROOT=${RUN_ROOT:-/workspace/users/$USER/runs_abc}
DATA_SRC=${DATA_SRC:-/workspace/users/$USER/datasets/fold_ladder}
TABLE=${TABLE:-$(ls "$DATA_SRC"/episode_table_*.pkl | tail -1)}
DATA=/tmp/$USER/$(basename "$DATA_SRC")
G=${GPUS_PER_RUN:-2}
read -r -a runs <<< "${RUNS:?set RUNS to space-separated run names, each with an ARGS_<name>}"
(( ${#runs[@]} * G <= 8 )) || { echo "${#runs[@]} runs x $G GPUs do not fit one node" >&2; exit 2; }
CPUS=${CPUS_PER_TASK:-$(( 192 / (${#runs[@]} * G) ))}
test -s "$IMAGE" && test -s "$SRC/.venv/pyvenv.cfg" && test -s "$TABLE"
echo "=== abc_pack job=$SLURM_JOB_ID node=$(hostname) restart=${SLURM_RESTART_COUNT:-0} commit=$(git -C "$SRC" rev-parse --short HEAD) runs=${runs[*]}"
# a step, so the copy lands on the compute node also when this file is run with bash
srun --overlap --gres=none --ntasks=1 bash -c "mkdir -p $DATA && rsync -a --exclude _sync $DATA_SRC/ $DATA/ && echo staged \$(du -sh $DATA | cut -f1) to $DATA"

i=0
for name in "${runs[@]}"; do
  var="ARGS_$name"; args=${!var:?set $var}
  port=$(( ${PORT_BASE:-29500} + 10 * i )); i=$(( i + 1 ))
  run="$RUN_ROOT/$name"; mkdir -p "$run"
  resume=""; [ -s "$run/checkpoints/last.ckpt" ] && resume="ckpt_path=$run/checkpoints/last.ckpt"
  echo "--- $name -> $run ${resume:+(resuming)}: $args"
  srun --exact --overlap --ntasks=$G --ntasks-per-node=$G --gres=gpu:h100:$G --gpu-bind=none --cpus-per-task=$CPUS --mem=200G \
    --container-image="$IMAGE" --container-mounts=/workspace:/workspace:rw,$DATA:$DATA:ro --container-workdir="$SRC" \
    --output="$run/train-%j.log" \
    bash -c "set -euo pipefail
      ulimit -n \$(ulimit -Hn)
      export HOME=/workspace/users/$USER HF_HOME=/workspace/users/$USER/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
      export EGOVERSE_DATASET_DIR=$DATA EGOVERSE_EPISODE_TABLE=$TABLE EGOVERSE_LOG_DIR=$RUN_ROOT EGOVERSE_CACHE_DIR=$RUN_ROOT/cache
      export MASTER_PORT=$port PYTHONPATH=$SRC HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=\${WANDB_MODE:-online} WANDB_PROJECT=\${WANDB_PROJECT:-egoverse}
      export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
      export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID}_$name TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID}_$name
      echo rank=\${SLURM_PROCID:-0} nofile=\$(ulimit -n) commit=\$(git -C $SRC rev-parse --short HEAD) job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0}
      exec $SRC/.venv/bin/python egomimic/trainHydra.py $args launch_params.gpus_per_node=$G \
        hydra.run.dir=$run paths.output_dir=$run norm_stats.save_cache_dir=$run $resume ${EXTRA:-}" &
done
wait
