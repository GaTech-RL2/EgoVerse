#!/bin/bash
# Roll one checkpoint out in abc_sim on one Lambda H100: policy_server.py in the
# training venv (pytorch image) and sim_client.py in ABC's venv, both in one
# container step so they share the GPU (mjwarp renders on it too).
#
#   CKPT=<epoch_N.ckpt> TASK=sim_put_the_plastic_bottles_in_the_bin OUT=<dir> sbatch --qos=loaner-normal \
#     egomimic/scripts/abc_sim/rollout_lambda.sbatch
#   TASK: any abc_sim name or alias (the sim_224 task_name); NUM_WORLDS=50;
#   POLICY=hold for the no-motion floor (no server); CLIENT_EXTRA e.g. --save-video
# Onto a running job: SLURM_JOB_ID=<job> bash egomimic/scripts/abc_sim/rollout_lambda.sbatch
#SBATCH --job-name=abc_rollout
#SBATCH --account=loaner
#SBATCH --partition=loaner
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/workspace/users/%u/logs/%x-%j.out
set -euo pipefail
SRC=${SRC:-/workspace/users/$USER/EgoVerse}
ABC=${ABC:-/workspace/users/$USER/abc}
IMAGE=${IMAGE:-/workspace/users/$USER/images/nvidia-pytorch-25.06-py3.sqsh}
TASK=${TASK:-sim_put_the_plastic_bottles_in_the_bin}
: "${OUT:?set OUT}"
POLICY=${POLICY:-served}
[ "$POLICY" = hold ] || : "${CKPT:?set CKPT}"
PORT=${PORT:-$(( 8700 + SLURM_JOB_ID % 200 ))}  # set PORT per step when several share a job
mkdir -p "$OUT"
echo "=== abc_rollout job=$SLURM_JOB_ID node=$(hostname) task=$TASK ckpt=${CKPT:-none} policy=$POLICY out=$OUT"
srun --exact --overlap --ntasks=1 --gres=gpu:h100:1 --cpus-per-task=${CPUS_PER_TASK:-16} --mem=120G --container-image="$IMAGE" --container-mounts=/workspace:/workspace:rw --container-workdir="$SRC" \
  bash -c "set -euo pipefail
    export HOME=/workspace/users/$USER HF_HOME=/workspace/users/$USER/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export PYTHONPATH=$SRC PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
    if [ '$POLICY' = served ]; then
      $SRC/.venv/bin/python -m egomimic.scripts.abc_sim.policy_server --ckpt '$CKPT' --port $PORT > '$OUT/server.log' 2>&1 &
      for i in \$(seq 1 120); do curl -fsS http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break; sleep 5; done
      curl -fsS http://127.0.0.1:$PORT/health || { echo 'server never came up'; tail -50 '$OUT/server.log'; exit 1; }
    fi
    cd $ABC
    MUJOCO_EGL_DEVICE_ID=0 .venv/bin/python $SRC/egomimic/scripts/abc_sim/sim_client.py --task '$TASK' \
      --server http://127.0.0.1:$PORT --policy '$POLICY' --out '$OUT' --num-worlds ${NUM_WORLDS:-50} ${CLIENT_EXTRA:-}
    echo '{\"job\": '$SLURM_JOB_ID', \"ckpt\": \"${CKPT:-}\", \"task\": \"$TASK\", \"commit\": \"'\$(git -C $SRC rev-parse --short HEAD)'\"}' > '$OUT/run_info.json'"
