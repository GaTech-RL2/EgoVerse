#!/bin/bash
#SBATCH --job-name=mug2_replay_test
#SBATCH --account=cvg-prof-m-2
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/mug2/replay-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/mug2/replay-%j.err
#SBATCH --partition=normal
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
# Replays a processed v2 episode through egomimic.robot.mug2_policy (the deployment wrapper) with
# a trained checkpoint: checks the inference path end-to-end and writes an overlay video.
#   CKPT=... H5=... sbatch -A <account> -p debug run_replay_test.sh

set -eo pipefail
EGOMAIN=${EGOMAIN:-/iopsstor/scratch/cscs/jiaqchen/code/EgoVerse-main}; export EGOMAIN
CKPT=${CKPT:-/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/FULL_1n/eve_mug2_bc/mug2_BC_1n4g_e1000/checkpoints/last.ckpt}
H5=${H5:-/iopsstor/scratch/cscs/jiaqchen/data/real_mug_v2/processed/20261010_195140.h5}
OUT=${OUT:-/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/mug2/replay_$(basename ${H5%.h5}).mp4}
mkdir -p /iopsstor/scratch/cscs/jiaqchen/egomim_out/main/mug2
srun --environment=/users/jiaqchen/.edf/faive2lerobot.toml --container-workdir=${EGOMAIN} bash -lc '
set -eo pipefail
EGOMAIN='"${EGOMAIN}"'; export EGOMAIN
source ${EGOMAIN}/eth_clariden/clariden.sh
cd ${EGOMAIN}
python -m egomimic.robot.mug2_policy --ckpt "'"${CKPT}"'" --h5 "'"${H5}"'" --out "'"${OUT}"'" --stride '"${STRIDE:-3}"'
'
