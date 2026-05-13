#!/bin/bash
# EgoEngine batch dispatcher: submit ONE SLURM job per HDF5.
# Each job runs data processing (Step 1 + Step 2) and then training, on a GPU node.
#
# Usage: ./egoengine_batch_workflow.sh [hdf5_path ...]
# If no args, the default 10-file batch below is used.

set -e
cd /coc/flash7/zhenyang/EgoVerse

DEFAULT_INPUTS=(
    # /coc/flash7/zhenyang/datasets/teleop_drawer_20.hdf5
    # /coc/flash7/zhenyang/datasets/teleop_drawer_50.hdf5
    # /coc/flash7/zhenyang/datasets/teleop_flower_1.hdf5
    # /coc/flash7/zhenyang/datasets/teleop_flower_5.hdf5
    # /coc/flash7/zhenyang/datasets/teleop_flower_20.hdf5
    # /coc/flash7/zhenyang/datasets/teleop_flower_50.hdf5
    # /coc/flash7/zhenyang/datasets/teleop_hammer_50.hdf5
    # /coc/flash7/zhenyang/datasets/teleop_mustard_1.hdf5
    # /coc/flash7/zhenyang/datasets/teleop_mustard_5.hdf5
    # /coc/flash7/zhenyang/datasets/teleop_mustard_20.hdf5
    # /coc/flash7/zhenyang/datasets/teleop_mustard_50.hdf5
    # /coc/flash7/zhenyang/datasets/aria_hdf5/drawer_human_20.hdf5
    # /coc/flash7/zhenyang/datasets/aria_hdf5/flower_human_20.hdf5
    # /coc/flash7/zhenyang/datasets/aria_hdf5/hammer_human_20.hdf5
    # /coc/flash7/zhenyang/datasets/aria_hdf5/mustard_egoengine_1.hdf5
    # /coc/flash7/zhenyang/datasets/aria_hdf5/mustard_egoengine_5.hdf5
    # /coc/flash7/zhenyang/datasets/aria_hdf5/mustard_egoengine_20.hdf5
    # /coc/flash7/zhenyang/datasets/aria_hdf5/mustard_egoengine_30.hdf5
    # /coc/flash7/zhenyang/datasets/aria_hdf5/mustard_human_20.hdf5
    /coc/flash7/zhenyang/datasets/aria_hdf5/drawer_egoengine_20.hdf5
    /coc/flash7/zhenyang/datasets/aria_hdf5/flower_egoengine_20.hdf5
    /coc/flash7/zhenyang/datasets/aria_hdf5/hammer_egoengine_20.hdf5
)

INPUTS=("$@")
if [ ${#INPUTS[@]} -eq 0 ]; then
    INPUTS=("${DEFAULT_INPUTS[@]}")
fi

for RAW in "${INPUTS[@]}"; do
    BASENAME=$(basename "${RAW}" .hdf5)
    DATASET_NAME="RBY1_egoengine_${BASENAME}"
    sbatch --job-name="ee_${BASENAME}" \
        --export=ALL,DATASET="${DATASET_NAME}",RAW="${RAW}" \
        submit_egoengine_training.sbatch
done
