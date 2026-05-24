#!/bin/bash
# Launch hierarchical training variants (p1_masked_attn + p2_block_heads) for each
# (task, solver) pair whose LeRobot dataset is already on disk.
#
# Usage: ./hierarchical_batch_workflow.sh
# Reads DEFAULT_PAIRS below. Each pair = "<TAG>:<DATASET_ROOT>".
# Where TAG is a short label and DATASET_ROOT is the LeRobot folder name under ./datasets/.

set -e
cd /coc/flash7/zhenyang/EgoVerse

# (label, train_dataset_dir_under_./datasets/)
PAIRS=(
    "basket_sew:RBY1_ye_basket_table_0518_0518_sew_human_data"
    "basket_mink:RBY1_ye_basket_table_0518_0518_mink_human_data"
    "push_cart_sew:RBY1_ye_push_cart_0518_0518_sew_human_data"
    "push_cart_mink:RBY1_ye_push_cart_0518_0518_mink_human_data"
)

# (config-name, suffix appended to run name)
HIER_CONFIGS=(
    "experiments/hierarchical/push_cart_p1_masked_attn:hier_p1"
    "experiments/hierarchical/push_cart_p2_block_heads:hier_p2"
)

JOB_IDS=()
SKIPPED=()
for PAIR in "${PAIRS[@]}"; do
    LABEL="${PAIR%%:*}"
    TRAIN_DATASET="${PAIR##*:}"
    FINAL_DIR=/coc/flash7/zhenyang/EgoVerse/datasets/${TRAIN_DATASET}

    if [ ! -f "${FINAL_DIR}/meta/info.json" ]; then
        echo "SKIP: ${LABEL} — LeRobot dataset not ready at ${FINAL_DIR}"
        SKIPPED+=("${LABEL}")
        continue
    fi

    for HC in "${HIER_CONFIGS[@]}"; do
        CFG="${HC%%:*}"
        SFX="${HC##*:}"
        RUN_NAME="RBY1_${LABEL}_${SFX}"
        JOB_NAME="${SFX}_${LABEL}"

        echo "Submitting ${LABEL} [${SFX}] -> ${RUN_NAME}"
        SUBMIT_OUT=$(sbatch --parsable --job-name="${JOB_NAME}" \
            --export=ALL,TRAIN_CONFIG="${CFG}",TRAIN_RUN_NAME="${RUN_NAME}",TRAIN_DATASET="${TRAIN_DATASET}" \
            submit_train_only.sbatch)
        JOB_IDS+=("${SUBMIT_OUT}")
        echo "  job_id=${SUBMIT_OUT}"
    done
done

echo ""
echo "Submitted ${#JOB_IDS[@]} hierarchical job(s): ${JOB_IDS[*]}"
if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo "Skipped (no LeRobot data yet): ${SKIPPED[*]}"
fi
if [ ${#JOB_IDS[@]} -gt 0 ]; then
    squeue -j "$(IFS=,; echo "${JOB_IDS[*]}")" 2>&1 || true
fi
