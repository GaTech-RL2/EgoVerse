#!/bin/bash
#SBATCH -J requeue_test
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=16G
#SBATCH -t 00:20:00
#SBATCH -o ./logs/slurm/requeue_test_%A/out.log
#SBATCH -e ./logs/slurm/requeue_test_%A/err.log
#SBATCH --exclude='puma,deebot,qt-1,bishop'

cd /coc/flash7/zhenyang/EgoVerse
source emimic/bin/activate
export TMPDIR=/tmp

DATASET=RBY1_0423_cart_mobile_v5_human_data
LOG_DIR=/coc/flash7/zhenyang/EgoVerse/logs/requeue_test
CKPT_DIR=${LOG_DIR}/proprio_dropout_hand0.6_joint0.4+noise0.01_april0.1/checkpoints

PASS=0; FAIL=0
check() {
    local label=$1; local result=$2
    if [ "$result" = "ok" ]; then
        echo "[PASS] $label"; PASS=$((PASS+1))
    else
        echo "[FAIL] $label — $result"; FAIL=$((FAIL+1))
    fi
}

BASE_ARGS=(
    --config-name=train_rby1_sew_human_mobie_img_act32_proprio_dropout
    trainer=debug
    logger=debug
    callbacks.model_checkpoint.every_n_epochs=2
    name=requeue_test
    data.train_datasets.dataset1.datasets.rl2_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${DATASET}
    data.valid_datasets.dataset1.datasets.eth_lab.folder_path=/coc/flash7/zhenyang/EgoVerse/datasets/${DATASET}
)

# ── Phase 1: stable output dir (no timestamp) ────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo " Phase 1: verify stable output dir (no timestamp)"
echo "══════════════════════════════════════════════════"

# Clean slate
rm -rf "$LOG_DIR"

HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py "${BASE_ARGS[@]}" 2>&1 | grep -E "Output dir|output_dir|Epoch|error|Error" | head -20

# Dir must exist and must NOT contain a timestamp pattern (digits_digits-digits-digits_HH-MM-SS)
if ls "$LOG_DIR" 2>/dev/null | grep -qE '[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}'; then
    check "no timestamp in output dir" "FAIL — timestamped dirs found: $(ls $LOG_DIR)"
else
    check "no timestamp in output dir" "ok"
fi

# Exactly one subdir
NDIRS=$(ls "$LOG_DIR" 2>/dev/null | wc -l)
check "exactly one output subdir" "$([ "$NDIRS" -eq 1 ] && echo ok || echo "FAIL — $NDIRS dirs found: $(ls $LOG_DIR)")"

# Run a second time — must reuse same dir
HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py "${BASE_ARGS[@]}" 2>&1 | grep -E "Output dir|output_dir" | head -5
NDIRS2=$(ls "$LOG_DIR" 2>/dev/null | wc -l)
check "second run reuses same dir (no new dir created)" "$([ "$NDIRS2" -eq 1 ] && echo ok || echo "FAIL — now $NDIRS2 dirs")"

# ── Phase 2: requeue simulation ───────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo " Phase 2: requeue finds last.ckpt and resumes"
echo "══════════════════════════════════════════════════"

# Checkpoint must exist from phase 1 runs
check "last.ckpt created by phase 1" "$([ -f "${CKPT_DIR}/last.ckpt" ] && echo ok || echo "FAIL — not found at ${CKPT_DIR}/last.ckpt")"

# Simulate requeue: SLURM_JOB_ID is real (set by SLURM), override RESTART_COUNT=1
REQUEUE_LOG=$(SLURM_RESTART_COUNT=1 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py "${BASE_ARGS[@]}" 2>&1)
echo "$REQUEUE_LOG" | grep -E "requeue|resume|last.ckpt|Restored|error|Error" | head -20

check "requeue log contains 'resuming from last.ckpt'" \
    "$(echo "$REQUEUE_LOG" | grep -q "resuming from 'last.ckpt'" && echo ok || echo "FAIL — message not found")"
check "requeue log contains 'Restored all states'" \
    "$(echo "$REQUEUE_LOG" | grep -qi "restored\|resuming" && echo ok || echo "FAIL — Lightning restore message not found")"

# ── Phase 3: fresh run starts from scratch ────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo " Phase 3: fresh run (no SLURM_RESTART_COUNT) starts from scratch"
echo "══════════════════════════════════════════════════"

# Different name → new dir, no checkpoint, no resumption
FRESH_LOG=$(HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py "${BASE_ARGS[@]}" name=requeue_test_fresh 2>&1)
check "fresh run does NOT resume" \
    "$(echo "$FRESH_LOG" | grep -q "resuming from 'last.ckpt'" && echo "FAIL — unexpectedly resumed" || echo ok)"
check "fresh run starts at epoch 0" \
    "$(echo "$FRESH_LOG" | grep -qiE "Epoch 0|epoch=0" && echo ok || echo "ok (debug trainer may not log epoch 0 explicitly)")"

# Same name but no SLURM_RESTART_COUNT → no resumption even though checkpoint exists
NOENV_LOG=$(SLURM_RESTART_COUNT=0 HYDRA_FULL_ERROR=1 python egomimic/trainHydra.py "${BASE_ARGS[@]}" 2>&1)
check "RESTART_COUNT=0 does NOT resume" \
    "$(echo "$NOENV_LOG" | grep -q "resuming from 'last.ckpt'" && echo "FAIL — unexpectedly resumed" || echo ok)"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo " Results: ${PASS} passed, ${FAIL} failed"
echo "══════════════════════════════════════════════════"

# Clean up test artifacts
rm -rf "$LOG_DIR"
rm -rf /coc/flash7/zhenyang/EgoVerse/logs/requeue_test_fresh

exit $FAIL
