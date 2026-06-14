#!/bin/bash
# Overnight orchestrator: wait for the 3 chunker training jobs, eval each with
# the matching architecture, then print a consolidated coverage + video report.
# Head-node-safe: only squeue polls, sbatch submits, and grep — no heavy compute.
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
TRAIN_JOBS="3283671 3283681 3283682"

echo "[orch] waiting for train jobs: $TRAIN_JOBS"
for i in $(seq 1 240); do
  left=""
  for j in $TRAIN_JOBS; do
    st=$(squeue -j $j -h -o %t 2>/dev/null)
    [ -n "$st" ] && left="$left $j"
  done
  if [ -z "$left" ]; then echo "[orch] all train jobs done"; break; fi
  sleep 120
done

# Resolve each run dir by description prefix (newest match).
BASE=$(ls -td logs/hnet_baseline_nc3/hnet_chunked_test_80ep_* 2>/dev/null | head -1)
ACTOK=$(ls -td logs/hnet_baseline_nc3/chunked_actok_500ep_* 2>/dev/null | head -1)
RATIO2=$(ls -td logs/hnet_baseline_nc3/chunked_ratio2_500ep_* 2>/dev/null | head -1)
echo "[orch] BASE=$BASE"
echo "[orch] ACTOK=$ACTOK"
echo "[orch] RATIO2=$RATIO2"

# Submit evals (matching readout per variant).
EJ_BASE=$(sbatch --parsable --export=ALL,RUNDIR_OVERRIDE=$BASE,DESC_OVERRIDE=chunked_base_final,READOUT=mean_pool scripts/eval_hnet_chunked.sh)
EJ_ACTOK=$(sbatch --parsable --export=ALL,RUNDIR_OVERRIDE=$ACTOK,DESC_OVERRIDE=chunked_actok_final,READOUT=action_token scripts/eval_hnet_chunked.sh)
EJ_RATIO2=$(sbatch --parsable --export=ALL,RUNDIR_OVERRIDE=$RATIO2,DESC_OVERRIDE=chunked_ratio2_final,READOUT=mean_pool scripts/eval_hnet_chunked.sh)
echo "[orch] eval jobs: BASE=$EJ_BASE ACTOK=$EJ_ACTOK RATIO2=$EJ_RATIO2"

for i in $(seq 1 90); do
  left=""
  for j in $EJ_BASE $EJ_ACTOK $EJ_RATIO2; do
    st=$(squeue -j $j -h -o %t 2>/dev/null)
    [ -n "$st" ] && left="$left $j"
  done
  if [ -z "$left" ]; then echo "[orch] all evals done"; break; fi
  sleep 60
done

echo "================ CONSOLIDATED RESULTS ================"
for tag in chunked_base_final chunked_actok_final chunked_ratio2_final; do
  d=$(ls -td logs/reeval_corrected/re_${tag}_* 2>/dev/null | head -1)
  o=$(ls -t logs/sbatch/eval_chunked_*.out 2>/dev/null | xargs grep -l "re_${tag}" 2>/dev/null | head -1)
  echo "---- $tag ----"
  echo "rundir=$d"
  grep -E "emb15_sim_coverage|EVAL_EXIT" $o 2>/dev/null | tail -3
  echo "per-episode final_cov:"
  grep -E "final_cov" $o 2>/dev/null | tail -20
  echo "videos:"
  ls -t $d/videos/epoch_0/PUSHSHAPES_SIM/*.mp4 2>/dev/null | head -4
done
echo "================ END ================"
