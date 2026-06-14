#!/bin/bash
# Iteration-2 orchestrator: wait for combo + ratio1.5, eval each, report.
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
TRAIN_JOBS="3283692 3283693"
echo "[orch2] waiting for: $TRAIN_JOBS"
for i in $(seq 1 240); do
  left=""
  for j in $TRAIN_JOBS; do
    st=$(squeue -j $j -h -o %t 2>/dev/null); [ -n "$st" ] && left="$left $j"
  done
  if [ -z "$left" ]; then echo "[orch2] train done"; break; fi
  sleep 120
done
COMBO=$(ls -td logs/hnet_baseline_nc3/chunked_actok_ratio2_500ep_* 2>/dev/null | head -1)
R15=$(ls -td logs/hnet_baseline_nc3/chunked_ratio15_500ep_* 2>/dev/null | head -1)
echo "[orch2] COMBO=$COMBO"
echo "[orch2] R15=$R15"
EJ_C=$(sbatch --parsable --export=ALL,RUNDIR_OVERRIDE=$COMBO,DESC_OVERRIDE=combo_final,READOUT=action_token scripts/eval_hnet_chunked.sh)
EJ_R=$(sbatch --parsable --export=ALL,RUNDIR_OVERRIDE=$R15,DESC_OVERRIDE=ratio15_final,READOUT=mean_pool scripts/eval_hnet_chunked.sh)
echo "[orch2] eval jobs: COMBO=$EJ_C R15=$EJ_R"
for i in $(seq 1 90); do
  left=""
  for j in $EJ_C $EJ_R; do
    st=$(squeue -j $j -h -o %t 2>/dev/null); [ -n "$st" ] && left="$left $j"
  done
  if [ -z "$left" ]; then echo "[orch2] evals done"; break; fi
  sleep 60
done
echo "================ ITER2 RESULTS ================"
for tag in combo_final ratio15_final; do
  d=$(ls -td logs/reeval_corrected/re_${tag}_* 2>/dev/null | head -1)
  o=$(ls -t logs/sbatch/eval_chunked_*.out 2>/dev/null | xargs grep -l "re_${tag}" 2>/dev/null | head -1)
  echo "---- $tag ----"
  echo "rundir=$d"
  grep -E "emb15_sim_coverage|EVAL_EXIT" $o 2>/dev/null | tail -3
  grep -E "final_cov" $o 2>/dev/null | tail -20
  ls -t $d/videos/epoch_0/PUSHSHAPES_SIM/*.mp4 2>/dev/null | head -2
done
echo "================ END2 ================"
