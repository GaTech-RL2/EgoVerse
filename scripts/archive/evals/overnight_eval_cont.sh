#!/bin/bash
# Continue-training orchestrator: wait for the 3 resume runs, extract each
# val-coverage curve (ep500->1200), eval the final ckpts, report.
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
TRAIN_JOBS="3284099 3284100 3284101"
echo "[cont] waiting for: $TRAIN_JOBS"
for i in $(seq 1 240); do
  left=""
  for j in $TRAIN_JOBS; do
    st=$(squeue -j $j -h -o %t 2>/dev/null); [ -n "$st" ] && left="$left $j"
  done
  if [ -z "$left" ]; then echo "[cont] train done"; break; fi
  sleep 120
done

curve () {  # $1 = job log glob tag
  f=$(ls -t logs/sbatch/chunkvar_$1_*.out 2>/dev/null | head -1)
  echo "  log=$f"
  grep -E "HNET_CHUNKTE_DBG" $f 2>/dev/null | \
    awk -F"final_cov=" '{c=$2+0; if($0 ~ /ep=0 /){if(n>0)printf "  ck%d mean=%.3f\n",b,s/n; b++; s=0;n=0} s+=c;n++} END{if(n>0)printf "  ck%d mean=%.3f\n",b,s/n}' | awk '!seen[$0]++'
}

echo "================ COVERAGE CURVES (ep500 -> ep1200) ================"
echo "---- ratio2-cont ----"; curve ratio2-cont
echo "---- actok-cont ----";  curve actok-cont
echo "---- flat-cont ----";   curve flat-cont

# Resolve resume run dirs and eval the final ckpts (proper 400-step, 20-seed).
R2=$(ls -td logs/hnet_baseline_nc3/ratio2_RESUME1200_* 2>/dev/null | head -1)
AK=$(ls -td logs/hnet_baseline_nc3/actok_RESUME1200_* 2>/dev/null | head -1)
FL=$(ls -td logs/hnet_baseline_nc3/flat_nostem_RESUME1200_* 2>/dev/null | head -1)
echo "[cont] R2=$R2  AK=$AK  FL=$FL"
EJ_R2=$(sbatch --parsable --export=ALL,RUNDIR_OVERRIDE=$R2,DESC_OVERRIDE=ratio2_1200,BACKBONE=hnet_chunked,READOUT=mean_pool scripts/eval_hnet_chunked.sh)
EJ_AK=$(sbatch --parsable --export=ALL,RUNDIR_OVERRIDE=$AK,DESC_OVERRIDE=actok_1200,BACKBONE=hnet_chunked,READOUT=action_token scripts/eval_hnet_chunked.sh)
EJ_FL=$(sbatch --parsable --export=ALL,RUNDIR_OVERRIDE=$FL,DESC_OVERRIDE=flat_1200,BACKBONE=flat,READOUT=mean_pool scripts/eval_hnet_chunked.sh)
echo "[cont] eval jobs: $EJ_R2 $EJ_AK $EJ_FL"
for i in $(seq 1 90); do
  left=""
  for j in $EJ_R2 $EJ_AK $EJ_FL; do
    st=$(squeue -j $j -h -o %t 2>/dev/null); [ -n "$st" ] && left="$left $j"
  done
  if [ -z "$left" ]; then echo "[cont] evals done"; break; fi
  sleep 60
done
echo "================ FINAL ep1200 COVERAGE ================"
for tag in ratio2_1200 actok_1200 flat_1200; do
  o=$(ls -t logs/sbatch/eval_chunked_*.out 2>/dev/null | xargs grep -l "re_${tag}" 2>/dev/null | head -1)
  echo "---- $tag ----"
  grep -E "emb15_sim_coverage|EVAL_EXIT" $o 2>/dev/null | tail -2
done
echo "================ ENDCONT ================"
