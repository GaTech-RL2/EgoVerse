#!/bin/bash
set -e
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
L=scripts/train_fused_variant.sh
RN=hnet_pushshapes_fused_pusher_resnet
CONV=hnet_pushshapes_fused_pusher
# 1. Smoke the MAXIMAL new config (ResNet + spatial=True + chunk_k=32) — exercises
#    every new code path at once. 3 debug epochs + 1 val.
SMOKE=$(sbatch --parsable -J smk_rn_sp_ck32 --time=00:30:00 \
  --export=ALL,MODEL=$RN,CHUNKK=32,EPOCHS=3,VALEVERY=2,DESC=smk_rn_sp_ck32 $L)
echo "SMOKE=$SMOKE"
# 2. The 4 full runs (500ep, val every 100), AUTO-START only if the smoke succeeds.
J1=$(sbatch --parsable -J fused_baseline     --dependency=afterok:$SMOKE \
  --export=ALL,MODEL=$CONV,CHUNKK=1,EPOCHS=500,VALEVERY=100,DESC=fused_baseline_sp $L)
J2=$(sbatch --parsable -J fused_chunk        --dependency=afterok:$SMOKE \
  --export=ALL,MODEL=$CONV,CHUNKK=32,EPOCHS=500,VALEVERY=100,DESC=fused_chunk_sp $L)
J3=$(sbatch --parsable -J fused_resnet       --dependency=afterok:$SMOKE \
  --export=ALL,MODEL=$RN,CHUNKK=1,EPOCHS=500,VALEVERY=100,DESC=fused_resnet_sp $L)
J4=$(sbatch --parsable -J fused_resnet_chunk --dependency=afterok:$SMOKE \
  --export=ALL,MODEL=$RN,CHUNKK=32,EPOCHS=500,VALEVERY=100,DESC=fused_resnet_chunk_sp $L)
echo "RUN_baseline=$J1"
echo "RUN_chunk=$J2"
echo "RUN_resnet=$J3"
echo "RUN_resnet_chunk=$J4"
echo "=== queue ==="
squeue -u paphiwetsa3 -o "%.10i %.20j %.2t %.10M %R" | head -20
