#!/bin/bash
set -e
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
L=scripts/train_fused_ar.sh
RN=hnet_pushshapes_fused_pusher_resnet
CONV=hnet_pushshapes_fused_pusher
# Smoke the AR (scheduled-sampling) path on conv, 3 debug epochs + 1 val.
SMOKE=$(sbatch --parsable -J smk_ar --time=00:30:00 \
  --export=ALL,MODEL=$CONV,SSPROB=0.5,EPOCHS=3,VALEVERY=2,DESC=smk_ar $L)
echo "SMOKE=$SMOKE"
# 2 AR runs (conv + resnet), 400ep (AR ~1.7x TF so this fits the 4h cap), gated on smoke.
J1=$(sbatch --parsable -J fused_ar_conv   --dependency=afterok:$SMOKE \
  --export=ALL,MODEL=$CONV,SSPROB=0.5,EPOCHS=400,VALEVERY=100,DESC=fused_ar_conv_sp $L)
J2=$(sbatch --parsable -J fused_ar_resnet --dependency=afterok:$SMOKE \
  --export=ALL,MODEL=$RN,SSPROB=0.5,EPOCHS=400,VALEVERY=100,DESC=fused_ar_resnet_sp $L)
echo "AR_conv=$J1"
echo "AR_resnet=$J2"
echo "=== queue ==="
squeue -u paphiwetsa3 -o "%.10i %.20j %.2t %.6M %R" | grep -E "ar|fused_|JOBID"
