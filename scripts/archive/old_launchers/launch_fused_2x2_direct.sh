#!/bin/bash
set -e
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
L=scripts/train_fused_variant.sh
RN=hnet_pushshapes_fused_pusher_resnet
CONV=hnet_pushshapes_fused_pusher
# Code already validated by the earlier smoke -> launch the 4 directly, WANDB online.
J1=$(sbatch --parsable -J fused_baseline     --export=ALL,MODEL=$CONV,CHUNKK=1,EPOCHS=500,VALEVERY=100,DESC=fused_baseline_sp $L)
J2=$(sbatch --parsable -J fused_chunk        --export=ALL,MODEL=$CONV,CHUNKK=32,EPOCHS=500,VALEVERY=100,DESC=fused_chunk_sp $L)
J3=$(sbatch --parsable -J fused_resnet       --export=ALL,MODEL=$RN,CHUNKK=1,EPOCHS=500,VALEVERY=100,DESC=fused_resnet_sp $L)
J4=$(sbatch --parsable -J fused_resnet_chunk --export=ALL,MODEL=$RN,CHUNKK=32,EPOCHS=500,VALEVERY=100,DESC=fused_resnet_chunk_sp $L)
echo "RUN_baseline=$J1"
echo "RUN_chunk=$J2"
echo "RUN_resnet=$J3"
echo "RUN_resnet_chunk=$J4"
echo "=== queue ==="
squeue -u paphiwetsa3 -o "%.10i %.20j %.2t %.8M %R" | grep -E "fused_|JOBID"
