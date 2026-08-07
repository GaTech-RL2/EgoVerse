#!/bin/bash
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
export PYTHONPATH=. MUJOCO_GL=egl
echo "== SMOKE: bf_var4_ema (EMACallback + GroupNorm encoder) =="
srun -A hoffman-lab -p hoffman-lab --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G -t 00:45:00 \
  --exclude=omgwth,cyborg,ig-88,hk47,spd-13,sonny,kitt,cheetah,heistotron,megazord,puma,baymax,deebot,megabot \
  .venv/bin/python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  +experiment=indomain_c4/dualstream_cotrain_2trunk_d16_200m \
  model=bf_var4_ema trainer=ddp_pi description=smoke_ema \
  callbacks=checkpoints_ema \
  launch_params.gpus_per_node=1 launch_params.nodes=1 ++trainer.strategy=auto \
  model.enable_grad_norm=false \
  norm_stats.precomputed_norm_path=/coc/flash7/paphiwetsa3/datasets/pushshapes_norm_stats_ws512.json \
  trainer.max_epochs=2 trainer.min_epochs=2 ++trainer.limit_train_batches=2 \
  trainer.limit_val_batches=0 trainer.check_val_every_n_epoch=1 \
  callbacks.model_checkpoint.every_n_epochs=1 ++callbacks.model_checkpoint.save_on_train_epoch_end=true \
  +model.scheduler_interval=epoch model.scheduler.max_steps=3000 model.scheduler.warmup_steps=150
RC=$?
echo "SMOKE_RC=$RC"
[ $RC -ne 0 ] && { echo "SMOKE FAILED"; exit 1; }
CK=$(ls -t logs/indomain_c4/smoke_ema_*/checkpoints/*.ckpt | head -1)
srun -A hoffman-lab -p hoffman-lab --cpus-per-task=2 --mem=16G -t 00:10:00 \
  .venv/bin/python -c "
import torch
ck = torch.load('$CK', map_location='cpu', weights_only=False)
ema = ck.get('ema_state_dict')
assert ema, 'NO ema_state_dict IN CKPT'
import itertools
k = next(iter(ema))
sd = ck['state_dict']
diff = (sd[k].float() - ema[k].float()).abs().max().item()
gn = sum('backbone' in n and 'weight' in n for n in ema)
print(f'EMA_OK tensors={len(ema)} sample_key={k} max_diff_vs_live={diff:.2e}')
"
RC2=$?
echo "EMA_CHECK_RC=$RC2"
[ $RC2 -ne 0 ] && exit 1
J=$(MODEL=bf_var4_ema NAME=bf_var4_ema EXTRA="callbacks=checkpoints_ema" sbatch --parsable -J bf_var4_ema bf_prdec_abl.sbatch)
echo "LAUNCHED bf_var4_ema job=$J"
echo ALL_DONE
