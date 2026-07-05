#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=0:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
# sym_w1 dualstream on 1x H200 (SINGLE GPU -> no NCCL -> no collective-timeout deaths).
P=/storage/project/r-dxu345-0/paphiwetsa3
DS=$P/projects/EgoVerse-gmm-dualstream
DAT=$P/datasets
cd $DS
export PYTHONPATH=. MUJOCO_GL=egl WANDB_INIT_TIMEOUT=300
export PATH="$P/install/bin:$PATH"
CELL=dualstream_tapered_sym_w1_gmmfix_mlprouter
# resume: latest run-dir ckpt, else the relayed ep949 seed
LAST=$(ls -t logs/indomain_c4/${CELL}_2026*/checkpoints/last.ckpt logs/indomain_c4/${CELL}_2026*/csv_logs/lightning_logs/version_0/checkpoints/last.ckpt 2>/dev/null | head -1)
[ -z "$LAST" ] && LAST=$DS/resume_ckpts/sym_w1_ep949.ckpt
RES="ckpt_path=$LAST"; echo "[RESUME] $CELL <- $LAST"
DATA_OV="++data.train_datasets.pushshapes_sim.resolver.folder_path=$DAT/circle_3000_plus_gen ++data.valid_datasets.pushshapes_sim.resolver.folder_path=$DAT/circle_3000_plus_gen ++data.train_datasets.pushshapes_sim_small_circle.resolver.folder_path=$DAT/small_circle_3000 ++data.valid_datasets.pushshapes_sim_small_circle.resolver.folder_path=$DAT/small_circle_3000"
srun .venv/bin/python -m egomimic.trainHydra --config-name=train_zarr_cartesian $RES \
  +experiment=indomain_c4/dualstream_cotrain_2trunk_d16_200m \
  model=hnet_dualstream_firstend_tapered_200m_sym_w1 \
  trainer=ddp_pi description=$CELL \
  launch_params.gpus_per_node=1 launch_params.nodes=1 ++trainer.strategy=auto \
  model.enable_grad_norm=false \
  norm_stats.precomputed_norm_path=$DS/norm_ref/norm_stats.json \
  trainer.max_epochs=2 trainer.min_epochs=2 \
  trainer.limit_train_batches=2 trainer.limit_val_batches=0 \
  trainer.check_val_every_n_epoch=1000 ++norm_stats.sample_frac=0.1 name=debug \
  +model.scheduler_interval=epoch model.scheduler.max_steps=2000 model.scheduler.warmup_steps=100 \
  callbacks.model_checkpoint.save_last=true callbacks.model_checkpoint.save_top_k=1 \
  callbacks.model_checkpoint.every_n_epochs=50 \
  +callbacks.ckpt_rolling._target_=lightning.pytorch.callbacks.ModelCheckpoint \
  +callbacks.ckpt_rolling.every_n_epochs=25 +callbacks.ckpt_rolling.save_top_k=1 \
  "+callbacks.ckpt_rolling.monitor=epoch" +callbacks.ckpt_rolling.mode=max \
  +callbacks.ckpt_rolling.save_last=true "+callbacks.ckpt_rolling.filename='rolling-{epoch}'" \
  +callbacks.ckpt_rolling.save_on_train_epoch_end=true \
  ${EXTRA:-} \
  +callbacks.ds_probe._target_=egomimic.pl_utils.callbacks.dual_stream_probe.DualStreamProbeCallback
