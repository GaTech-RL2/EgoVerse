#!/bin/bash
# H200 throughput profile of the encdec H-Net (cotrain cell), batch sweep.
# One alloc reused; per-config wall time + csv Timing columns.
source /etc/profile 2>/dev/null; source ~/.bashrc 2>/dev/null
G=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
DS=/storage/project/r-dxu345-0/paphiwetsa3/datasets
cd $G
source .venv/bin/activate
export PATH="/storage/project/r-dxu345-0/paphiwetsa3/install/bin:$PATH"
export MUJOCO_GL=egl PYTHONPATH=.
OUT=prof/h200_profile_results.txt
: > $OUT

JID=$(salloc -A gts-dxu345-rl2 -q inferno --partition=gpu-h200 --nodes=1 \
      --gres=gpu:h200:1 --cpus-per-task=8 --mem=250G --time=2:00:00 --no-shell 2>&1 \
      | grep -oE "job allocation [0-9]+" | grep -oE "[0-9]+" | head -1)
echo "alloc=$JID" | tee -a $OUT
sleep 8
squeue -j $JID -h -o "%T %R" | tee -a $OUT

DATA_OV="++data.train_datasets.pushshapes_sim.resolver.folder_path=$DS/circle_3000 \
 ++data.valid_datasets.pushshapes_sim.resolver.folder_path=$DS/circle_3000 \
 ++data.train_datasets.pushshapes_sim_small_circle.resolver.folder_path=$DS/small_circle_3000 \
 ++data.valid_datasets.pushshapes_sim_small_circle.resolver.folder_path=$DS/small_circle_3000"

for BS in 11 33 64 128; do
  echo "===== BS=$BS =====" | tee -a $OUT
  T0=$(date +%s)
  srun --overlap --jobid=$JID .venv/bin/python -m egomimic.trainHydra \
    --config-name=train_zarr_cartesian \
    +experiment=indomain_c4/hnet_ed_cotrain \
    logger=csv name=prof description=h200_bs$BS \
    launch_params.gpus_per_node=1 ++trainer.strategy=ddp_find_unused_parameters_true \
    ++trainer.precision=32 \
    ++norm_stats.precomputed_norm_path=$G/prof/norm_stats_cotrain.json \
    $DATA_OV \
    ++data.train_dataloader_params.pushshapes_sim.batch_size=$BS \
    ++data.train_dataloader_params.pushshapes_sim_small_circle.batch_size=$BS \
    ++data.train_dataloader_params.pushshapes_sim.num_workers=8 \
    ++data.train_dataloader_params.pushshapes_sim_small_circle.num_workers=8 \
    ++trainer.max_epochs=3 ++trainer.min_epochs=3 ++trainer.limit_train_batches=30 \
    ++trainer.limit_val_batches=0 ++trainer.num_sanity_val_steps=0 \
    ++model.scheduler_interval=epoch ++model.scheduler.max_steps=3 ++model.scheduler.warmup_steps=1 \
    ++model.enable_grad_norm=false \
    ++callbacks.ratio_loss_scheduler.on_value=1.0 ++callbacks.ratio_loss_scheduler.off_value=1.0 \
    ++callbacks.model_checkpoint.every_n_epochs=999 \
    > prof/h200_bs$BS.log 2>&1
  RC=$?
  T1=$(date +%s)
  echo "BS=$BS rc=$RC wall=$((T1-T0))s" | tee -a $OUT
  # per-epoch timing from the run's csv
  CSV=$(ls -t logs/prof/h200_bs${BS}_*/csv_logs/lightning_logs/version_0/metrics.csv 2>/dev/null | head -1)
  [ -n "$CSV" ] && awk -F, 'NR==1{for(i=1;i<=NF;i++){if($i=="Timing/Forward_Pass_Sec")f=i;if($i=="Timing/Process_Batch_Sec")p=i;if($i=="epoch")e=i}} NR>1&&$e!=""{printf "  ep%s fwd=%.3fs data=%.3fs\n",$e,$f,$p}' "$CSV" | tee -a $OUT
  # GPU peak mem
  grep -aE "CUDA out of memory" prof/h200_bs$BS.log > /dev/null && echo "  BS=$BS OOM" | tee -a $OUT
done
scancel $JID
echo "PROFILE_DONE" | tee -a $OUT
