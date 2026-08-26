#!/bin/bash
set -u
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
export PYTHONPATH=. SDL_VIDEODRIVER=dummy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SRUN=/opt/slurm/Ubuntu-20.04/current/bin/srun
DATA=/coc/flash7/paphiwetsa3/datasets/new_circle_3_clean
CKPT=$(ls -t logs/pixeldec/DEC_*/checkpoints/last.ckpt 2>/dev/null | head -1)
NORM=logs/pixelpol_A/A_diffuse_v2_2026-05-31_17-27-49/norm_stats/norm_stats.json
echo "ckpt=$CKPT"
JID=$1
run_eval () {
  local DESC=$1; local NCTX=$2; local NS=$3
  $SRUN --jobid=$JID --overlap .venv/bin/python -m egomimic.trainHydra \
    --config-name=train_zarr_cartesian mode=eval name=pixeldec_avg description=$DESC \
    ckpt_path="$CKPT" data=tsimulation_full \
    data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
    data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
    data.valid_dataloader_params.pushshapes_sim.batch_size=1 \
    model=dfot/pixel_decoupled model.robomimic_model.inference_mode=pixel_decoupled \
    model.robomimic_model.sampler_n_steps=25 \
    ++model.robomimic_model.sp_n_context=$NCTX ++model.robomimic_model.sp_n_samples=$NS \
    evaluator=eval_dfot_sim evaluator.max_steps=300 evaluator.max_videos=2 \
    evaluator.limit_val_batches=1 "evaluator.init_seeds=[0,1]" \
    trainer=debug trainer.limit_val_batches=1 \
    norm_stats.precomputed_norm_path="$NORM" logger=csv > /tmp/o_DECavg_${DESC}.txt 2>&1
  echo "DECavg_${DESC}_DONE rc=$?  $(grep -aoE 'emb15_sim_coverage[^0-9]+[0-9.]+' /tmp/o_DECavg_${DESC}.txt | tail -1)"
}
run_eval avg_nctx4_s8 4 8
echo ALL_DECAVG_DONE
