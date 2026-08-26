#!/bin/bash
set -u
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
export PYTHONPATH=. SDL_VIDEODRIVER=dummy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SRUN=/opt/slurm/Ubuntu-20.04/current/bin/srun
DATA=/coc/flash7/paphiwetsa3/datasets/new_circle_3_clean
CKPT=logs/pixelpol_A/A_diffuse_v2_2026-05-31_17-27-49/checkpoints/last.ckpt
NORM=logs/pixelpol_A/A_diffuse_v2_2026-05-31_17-27-49/norm_stats/norm_stats.json
JID=$1
run_eval () {
  local DESC=$1; local NCTX=$2; local COMMIT=$3; local AH=$4
  $SRUN --jobid=$JID --overlap .venv/bin/python -m egomimic.trainHydra \
    --config-name=train_zarr_cartesian mode=eval name=pixelpol_A_eval description=$DESC \
    ckpt_path="$CKPT" data=tsimulation_full \
    data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
    data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
    data.valid_dataloader_params.pushshapes_sim.batch_size=1 \
    model=dfot/pixel_policy model.robomimic_model.inference_mode=pixel_policy \
    model.robomimic_model.sampler_n_steps=25 \
    model.robomimic_model.action_horizon=$AH \
    ++model.robomimic_model.sp_n_context=$NCTX ++model.robomimic_model.sp_commit=$COMMIT \
    evaluator=eval_dfot_sim evaluator.max_steps=300 evaluator.max_videos=2 \
    trainer=debug trainer.limit_val_batches=1 \
    norm_stats.precomputed_norm_path="$NORM" logger=csv > /tmp/o_A_${DESC}.txt 2>&1
  echo "A_${DESC}_DONE rc=$?  $(grep -aoE 'emb15_sim_coverage[^0-9]+[0-9.]+' /tmp/o_A_${DESC}.txt | tail -1)"
}
run_eval chunk16 1 16 17
run_eval chunk32 1 32 33
echo "ALL_A_CHUNK_EVALS_DONE"
