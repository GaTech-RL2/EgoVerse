#!/bin/bash
set -u
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
export COLUMNS=250 PYTHONPATH=. SDL_VIDEODRIVER=dummy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SALLOC=/opt/slurm/Ubuntu-20.04/current/bin/salloc
SRUN=/opt/slurm/Ubuntu-20.04/current/bin/srun
SCANCEL=/opt/slurm/Ubuntu-20.04/current/bin/scancel
DATA=/coc/flash7/paphiwetsa3/datasets/new_circle_3_clean
CKPT=$(ls -t logs/pixelpol_A/A_diffuse_v2*/checkpoints/*.ckpt 2>/dev/null | head -1)
NORM=$(ls -t logs/pixelpol_A/A_diffuse_v2*/norm_stats/norm_stats.json 2>/dev/null | head -1)
echo "ckpt=$CKPT"
echo "norm=$NORM"
JID=$($SALLOC -A rl2-lab -p rl2-lab --time=2:00:00 --gres=gpu:a40:1 --cpus-per-task=8 --mem=48G --no-shell 2>&1 | grep -oP "allocation \K[0-9]+")
echo "alloc=$JID"

run_eval () {
  local DESC=$1; local NCTX=$2; local COMMIT=$3
  $SRUN --jobid=$JID --overlap .venv/bin/python -m egomimic.trainHydra \
    --config-name=train_zarr_cartesian mode=eval name=pixelpol_A_eval description=$DESC \
    ckpt_path="$CKPT" data=tsimulation_full \
    data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
    data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
    data.valid_dataloader_params.pushshapes_sim.batch_size=1 \
    model=dfot/pixel_policy model.robomimic_model.inference_mode=pixel_policy \
    model.robomimic_model.sampler_n_steps=25 \
    ++model.robomimic_model.sp_n_context=$NCTX ++model.robomimic_model.sp_commit=$COMMIT \
    evaluator=eval_dfot_sim evaluator.max_steps=300 evaluator.max_videos=2 \
    trainer=debug trainer.limit_val_batches=1 \
    norm_stats.precomputed_norm_path="$NORM" logger=csv > /tmp/o_A_${DESC}.txt 2>&1
  echo "A_${DESC}_DONE rc=$?  $(grep -aoE 'emb15_sim_coverage[^0-9]+[0-9.]+' /tmp/o_A_${DESC}.txt | tail -1)"
}

run_eval chunk 1 8
run_eval recede 4 1
$SCANCEL $JID 2>/dev/null
echo "ALL_A_EVALS_DONE"
