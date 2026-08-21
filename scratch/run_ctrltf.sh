#!/bin/bash
set -u
export COLUMNS=250
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
JID=$(salloc -A rl2-lab -p rl2-lab --time=1:00:00 --gres=gpu:a40:1 --cpus-per-task=8 --mem=48G --no-shell 2>&1 | grep -oP 'allocation \K[0-9]+')
echo "alloc=$JID"
DATA=/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact/external_ckpts/pushshapes_circle_750_norm_stats.json
srun --jobid=$JID .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian mode=eval \
  name=ctrltf description=policy2d_tf ckpt_path=external_ckpts/policy2d_last.ckpt \
  data=tsimulation_full \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  model=dfot_pushshapes_image_spatial_policy evaluator=eval_dfot_controller_tf evaluator.evals.0.n_steps_eval=120 evaluator.evals.0.max_episodes=12 \
  trainer=debug trainer.limit_val_batches=1 \
  norm_stats.precomputed_norm_path=$NORM logger=csv > /tmp/o_ctrltf.txt 2>&1
echo "TF_DONE rc=$?"
echo "############ TF controller action MSE (vs offline RH 0.017) ############"
grep -E "TF_SANITY|t, pred_norm" /tmp/o_ctrltf.txt | head -10
grep -oE "tf_action_mse_(overall|step_0[0-9]) +[│|] +[0-9.eE+-]+" /tmp/o_ctrltf.txt | sed -E 's/[│|]/ /g; s/  +/ /g' | sort -u
grep -iE "error|exception|Traceback|not in struct|AttributeError|KeyError" /tmp/o_ctrltf.txt | head -6
scancel $JID 2>/dev/null
