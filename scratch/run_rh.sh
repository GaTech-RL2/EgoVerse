#!/bin/bash
# Receding-horizon action eval on the current 2D policy (action_loss_weight=1.0).
# Self-allocates its own a40 (spd-13 is busy with the alw5 retrain).
set -u
export COLUMNS=250
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
JID=$(salloc -A rl2-lab -p rl2-lab --time=2:00:00 --gres=gpu:a40:1 --cpus-per-task=8 --mem=48G --no-shell 2>&1 | grep -oP 'allocation \K[0-9]+')
echo "alloc=$JID"
DATA=/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact/external_ckpts/pushshapes_circle_750_norm_stats.json
srun --jobid=$JID .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian mode=eval \
  name=rh description=policy2d_rh ckpt_path=external_ckpts/policy2d_last.ckpt \
  data=tsimulation_full \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  model=dfot/image_spatial_policy evaluator=eval_dfot_image_spatial_policy_rh \
  trainer=debug trainer.limit_val_batches=1 \
  norm_stats.precomputed_norm_path=$NORM logger=csv > /tmp/o_rh.txt 2>&1
echo "RH_DONE rc=$?"
echo "############ RH metrics (current 2D policy, alw=1.0) ############"
grep -oE "rh_[a-z0-9_]+ +[│|] +[0-9.eE+-]+" /tmp/o_rh.txt | sed -E 's/[│|]/ /g; s/  +/ /g' | sort -u
grep -iE "error|exception|Traceback|not in struct|out of memory" /tmp/o_rh.txt | head -5
echo "freeing node $JID"; scancel $JID 2>/dev/null
