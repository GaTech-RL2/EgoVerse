#!/bin/bash
# Closed-loop sim rollout of the 2D spatial policy via the new spatial_rh
# controller. SMOKE: 2 episodes x 300 steps to confirm it runs + produces a
# coverage number. PYTHONPATH=. so PackedSimEval can import Tsimulation.
set -u
export COLUMNS=250
export PYTHONPATH=.
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
JID=$(salloc -A rl2-lab -p rl2-lab --time=2:00:00 --gres=gpu:a40:1 --cpus-per-task=8 --mem=48G --no-shell 2>&1 | grep -oP 'allocation \K[0-9]+')
echo "alloc=$JID"
DATA=/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact/external_ckpts/pushshapes_circle_750_norm_stats.json
srun --jobid=$JID --export=ALL .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian mode=eval \
  name=simrh description=policy2d_sim ckpt_path=external_ckpts/policy2d_last.ckpt \
  data=tsimulation_full \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$DATA \
  model=dfot_pushshapes_image_spatial_policy \
  model.robomimic_model.inference_mode=spatial_rh \
  evaluator=eval_dfot_sim \
  evaluator.max_steps=300 evaluator.max_videos=2 \
  trainer=debug trainer.limit_val_batches=1 \
  norm_stats.precomputed_norm_path=$NORM logger=csv > /tmp/o_simrh.txt 2>&1
echo "SIM_DONE rc=$?"
echo "############ SIM coverage (2D policy, alw=1.0) ############"
grep -oE "sim_(coverage|success_rate) +[│|] +[0-9.eE+-]+" /tmp/o_simrh.txt | sed -E 's/[│|]/ /g; s/  +/ /g' | sort -u
echo "--- per-episode coverage if logged / errors ---"
grep -iE "coverage|error|exception|Traceback|NotImplementedError|not in struct|out of memory|KeyError|shape" /tmp/o_simrh.txt | grep -ivE "INFO|sim_coverage" | head -12
echo "freeing $JID"; scancel $JID 2>/dev/null
