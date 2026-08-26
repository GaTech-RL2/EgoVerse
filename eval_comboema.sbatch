#!/bin/bash
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=ig-88,megazord,deebot,cyborg,hk47,spd-13,sonny,kitt,baymax
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/%x_%j.out
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
RUNDIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptFlowComboEma/hptFlowComboEma_2026-07-18_14-29-22
SNAP=/checkpoints/snap_.ckpt
python - <<PY
import torch
ck = torch.load("/checkpoints/last.ckpt", map_location="cpu", weights_only=False)
print("TRUE_EPOCH", ck["epoch"])
if "" == "ema":
    ema = ck["ema_state_dict"]
    sd = ck["state_dict"]
    n = 0
    for k, v in ema.items():
        assert k in sd, k
        sd[k] = v; n += 1
    print("EMA_BAKED", n, "tensors")
    ck.pop("ema_state_dict")
torch.save(ck, "")
PY
export ROLLOUT_VIDEO_DIR=/eval/video__40s
mkdir -p "$ROLLOUT_VIDEO_DIR" logs/sbatch
srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian   name=evalComboEma_ description=eval_comboEma_ mode=eval   data=tsimulation model=hpt_bc_flow_pushshapes_ema evaluator=eval_hpt_standard   "ckpt_path='${SNAP}'"   evaluator.init_mode=seeds   "+evaluator.init_seeds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]"   evaluator.max_steps=1800 evaluator.coverage_threshold=1.01 +evaluator.run_full_horizon=true   evaluator.video_fps=30   evaluator.limit_val_batches=1 trainer.limit_val_batches=1   trainer=debug trainer.profiler=null logger=csv   data.train_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/circle_3000   data.valid_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/circle_3000   norm_stats.precomputed_norm_path=/coc/flash7/paphiwetsa3/projects/EgoVerse2/external_ckpts/hptFlowC3000_circle3000_norm_stats.json
echo EVAL_DONE_
