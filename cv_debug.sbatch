#!/bin/bash
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --exclude=ig-88,spd-13,sonny,cyborg,hk47,heistotron,baymax,deebot,kitt,cheetah,megazord,puma
#SBATCH --output=cv_debug_%j.out
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
export PYTHONPATH=. MUJOCO_GL=egl
.venv/bin/python - <<'PYEOF'
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from egomimic.eval.core.ckpt_loading import load_algo_from_ckpt
from egomimic.eval.probes.eval_boundary_strip import _compose_bprobs_to_frame_level
from egomimic.eval.explorer.export import per_frame_chunk_ids

CK = "eval_bf/bf_cvae_zs_do1_ep499_snap.ckpt"
import glob
D = sorted(glob.glob("logs/indomain_c4/bf_cvae_zs_do1_2026-07-1*"))[-1]
algo, _ = load_algo_from_ckpt(CK, None)
algo.nets.cuda(); algo.nets.eval()
cfg = OmegaConf.load(D + "/.hydra/config.yaml")
dm = instantiate(cfg.data); dm.setup(stage="validate")
first = next(iter(dm.val_dataloader()))
batch = first[0] if isinstance(first, tuple) else first
batch = algo.process_batch_for_training(batch)
viz = algo.collect_chunkviz(batch)
for emb, v in viz.items():
    cu = batch[emb]["cu_seqlens"]
    T = int(cu[-1])
    print(f"emb{emb}: T_total={T} levels={[ (tuple(p.shape), tuple(m.shape), int(m.sum())) for p, m in v['levels'] ]}")
    comp = _compose_bprobs_to_frame_level(v["levels"], T)
    print(f"   composed shapes: {[tuple(c[0].shape) for c in comp]}")
    stages = per_frame_chunk_ids(v["levels"], T)
    print(f"   stages: {[(s[0], len(s[1])) for s in stages]}")
PYEOF
echo CV_DEBUG_DONE
