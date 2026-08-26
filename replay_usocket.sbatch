#!/bin/bash
#SBATCH --job-name=usock_replay
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/usock_replay_%A_%a.out
# Replay a shard of u_socket episodes, dump per-episode JSON incl. latch + drift.
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
export PYTHONPATH=.
SHARD=${SLURM_ARRAY_TASK_ID}
NSHARD=${NSHARD:-10}
OUT=/coc/flash7/paphiwetsa3/projects/EgoVerse2/usock_replay_out
mkdir -p "$OUT"
.venv/bin/python - "$SHARD" "$NSHARD" "$OUT" <<'PYEOF'
import sys, json, glob, os, traceback
sys.path.insert(0, ".")
from pathlib import Path
from Tsimulation.examples.replay_zarr import replay_one

shard, nshard, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
D = "/coc/flash7/paphiwetsa3/datasets/u_socket_3000"
paths = sorted(glob.glob(os.path.join(D, "episode_*.zarr")))
mine = [p for i, p in enumerate(paths) if i % nshard == shard]
print(f"shard {shard}/{nshard}: {len(mine)} episodes", flush=True)
res = []
for i, p in enumerate(mine):
    try:
        r = replay_one(Path(p), tol=0.05)
    except Exception as e:
        r = {"name": os.path.basename(p), "error": f"{type(e).__name__}: {e}", "ok": False}
        traceback.print_exc()
    res.append(r)
    if i % 25 == 0:
        print(f"  {i}/{len(mine)} ok={sum(1 for x in res if x.get('ok'))}", flush=True)
with open(os.path.join(out, f"shard{shard}.json"), "w") as f:
    json.dump(res, f)
nok = sum(1 for x in res if x.get("ok"))
print(f"SHARD{shard}_DONE ok={nok} fail={len(res)-nok}")
PYEOF
echo "REPLAY_SHARD_${SHARD}_EXIT=$?"
