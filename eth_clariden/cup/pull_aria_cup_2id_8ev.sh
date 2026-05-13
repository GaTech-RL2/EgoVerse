#!/bin/bash
#SBATCH --job-name=pull_aria_cup_2id_8ev
#SBATCH --account=cvg-prof-m-2
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup/aria-pull-2id-8ev-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup/aria-pull-2id-8ev-%j.err
#SBATCH --partition=normal
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Pulls Aria for BC+2ID+8EV: indomain + 4 EgoVerse labs (rl2, song, wang, eth).
# Largest pull of the cup variants.

mkdir -p /iopsstor/scratch/cscs/jiaqchen/egomim_out/main/cup

srun --environment=/users/jiaqchen/.edf/faive2lerobot.toml bash -c '
set -e
source /capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse-main/eth_clariden/clariden.sh

DEST=/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/aria_cup_2id_8ev
mkdir -p "$DEST"

python <<PY
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df
load_env()
df = episode_table_to_df(create_default_engine())
def pick(filter_dict, n):
    m = (df["embodiment"].astype(str) == "aria") & df["zarr_processed_path"].astype(str).str.startswith("s3://")
    for k, v in filter_dict.items():
        m &= (df[k].astype(str) == v)
    return df.loc[m, ["episode_hash", "zarr_processed_path"]].head(n)

streams = [
    ("indomain", {"task": "cup_on_saucer_in_domain", "lab": "eth"}, 1000),
    ("rl2",      {"task": "cup_on_saucer", "lab": "rl2", "operator": "rl2"}, 1000),
    ("song",     {"task": "cup_on_saucer", "lab": "song"}, 1000),
    ("wang",     {"task": "cup_on_saucer", "lab": "wang"}, 1000),
    ("eth",      {"task": "cup_on_saucer", "lab": "eth"}, 1000),
]
import csv
rows = []
for name, flt, n in streams:
    sub = pick(flt, n)
    print(f"[{name}] {flt} -> {len(sub)} episodes")
    for h, p in sub.values.tolist():
        rows.append((name, h, p))
with open("/iopsstor/scratch/cscs/jiaqchen/aria_cup_2id_8ev_pick.csv", "w") as f:
    csv.writer(f).writerows(rows)
print(f"total: {len(rows)} episodes")
PY

set -a; source ~/.egoverse_env; set +a
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
unset AWS_SESSION_TOKEN
export AWS_DEFAULT_REGION=auto AWS_REGION=auto

while IFS=, read -r STREAM EPI ZP; do
    STREAM=$(printf "%s" "$STREAM" | tr -d "\r")
    EPI=$(printf "%s" "$EPI" | tr -d "\r")
    ZP=$(printf "%s" "$ZP" | tr -d "\r")
    [ -z "$EPI" ] && continue
    OUT="$DEST/${EPI}.zarr"
    if [ -d "$OUT" ] && [ "$(find "$OUT" -type f 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "skip $EPI (present)"; continue
    fi
    echo "syncing [$STREAM] $EPI"
    rm -rf "$OUT" && mkdir -p "$OUT"
    s5cmd --endpoint-url "$R2_ENDPOINT_URL" --numworkers 8 sync "${ZP}/*" "${OUT}/" 2>&1 | tail -3
done < /iopsstor/scratch/cscs/jiaqchen/aria_cup_2id_8ev_pick.csv

echo "=== Final state ==="
ls "$DEST" | wc -l
du -sh "$DEST"
'
