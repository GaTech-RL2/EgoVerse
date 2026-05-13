#!/bin/bash
#SBATCH --job-name=pull_aria_obj_2id_8ev
#SBATCH --account=cvg-prof-m-2
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/obj/aria-pull-2id-8ev-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/main/obj/aria-pull-2id-8ev-%j.err
#SBATCH --partition=normal
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Pulls Aria right_arm for object_in_container BC+2ID+8EV.
# Filters mirror OLD repo `data/obj/multi_data_BC+2ID+8EV.yaml`:
#   indomain (ETH/EPC):     task=object_in_container,           lab=eth,  operator=EPC
#   rl2:                    task=object_in_container_indomain,  lab=rl2
#   song:                   task=object_in_container,           lab=song
#   wang:                   task=object_in_container,           lab=wang
#   eth (Wenkai Xuan):      task=object_in_container,           lab=eth,  operator="Wenkai Xuan"

mkdir -p /iopsstor/scratch/cscs/jiaqchen/egomim_out/main/obj

srun --environment=/users/jiaqchen/.edf/faive2lerobot.toml bash -c '
set -e
source /capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse-main/eth_clariden/clariden.sh

DEST=/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/aria_obj_2id_8ev
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
    ("indomain_eth_epc", {"task": "object_in_container",          "lab": "eth", "operator": "EPC"},          1000),
    ("rl2",              {"task": "object_in_container_indomain", "lab": "rl2"},                            1000),
    ("song",             {"task": "object_in_container",          "lab": "song"},                           1000),
    ("wang",             {"task": "object_in_container",          "lab": "wang"},                           1000),
    ("eth_wenkai",       {"task": "object_in_container",          "lab": "eth", "operator": "Wenkai Xuan"}, 1000),
]
import csv
rows = []
for name, flt, n in streams:
    sub = pick(flt, n)
    print(f"[{name}] {flt} -> {len(sub)} episodes")
    for h, p in sub.values.tolist():
        rows.append((name, h, p))
with open("/iopsstor/scratch/cscs/jiaqchen/aria_obj_2id_8ev_pick.csv", "w") as f:
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
done < /iopsstor/scratch/cscs/jiaqchen/aria_obj_2id_8ev_pick.csv

echo "=== Final state ==="
ls "$DEST" | wc -l
du -sh "$DEST"
'
