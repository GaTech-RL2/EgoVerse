#!/bin/bash
# Canonical train-only normalization publisher for capacity-controlled
# U-Socket + ChainGripper H16 Diffusion Policy runs.

set -Eeuo pipefail

EXPECTED_HEAD=${EXPECTED_HEAD:?set exact clean source commit}
EXPECTED_LAUNCHER_SHA=${EXPECTED_LAUNCHER_SHA:?set launcher SHA256}
EXPECTED_SPLIT_SHA=${EXPECTED_SPLIT_SHA:?set split manifest SHA256}

REPO=/coc/flash7/paphiwetsa3/worktrees/flow-transfer-dp20m-newdata-h16-20260830
PY_ENV=/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv
EXP_ROOT=/coc/flash7/paphiwetsa3/experiments/flow_transfer_dp20m_newdata_val01_h16_20260830
LAUNCHER=$REPO/scripts/train/flow_transfer_dp_capacity_norm.sbatch
SPLIT_MANIFEST=/coc/flash7/paphiwetsa3/experiments/flow_transfer_dp20m_newdata_val01_h16_20260830/preflight_20260830/dataset_split/split_manifest.json
EXPERIMENT=pusht/pipeline_diffusion_usocket_chain_newdata_val01_h16_20m

assert_sha() {
  local expected=$1 path=$2
  test -s "$path"
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
}

test "${SLURM_RESTART_COUNT:-0}" = 0
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" -ge 8
test -x "$PY_ENV/bin/python"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
assert_sha "$EXPECTED_LAUNCHER_SHA" "$LAUNCHER"
assert_sha "$EXPECTED_SPLIT_SHA" "$SPLIT_MANIFEST"

OUT=$EXP_ROOT/norm/job_${SLURM_JOB_ID:?}
ART=$OUT/artifact/norm_stats/norm_stats.json
test ! -e "$OUT"
mkdir -p "$OUT/provenance"

cp "$LAUNCHER" "$SPLIT_MANIFEST" "$OUT/provenance/"
cp "$(dirname "$SPLIT_MANIFEST")"/*.tsv "$OUT/provenance/"
git -C "$REPO" log -1 --format=fuller > "$OUT/provenance/git_commit.txt"
git -C "$REPO" status --porcelain=v1 --untracked-files=all \
  > "$OUT/provenance/git_status.txt"
/opt/slurm/Ubuntu-20.04/24.11.0/bin/scontrol show job -dd "$SLURM_JOB_ID" \
  > "$OUT/provenance/slurm_job.txt"

source "$PY_ENV/bin/activate"
cd "$REPO"
export PYTHONPATH=$REPO
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

OVERRIDES=(
  --config-name=train_zarr_cartesian
  "+experiment=$EXPERIMENT"
  mode=norm_stats
  "hydra.run.dir=$OUT/hydra_run"
  "++paths.root_dir=$OUT"
  "paths.output_dir=$OUT/hydra_run"
  "paths.work_dir=$REPO"
  norm_stats.norm_mode=minmax
  norm_stats.reduce_all_but_last=true
  norm_stats.sample_frac=1.0
  norm_stats.num_workers=8
  "norm_stats.save_cache_dir=$OUT/artifact"
  norm_stats.precomputed_norm_path=null
)

RESOLVED=$OUT/provenance/resolved_config.yaml
python -m egomimic.trainHydra "${OVERRIDES[@]}" --cfg job --resolve > "$RESOLVED"

python - "$RESOLVED" "$SPLIT_MANIFEST" <<'PY'
import json
import pathlib
import sys

from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
split = json.loads(pathlib.Path(sys.argv[2]).read_text())
domains = {"pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper"}
assert cfg.mode == "norm_stats" and cfg.seed == 42
assert set(cfg.data.train_datasets) == set(cfg.data.valid_datasets) == domains
assert cfg.data.valid_combined_mode == "max_size"
for domain in domains:
    train = cfg.data.train_datasets[domain]
    valid = cfg.data.valid_datasets[domain]
    assert train.mode == "train" and valid.mode == "valid"
    assert float(train.valid_ratio) == float(valid.valid_ratio) == 0.01
assert split["split_seed"] == 42
assert split["domains"]["pushshapes_sim_u_socket"]["train_count"] == 2970
assert split["domains"]["pushshapes_sim_u_socket"]["valid_count"] == 29
assert split["domains"]["pushshapes_sim_chain_gripper"]["train_count"] == 3682
assert split["domains"]["pushshapes_sim_chain_gripper"]["valid_count"] == 37
assert all(v["identity_overlap_count"] == 0 for v in split["domains"].values())
assert all(v["physical_path_overlap_count"] == 0 for v in split["domains"].values())
PY

python -m egomimic.trainHydra "${OVERRIDES[@]}"
test -s "$ART"

python - "$ART" "$OUT/validation.json" "$EXPECTED_HEAD" "$SPLIT_MANIFEST" <<'PY'
import hashlib
import json
import math
import pathlib
import sys

artifact = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
head = sys.argv[3]
split_manifest = pathlib.Path(sys.argv[4])
raw = artifact.read_bytes()
payload = json.loads(raw)
expected_frames = {"19": 516610, "20": 1225343}
assert payload["norm_mode"] == "minmax"
assert payload["reduce_all_but_last"] is True
assert payload["frames"] == sum(expected_frames.values()) == 1741953
metadata = payload["norm_run_metadata"]
assert metadata["total_dataset_frames"] == metadata["total_sampled_frames"] == payload["frames"]
for embodiment, frames in expected_frames.items():
    item = metadata["embodiments"][embodiment]
    assert item["dataset_size"] == item["sampled_frames"] == frames
    assert item["sample_frac"] == 1.0 and item["seed"] == 42
expected_widths = {
    # Normalization is computed on the raw six-value state key. The observation
    # encoder applies input_slice [0, 3] after normalization.
    "19": {"state_agent_obj": 6, "actions": 4},
    "20": {"state_agent_obj": 6, "actions": 6},
}
assert set(payload["stats"]) == set(expected_widths)
for embodiment, keys in expected_widths.items():
    assert set(payload["stats"][embodiment]) == set(keys)
    for key, width in keys.items():
        stats = payload["stats"][embodiment][key]
        for values in stats.values():
            assert len(values) == width
            assert all(math.isfinite(float(value)) for value in values)
record = {
    "status": "PASS",
    "source_commit": head,
    "artifact": str(artifact),
    "artifact_sha256": hashlib.sha256(raw).hexdigest(),
    "split_manifest": str(split_manifest),
    "split_manifest_sha256": hashlib.sha256(split_manifest.read_bytes()).hexdigest(),
    "split_seed": 42,
    "valid_ratio": 0.01,
    "train_window_counts": expected_frames,
    "total_train_windows": payload["frames"],
}
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, sort_keys=True))
PY

test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
assert_sha "$EXPECTED_LAUNCHER_SHA" "$LAUNCHER"
assert_sha "$EXPECTED_SPLIT_SHA" "$SPLIT_MANIFEST"

sha256sum "$ART" > "$OUT/norm_stats.sha256"
sha256sum "$OUT/provenance/"* > "$OUT/provenance/artifacts.sha256"
chmod 0444 "$ART" "$OUT/norm_stats.sha256" "$OUT/validation.json" \
  "$OUT/provenance/"*

printf '[norm] ALL_PASS artifact=%s\n' "$ART"
