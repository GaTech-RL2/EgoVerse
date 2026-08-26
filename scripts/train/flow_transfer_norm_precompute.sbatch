#!/bin/bash
# Precompute one immutable full-data normalization artifact for each cotrain
# horizon. Its obstacle-Chain entry is also used by the matching BC arm.

set -Eeuo pipefail

ARM=${ARM:?submit with ARM=cotrain_obstacle_latent or cotrain_obstacle_dp}
EXPECTED_HEAD=${EXPECTED_HEAD:?submit with EXPECTED_HEAD}
EXPECTED_SCRIPT_SHA=${EXPECTED_SCRIPT_SHA:?submit with EXPECTED_SCRIPT_SHA}

REPO=/coc/flash7/paphiwetsa3/worktrees/flow-transfer-direct-dense-obstacle-dp-schedulefix-20260826
PY_ENV=/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv
EXPECTED_ENV_SHA=8fd1504c955756adf8167f7bd34fc1a09cb844d268898b3ac30693e99ac60e87
EXP_ROOT=/coc/flash7/paphiwetsa3/experiments/flow_transfer_direct_dense_obstacle3k_cotrain_world2_normfix_20260826
SCRIPT=$REPO/scripts/train/flow_transfer_norm_precompute.sbatch
U_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/u_socket_3000_v2_clean
export CHAIN_OBSTACLE_ROOT=/coc/flash7/paphiwetsa3/experiments/chain_gripper_obstacle_mimicgen_128_20260826/output_3000_balanced_v1
OBSTACLE_AUDIT=$CHAIN_OBSTACLE_ROOT/audit_report.json
OBSTACLE_MANIFEST=$CHAIN_OBSTACLE_ROOT/subset_manifest.json
EXPECTED_OBSTACLE_AUDIT_SHA=1f7f341b2929b1f0d4e9178cc832fd0818c82c2d6086c644634bc4b56f4978dc
EXPECTED_OBSTACLE_MANIFEST_SHA=b5c9385b3b100023552601b050a5b1e6c8bf8cf270d2936f8f9373f57484b6b6
EXPECTED_OBSTACLE_INVENTORY_SHA=1c1433966f1da12ce15380daf4cd4a47213fb92c31738546a992cc4457f4cc1d
EXPECTED_U_COUNT=2999
EXPECTED_U_SHA=fb6e107b585d6a5fae9e9ea4186f441b625a680a36d5691469d9c4125cd9a0fa
EXPECTED_U_TRAIN_FRAMES=521676
EXPECTED_CHAIN_TRAIN_FRAMES=1272946
SLURM_BIN=/opt/slurm/Ubuntu-20.04/24.11.0/bin
export PATH=$SLURM_BIN:$PATH

case "$ARM" in
  cotrain_obstacle_latent)
    EXPERIMENT=pusht/pipeline_sampler_usocket_chain_obstacle_dense_medium
    ACTION_HORIZON=100
    ;;
  cotrain_obstacle_dp)
    EXPERIMENT=pusht/pipeline_diffusion_usocket_chain_obstacle_h16
    ACTION_HORIZON=16
    ;;
  *)
    printf 'Unknown ARM=%s\n' "$ARM" >&2
    exit 64
    ;;
esac

OUT=$EXP_ROOT/norm_artifacts/$ARM/job_${SLURM_JOB_ID}
ART=$OUT/artifact/norm_stats/norm_stats.json

environment_manifest() {
  "$PY_ENV/bin/python" - <<'PY'
import importlib.metadata
import platform

print(f"python=={platform.python_version()}")
entries = set()
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name")
    if name:
        normalized = name.lower().replace("_", "-")
        entries.add(f"{normalized}=={distribution.version}")
for entry in sorted(entries):
    print(entry)
PY
}

test -x "$PY_ENV/bin/python"
test "$(environment_manifest | sha256sum | awk '{print $1}')" = "$EXPECTED_ENV_SHA"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$SCRIPT" | awk '{print $1}')" = "$EXPECTED_SCRIPT_SHA"
test "${SLURM_JOB_NUM_NODES:?}" = 1
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" -ge 8
test ! -e "$OUT"
mkdir -p "$OUT/provenance"
environment_manifest > "$OUT/provenance/python_environment.txt"
sha256sum "$OUT/provenance/python_environment.txt" > "$OUT/provenance/python_environment.sha256"

inventory() {
  local root=$1 expected_count=$2 expected_sha=$3 output=$4
  find "$root" -mindepth 1 -maxdepth 1 -xtype d -name '*.zarr' -printf '%f\n' | LC_ALL=C sort > "$output"
  test "$(wc -l < "$output")" = "$expected_count"
  test "$(sha256sum "$output" | awk '{print $1}')" = "$expected_sha"
}
inventory "$U_DATA" "$EXPECTED_U_COUNT" "$EXPECTED_U_SHA" "$OUT/provenance/usocket_inventory.txt"

: > "$OUT/provenance/chain_obstacle_inventory.txt"
for level_number in $(seq 1 30); do
  level=$(printf '%02d' "$level_number")
  level_root=$CHAIN_OBSTACLE_ROOT/level_${level}/chain_gripper/T
  test "$(find "$level_root" -mindepth 1 -maxdepth 1 -xtype d -name '*.zarr' | wc -l)" = 100
  find "$level_root" -mindepth 1 -maxdepth 1 -xtype d -name '*.zarr' -printf "level_${level}/%f\n" | LC_ALL=C sort >> "$OUT/provenance/chain_obstacle_inventory.txt"
done
test "$(wc -l < "$OUT/provenance/chain_obstacle_inventory.txt")" = 3000
test "$(sha256sum "$OUT/provenance/chain_obstacle_inventory.txt" | awk '{print $1}')" = "$EXPECTED_OBSTACLE_INVENTORY_SHA"
test "$(sha256sum "$OBSTACLE_AUDIT" | awk '{print $1}')" = "$EXPECTED_OBSTACLE_AUDIT_SHA"
test "$(sha256sum "$OBSTACLE_MANIFEST" | awk '{print $1}')" = "$EXPECTED_OBSTACLE_MANIFEST_SHA"
cmp -s "$CHAIN_OBSTACLE_ROOT/inventory.txt" "$OUT/provenance/chain_obstacle_inventory.txt"

"$PY_ENV/bin/python" - "$OBSTACLE_AUDIT" "$OBSTACLE_MANIFEST" <<'PY'
import json
import pathlib
import sys

audit = json.loads(pathlib.Path(sys.argv[1]).read_text())
manifest = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert audit["status"] == "PASS"
assert audit["levels"] == list(range(1, 31))
assert audit["target_total_per_level"] == 100
assert audit["total_episodes"] == 3000
assert audit["total_frames"] == 1272946
assert audit["manual_per_level"] == 16
assert audit["generated_per_level"] == 84
assert len(manifest["episodes"]) == 3000
assert manifest["total_frames"] == 1272946
assert manifest["target_per_level"] == 100
PY

cp "$OBSTACLE_AUDIT" "$OUT/provenance/obstacle_audit_report.json"
cp "$OBSTACLE_MANIFEST" "$OUT/provenance/obstacle_subset_manifest.json"
cp "$SCRIPT" "$OUT/provenance/launcher.sbatch"
git -C "$REPO" log -1 --format=fuller > "$OUT/provenance/git_commit.txt"
git -C "$REPO" status --porcelain=v1 --untracked-files=all > "$OUT/provenance/git_status.txt"
$SLURM_BIN/scontrol show job -dd "$SLURM_JOB_ID" > "$OUT/provenance/slurm_job.txt"

RUNTIME_ENV=$OUT/provenance/runtime_env
mkdir "$RUNTIME_ENV"
ln -s "$PY_ENV" "$RUNTIME_ENV/emimic"
cd "$RUNTIME_ENV"
source emimic/bin/activate
cd "$REPO"
export PYTHONPATH=$REPO
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

ARGS=(
  --config-name=train_zarr_cartesian
  "+experiment=$EXPERIMENT"
  "hydra.run.dir=$OUT/hydra_run"
  norm_stats.norm_mode=minmax
  norm_stats.reduce_all_but_last=true
  norm_stats.sample_frac=1.0
  norm_stats.num_workers=4
  "norm_stats.save_cache_dir=$OUT/artifact"
  norm_stats.precomputed_norm_path=null
  mode=norm_stats
)

python -m egomimic.trainHydra "${ARGS[@]}" --cfg job > "$OUT/resolved_config.yaml"
python - "$OUT/resolved_config.yaml" "$ACTION_HORIZON" <<'PY'
import pathlib
import sys

from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
horizon = int(sys.argv[2])
assert cfg.seed == 42
assert cfg.norm_stats.norm_mode == "minmax"
assert cfg.norm_stats.reduce_all_but_last is True
assert cfg.norm_stats.sample_frac == 1.0
assert cfg.norm_stats.precomputed_norm_path is None
assert cfg.mode == "norm_stats"
assert cfg.model.enable_grad_norm is False
assert set(cfg.data.train_datasets) == {
    "pushshapes_sim_u_socket",
    "pushshapes_sim_chain_gripper",
}
assert set(cfg.data.valid_datasets) == set(cfg.data.train_datasets)
assert all(
    float(dataset.valid_ratio) == 0.0
    for dataset in cfg.data.train_datasets.values()
)
assert all(
    dataset.mode == "train"
    for dataset in cfg.data.train_datasets.values()
)
assert all(
    float(dataset.valid_ratio) == 0.02
    for dataset in cfg.data.valid_datasets.values()
)
assert all(
    dataset.mode == "valid"
    for dataset in cfg.data.valid_datasets.values()
)
u = cfg.data.train_datasets.pushshapes_sim_u_socket.resolver
chain = cfg.data.train_datasets.pushshapes_sim_chain_gripper.resolver
assert u.key_map.action_horizon == horizon
assert chain.key_map.action_horizon == horizon
assert len(chain.folder_paths) == 30
assert str(chain.folder_paths[0]).endswith("/level_01/chain_gripper/T")
assert str(chain.folder_paths[-1]).endswith("/level_30/chain_gripper/T")
assert "arc_length" not in pathlib.Path(sys.argv[1]).read_text().lower()
PY

python -m egomimic.trainHydra "${ARGS[@]}"
test -s "$ART"

python - "$ART" "$ARM" "$OUT/validation.json" "$ACTION_HORIZON" \
  "$EXPECTED_U_TRAIN_FRAMES" "$EXPECTED_CHAIN_TRAIN_FRAMES" <<'PY'
import hashlib
import json
import math
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
arm = sys.argv[2]
out = pathlib.Path(sys.argv[3])
action_horizon = int(sys.argv[4])
expected_u_frames = int(sys.argv[5])
expected_chain_frames = int(sys.argv[6])
b = p.read_bytes()
x = json.loads(b)
assert x["norm_mode"] == "minmax"
assert x["reduce_all_but_last"] is True
metadata = x["norm_run_metadata"]
expected_frames = {"19": expected_u_frames, "20": expected_chain_frames}
assert x["frames"] == sum(expected_frames.values())
assert metadata["total_dataset_frames"] == sum(expected_frames.values())
assert metadata["total_sampled_frames"] == sum(expected_frames.values())
assert set(metadata["embodiments"]) == set(expected_frames)
for embodiment, expected in expected_frames.items():
    item = metadata["embodiments"][embodiment]
    assert item["dataset_size"] == expected
    assert item["sampled_frames"] == expected
    assert item["sample_frac"] == 1.0
    assert item["seed"] == 42
    assert item["max_samples"] is None
assert set(x["stats"]) == {"19", "20"}
stat_names = {
    "mean", "std", "min", "max", "median", "quantile_1",
    "quantile_99", "quantile_0_01", "quantile_99_99",
}
for emb, dims in {
    "19": {"state_agent_obj": 6, "actions": 4},
    "20": {"state_agent_obj": 6, "actions": 6},
}.items():
    assert set(x["stats"][emb]) == set(dims)
    for key, width in dims.items():
        stats = x["stats"][emb][key]
        assert set(stats) == stat_names
        for values in stats.values():
            assert isinstance(values, list)
            assert len(values) == width
            assert all(math.isfinite(float(value)) for value in values)
        assert all(a <= b for a, b in zip(stats["min"], stats["max"]))
        assert all(value >= 0 for value in stats["std"])
active_stats = {
    embodiment: {
        key: {name: x["stats"][embodiment][key][name] for name in ("min", "max")}
        for key in sorted(x["stats"][embodiment])
    }
    for embodiment in sorted(x["stats"])
}
active_bytes = json.dumps(
    active_stats, sort_keys=True, separators=(",", ":")
).encode()
active_sha = hashlib.sha256(active_bytes).hexdigest()
sha = hashlib.sha256(b).hexdigest()
out.write_text(json.dumps({
    "status": "PASS",
    "arm": arm,
    "action_horizon": action_horizon,
    "artifact": str(p),
    "sha256": sha,
    "bytes": len(b),
    "per_embodiment_train_frames": expected_frames,
    "total_train_frames": x["frames"],
    "sample_frac": 1.0,
    "seed": 42,
    "active_minmax_sha256": active_sha,
}, indent=2, sort_keys=True) + "\n")
print(sha)
PY

sha256sum "$ART" > "$OUT/norm_stats.sha256"
chmod 0444 "$ART" "$OUT/norm_stats.sha256" "$OUT/validation.json"
