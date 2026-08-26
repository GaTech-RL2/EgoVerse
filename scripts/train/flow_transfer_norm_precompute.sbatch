#!/bin/bash
# Precompute one immutable full-data normalization artifact for each cotrain
# horizon. Its obstacle-Chain entry is also used by the matching BC arm.

set -Eeuo pipefail

ARM=${ARM:?submit with ARM=cotrain_obstacle_latent or cotrain_obstacle_dp}
EXPECTED_HEAD=${EXPECTED_HEAD:?submit with EXPECTED_HEAD}
EXPECTED_SCRIPT_SHA=${EXPECTED_SCRIPT_SHA:?submit with EXPECTED_SCRIPT_SHA}

REPO=/coc/flash7/paphiwetsa3/worktrees/flow-transfer-direct-dense-obstacle-dp-schedulefix-20260826
PY_ENV=/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv
EXP_ROOT=/coc/flash7/paphiwetsa3/experiments/flow_transfer_direct_dense_obstacle_cotrain_world2_fix_20260826
SCRIPT=$REPO/scripts/train/flow_transfer_norm_precompute.sbatch
U_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/u_socket_3000_v2_clean
CHAIN_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_3000_v2
export CHAIN_OBSTACLE_ROOT=/coc/flash7/paphiwetsa3/experiments/chain_gripper_obstacle_mimicgen_128_20260826/output_128
OBSTACLE_AUDIT=$CHAIN_OBSTACLE_ROOT/audit_report.json
EXPECTED_OBSTACLE_AUDIT_SHA=a06ce0def2ec62f5ea1491dc0efc7ab451bfd9e4eaaf16b2f7f84537c394f2bb
EXPECTED_U_COUNT=2999
EXPECTED_U_SHA=fb6e107b585d6a5fae9e9ea4186f441b625a680a36d5691469d9c4125cd9a0fa
EXPECTED_CHAIN_COUNT=3000
EXPECTED_CHAIN_SHA=3ca50bf2558a3afbe11784a0b3d6424d3d3d9b1d9a564470618d1b64e2e99ede
SLURM_BIN=/opt/slurm/Ubuntu-20.04/24.11.0/bin
export PATH=$SLURM_BIN:$PATH

case "$ARM" in
  cotrain_obstacle_latent)
    EXPERIMENT=pusht/pipeline_sampler_usocket_chain_obstacle_dense_medium
    ACTION_HORIZON=100
    EXPECTED_LAST_FRAMES=0
    ;;
  cotrain_obstacle_dp)
    EXPERIMENT=pusht/pipeline_diffusion_usocket_chain_obstacle_h16
    ACTION_HORIZON=16
    EXPECTED_LAST_FRAMES=0
    ;;
  *)
    printf 'Unknown ARM=%s\n' "$ARM" >&2
    exit 64
    ;;
esac

OUT=$EXP_ROOT/norm_artifacts/$ARM/job_${SLURM_JOB_ID}
ART=$OUT/artifact/norm_stats/norm_stats.json

test -x "$PY_ENV/bin/python"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$SCRIPT" | awk '{print $1}')" = "$EXPECTED_SCRIPT_SHA"
test "${SLURM_JOB_NUM_NODES:?}" = 1
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" -ge 8
test ! -e "$OUT"
mkdir -p "$OUT/provenance"

inventory() {
  local root=$1 expected_count=$2 expected_sha=$3 output=$4
  find "$root" -mindepth 1 -maxdepth 1 -xtype d -name '*.zarr' -printf '%f\n' | LC_ALL=C sort > "$output"
  test "$(wc -l < "$output")" = "$expected_count"
  test "$(sha256sum "$output" | awk '{print $1}')" = "$expected_sha"
}
inventory "$U_DATA" "$EXPECTED_U_COUNT" "$EXPECTED_U_SHA" "$OUT/provenance/usocket_inventory.txt"
inventory "$CHAIN_DATA" "$EXPECTED_CHAIN_COUNT" "$EXPECTED_CHAIN_SHA" "$OUT/provenance/chain_clean_inventory.txt"

: > "$OUT/provenance/chain_obstacle_inventory.txt"
for level_number in $(seq 1 30); do
  level=$(printf '%02d' "$level_number")
  level_root=$CHAIN_OBSTACLE_ROOT/level_${level}/chain_gripper/T
  test "$(find "$level_root" -mindepth 1 -maxdepth 1 -type d -name '*.zarr' | wc -l)" = 128
  find "$level_root" -mindepth 1 -maxdepth 1 -type d -name '*.zarr' -printf "level_${level}/%f\n" | LC_ALL=C sort >> "$OUT/provenance/chain_obstacle_inventory.txt"
done
test "$(wc -l < "$OUT/provenance/chain_obstacle_inventory.txt")" = 3840
test "$(sha256sum "$OBSTACLE_AUDIT" | awk '{print $1}')" = "$EXPECTED_OBSTACLE_AUDIT_SHA"

cp "$OBSTACLE_AUDIT" "$OUT/provenance/obstacle_audit_report.json"
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
  "~logger"
  "~callbacks"
  "~evaluator"
  mode=train
  trainer.accelerator=cpu
  trainer.devices=1
  trainer.strategy=auto
  trainer.num_nodes=1
  trainer.sync_batchnorm=false
  trainer.precision=32-true
  trainer.max_steps=0
  trainer.max_epochs=-1
  trainer.min_epochs=null
  trainer.limit_train_batches=0
  trainer.limit_val_batches=0
  trainer.val_check_interval=1
  trainer.num_sanity_val_steps=0
  +trainer.enable_checkpointing=false
  +trainer.enable_progress_bar=false
  +trainer.enable_model_summary=false
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
assert cfg.trainer.accelerator == "cpu"
assert cfg.trainer.devices == 1
assert cfg.trainer.strategy == "auto"
assert cfg.trainer.precision == "32-true"
assert cfg.trainer.max_steps == 0
assert cfg.trainer.limit_train_batches == 0
assert cfg.trainer.limit_val_batches == 0
assert set(cfg.data.train_datasets) == {
    "pushshapes_sim_u_socket",
    "pushshapes_sim_chain_gripper",
}
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

python - "$ART" "$ARM" "$OUT/validation.json" "$EXPECTED_LAST_FRAMES" <<'PY'
import hashlib
import json
import math
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
arm = sys.argv[2]
out = pathlib.Path(sys.argv[3])
expected_last_frames = int(sys.argv[4])
b = p.read_bytes()
x = json.loads(b)
assert x["norm_mode"] == "minmax"
assert x["reduce_all_but_last"] is True
assert x["frames"] > 0
assert x["norm_run_metadata"]["frames"] == x["frames"]
if expected_last_frames:
    assert x["frames"] == expected_last_frames
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
sha = hashlib.sha256(b).hexdigest()
out.write_text(json.dumps({
    "status": "PASS",
    "arm": arm,
    "artifact": str(p),
    "sha256": sha,
    "bytes": len(b),
    "last_dataset_train_frames": x["frames"],
    "sample_frac": 1.0,
    "seed": 42,
}, indent=2, sort_keys=True) + "\n")
print(sha)
PY

sha256sum "$ART" > "$OUT/norm_stats.sha256"
chmod 0444 "$ART" "$OUT/norm_stats.sha256" "$OUT/validation.json"
