#!/bin/bash
# DP-only full launcher gated by the matching world-size-2 A40 real-data smoke.
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a40:2
#SBATCH --requeue
#SBATCH --signal=USR1@300
#SBATCH --open-mode=append

set -Eeuo pipefail

EXPECTED_HEAD=${EXPECTED_HEAD:?set the exact clean 40-character commit SHA}
EXPECTED_LAUNCHER_SHA=${EXPECTED_LAUNCHER_SHA:?set the exact launcher SHA256}
EXPECTED_SMOKE_LAUNCHER_SHA=${EXPECTED_SMOKE_LAUNCHER_SHA:?set the exact DP A40 smoke launcher SHA256}
NORM_ARTIFACT=${NORM_ARTIFACT:?set the exact norm_stats.json path}
EXPECTED_NORM_SHA=${EXPECTED_NORM_SHA:?set the exact norm artifact SHA256}
DP_SMOKE_RESULT=${DP_SMOKE_RESULT:?set the exact DP SMOKE_RESULT.json path}
EXPECTED_DP_SMOKE_RESULT_SHA=${EXPECTED_DP_SMOKE_RESULT_SHA:?set its SHA256}

U_INVENTORY=${U_INVENTORY:?set the exact U-Socket inventory path}
EXPECTED_U_INVENTORY_SHA=${EXPECTED_U_INVENTORY_SHA:?set its SHA256}
U_EPISODE_METADATA=${U_EPISODE_METADATA:?set the exact U-Socket episode metadata TSV path}
EXPECTED_U_EPISODE_METADATA_SHA=${EXPECTED_U_EPISODE_METADATA_SHA:?set its SHA256}
EXPECTED_U_COUNT=${EXPECTED_U_COUNT:?set its exact episode count}
EXPECTED_U_TRAIN_FRAMES=${EXPECTED_U_TRAIN_FRAMES:?set its exact frame count}

CHAIN_BASE_INVENTORY=${CHAIN_BASE_INVENTORY:?set the exact base-Chain inventory path}
EXPECTED_CHAIN_BASE_INVENTORY_SHA=${EXPECTED_CHAIN_BASE_INVENTORY_SHA:?set its SHA256}
CHAIN_BASE_EPISODE_METADATA=${CHAIN_BASE_EPISODE_METADATA:?set the exact base-Chain episode metadata TSV path}
EXPECTED_CHAIN_BASE_EPISODE_METADATA_SHA=${EXPECTED_CHAIN_BASE_EPISODE_METADATA_SHA:?set its SHA256}
EXPECTED_CHAIN_BASE_COUNT=${EXPECTED_CHAIN_BASE_COUNT:?set its exact episode count}

CHAIN_GEN_INVENTORY=${CHAIN_GEN_INVENTORY:?set the exact generated-Chain inventory path}
EXPECTED_CHAIN_GEN_INVENTORY_SHA=${EXPECTED_CHAIN_GEN_INVENTORY_SHA:?set its SHA256}
CHAIN_GEN_EPISODE_METADATA=${CHAIN_GEN_EPISODE_METADATA:?set the exact generated-Chain episode metadata TSV path}
EXPECTED_CHAIN_GEN_EPISODE_METADATA_SHA=${EXPECTED_CHAIN_GEN_EPISODE_METADATA_SHA:?set its SHA256}
EXPECTED_CHAIN_GEN_COUNT=${EXPECTED_CHAIN_GEN_COUNT:?set its exact episode count}
EXPECTED_CHAIN_TRAIN_FRAMES=${EXPECTED_CHAIN_TRAIN_FRAMES:?set the combined effective Chain frame count}

REPO=/coc/flash7/paphiwetsa3/worktrees/flow-transfer-cotrain-newdata-h16-full-20260828
PY_ENV=/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv
EXPECTED_ENV_SHA=8fd1504c955756adf8167f7bd34fc1a09cb844d268898b3ac30693e99ac60e87
EXP_ROOT=/coc/flash7/paphiwetsa3/experiments/flow_transfer_newdata3719_cotrain_h16_20260828
SMOKE_ROOT=$EXP_ROOT
LAUNCHER=$REPO/scripts/train/flow_transfer_newdata_h16_world2_dp_a40_full.sbatch
SMOKE_LAUNCHER=$REPO/scripts/train/flow_transfer_newdata_h16_world2_dp_a40_smoke.sbatch
SMOKE_GATE_VERIFIER=$REPO/egomimic/scripts/verify_newdata_dp_a40_full_smoke_gate.py
U_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/u_socket_3000_v2_clean
CHAIN_BASE_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_3000_v2
CHAIN_GEN_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_gen
EXCLUDED_CHAIN_EPISODE=episode_T_chain_gripper_obs7_000050
EXCLUDED_CHAIN_FOLDER=${EXCLUDED_CHAIN_EPISODE}.zarr
EXCLUDED_CHAIN_FRAMES=3118
EXPECTED_WORLD_SIZE=2
EXPECTED_MAX_STEPS=240000
WANDB_PROJECT=pushshapes-flow-transfer
WANDB_ENTITY=rl2-group
WANDB_GROUP=flow_transfer_newdata3719_cotrain_h16_20260828
SLURM_BIN=/opt/slurm/Ubuntu-20.04/24.11.0/bin
export PATH=$SLURM_BIN:$PATH
SRUN=$SLURM_BIN/srun
EXPERIMENT=pusht/pipeline_diffusion_usocket_chain_newdata_h16
WANDB_ID=ft_cotrain_newdata3719_dp_h16_s42_world2_a40x2_20260828

is_sha256() {
  [[ "$1" =~ ^[0-9a-f]{64}$ ]]
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

environment_manifest() {
  "$PY_ENV/bin/python" - <<'PY'
import importlib.metadata
import platform

print(f"python=={platform.python_version()}")
entries = set()
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name")
    if name:
        entries.add(f"{name.lower().replace('_', '-')}=={distribution.version}")
for entry in sorted(entries):
    print(entry)
PY
}

validate_inventory() {
  local label=$1 root=$2 inventory=$3 inventory_sha=$4
  local metadata=$5 metadata_sha=$6 expected_count=$7
  test -d "$root"
  test -s "$inventory"
  test -s "$metadata"
  test "$(stat -c '%a' "$inventory")" = 444
  test "$(stat -c '%a' "$metadata")" = 444
  test "$(sha256sum "$inventory" | awk '{print $1}')" = "$inventory_sha"
  test "$(sha256sum "$metadata" | awk '{print $1}')" = "$metadata_sha"
  "$PY_ENV/bin/python" - \
    "$label" "$root" "$inventory" "$metadata" "$expected_count" <<'PY'
import hashlib
import json
import pathlib
import sys

label, root_arg, inventory_arg, metadata_arg, expected_count_arg = sys.argv[1:]
root = pathlib.Path(root_arg)
inventory_path = pathlib.Path(inventory_arg)
metadata_path = pathlib.Path(metadata_arg)
expected_count = int(expected_count_arg)
episodes = sorted(
    (path for path in root.iterdir() if path.name.endswith(".zarr") and path.is_dir()),
    key=lambda path: path.name,
)
names = [path.name for path in episodes]
assert len(names) == expected_count, (label, len(names), expected_count)
assert len(names) == len(set(names)), f"{label} has duplicate episode names"
assert inventory_path.read_bytes() == "".join(f"{name}\n" for name in names).encode(), (
    label,
    "live inventory differs from pinned artifact",
)
metadata_rows = []
for episode in episodes:
    zarr_json = episode / "zarr.json"
    raw = zarr_json.read_bytes()
    frames = json.loads(raw)["attributes"]["total_frames"]
    assert isinstance(frames, int) and not isinstance(frames, bool) and frames > 0
    metadata_rows.append(
        f"{episode.name}\t{frames}\t{hashlib.sha256(raw).hexdigest()}\n"
    )
assert metadata_path.read_bytes() == "".join(metadata_rows).encode(), (
    label,
    "live episode metadata differs from pinned artifact",
)
PY
}

validate_all_inventories() {
  validate_inventory usocket "$U_DATA" "$U_INVENTORY" \
    "$EXPECTED_U_INVENTORY_SHA" "$U_EPISODE_METADATA" \
    "$EXPECTED_U_EPISODE_METADATA_SHA" "$EXPECTED_U_COUNT"
  validate_inventory chain_base "$CHAIN_BASE_DATA" "$CHAIN_BASE_INVENTORY" \
    "$EXPECTED_CHAIN_BASE_INVENTORY_SHA" "$CHAIN_BASE_EPISODE_METADATA" \
    "$EXPECTED_CHAIN_BASE_EPISODE_METADATA_SHA" "$EXPECTED_CHAIN_BASE_COUNT"
  validate_inventory chain_gen "$CHAIN_GEN_DATA" "$CHAIN_GEN_INVENTORY" \
    "$EXPECTED_CHAIN_GEN_INVENTORY_SHA" "$CHAIN_GEN_EPISODE_METADATA" \
    "$EXPECTED_CHAIN_GEN_EPISODE_METADATA_SHA" "$EXPECTED_CHAIN_GEN_COUNT"
  "$PY_ENV/bin/python" - \
    "$U_EPISODE_METADATA" "$CHAIN_BASE_EPISODE_METADATA" \
    "$CHAIN_GEN_EPISODE_METADATA" "$EXCLUDED_CHAIN_FOLDER" \
    "$EXCLUDED_CHAIN_FRAMES" "$EXPECTED_U_TRAIN_FRAMES" \
    "$EXPECTED_CHAIN_TRAIN_FRAMES" <<'PY'
import pathlib
import sys

(
    u_metadata_arg,
    chain_base_metadata_arg,
    chain_gen_metadata_arg,
    excluded_folder,
    excluded_frames_arg,
    expected_u_frames_arg,
    expected_chain_frames_arg,
) = sys.argv[1:]


def frames_by_episode(path_arg):
    output = {}
    for row in pathlib.Path(path_arg).read_text().splitlines():
        episode, frames, metadata_sha = row.split("\t")
        assert len(metadata_sha) == 64
        int(metadata_sha, 16)
        output[episode] = int(frames)
    return output


u_frames = frames_by_episode(u_metadata_arg)
base_frames = frames_by_episode(chain_base_metadata_arg)
gen_frames = frames_by_episode(chain_gen_metadata_arg)
excluded_frames = int(excluded_frames_arg)
assert sum(u_frames.values()) == int(expected_u_frames_arg)
assert excluded_folder not in base_frames
assert gen_frames.get(excluded_folder) == excluded_frames
effective_chain_frames = sum(base_frames.values()) + sum(gen_frames.values()) - excluded_frames
assert effective_chain_frames == int(expected_chain_frames_arg), (
    effective_chain_frames,
    int(expected_chain_frames_arg),
)
PY
}

validate_norm() {
  test -s "$NORM_ARTIFACT"
  test "$(stat -c '%a' "$NORM_ARTIFACT")" = 444
  test "$(sha256sum "$NORM_ARTIFACT" | awk '{print $1}')" = "$EXPECTED_NORM_SHA"
  "$PY_ENV/bin/python" - "$NORM_ARTIFACT" "$EXPECTED_NORM_SHA" \
    "$EXPECTED_U_TRAIN_FRAMES" "$EXPECTED_CHAIN_TRAIN_FRAMES" <<'PY'
import hashlib
import json
import math
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected_sha = sys.argv[2]
expected_frames = {"19": int(sys.argv[3]), "20": int(sys.argv[4])}
raw = path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == expected_sha
payload = json.loads(raw)
assert payload["norm_mode"] == "minmax"
assert payload["reduce_all_but_last"] is True
assert payload["frames"] == sum(expected_frames.values())
metadata = payload["norm_run_metadata"]
assert metadata["total_dataset_frames"] == sum(expected_frames.values())
assert metadata["total_sampled_frames"] == sum(expected_frames.values())
assert set(metadata["embodiments"]) == set(expected_frames)
for embodiment, frames in expected_frames.items():
    item = metadata["embodiments"][embodiment]
    assert item["dataset_size"] == frames
    assert item["sampled_frames"] == frames
    assert item["sample_frac"] == 1.0
    assert item["max_samples"] is None
assert set(payload["stats"]) == {"19", "20"}
for embodiment, widths in {
    "19": {"state_agent_obj": 6, "actions": 4},
    "20": {"state_agent_obj": 6, "actions": 6},
}.items():
    for key, width in widths.items():
        for values in payload["stats"][embodiment][key].values():
            assert len(values) == width
            assert all(math.isfinite(float(value)) for value in values)
PY
}

verify_smoke_gate() {
  PYTHONPATH=$REPO "$PY_ENV/bin/python" "$SMOKE_GATE_VERIFIER" \
    --expected-head "$EXPECTED_HEAD" \
    --norm-artifact "$NORM_ARTIFACT" \
    --expected-norm-sha "$EXPECTED_NORM_SHA" \
    --smoke-root "$SMOKE_ROOT" \
    --dp-result "$DP_SMOKE_RESULT" \
    --expected-dp-result-sha "$EXPECTED_DP_SMOKE_RESULT_SHA" \
    --expected-smoke-launcher-sha "$EXPECTED_SMOKE_LAUNCHER_SHA"
}

validate_static_inputs() {
  test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
  test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
  test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$EXPECTED_LAUNCHER_SHA"
  test "$(sha256sum "$SMOKE_LAUNCHER" | awk '{print $1}')" = "$EXPECTED_SMOKE_LAUNCHER_SHA"
  test "$(environment_manifest | sha256sum | awk '{print $1}')" = "$EXPECTED_ENV_SHA"
  test "$(sha256sum "$DP_SMOKE_RESULT" | awk '{print $1}')" = "$EXPECTED_DP_SMOKE_RESULT_SHA"
  validate_norm
  validate_all_inventories
}

test -x "$PY_ENV/bin/python"
test -x "$SRUN"
test -f "$SMOKE_GATE_VERIFIER"
test "${#EXPECTED_HEAD}" = 40
[[ "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]]
for value in \
  "$EXPECTED_LAUNCHER_SHA" "$EXPECTED_SMOKE_LAUNCHER_SHA" \
  "$EXPECTED_NORM_SHA" "$EXPECTED_DP_SMOKE_RESULT_SHA" \
  "$EXPECTED_U_INVENTORY_SHA" "$EXPECTED_U_EPISODE_METADATA_SHA" \
  "$EXPECTED_CHAIN_BASE_INVENTORY_SHA" \
  "$EXPECTED_CHAIN_BASE_EPISODE_METADATA_SHA" \
  "$EXPECTED_CHAIN_GEN_INVENTORY_SHA" \
  "$EXPECTED_CHAIN_GEN_EPISODE_METADATA_SHA"; do
  is_sha256 "$value"
done
for value in \
  "$EXPECTED_U_COUNT" "$EXPECTED_CHAIN_BASE_COUNT" "$EXPECTED_CHAIN_GEN_COUNT" \
  "$EXPECTED_U_TRAIN_FRAMES" "$EXPECTED_CHAIN_TRAIN_FRAMES"; do
  is_positive_integer "$value"
done
# Bind this full-run namespace to the final candidate corpus, not just arbitrary
# externally consistent manifests.
test "$EXPECTED_U_COUNT" = 2999
test "$EXPECTED_CHAIN_BASE_COUNT" = 3000
test "$EXPECTED_CHAIN_GEN_COUNT" = 720
test "$EXPECTED_U_TRAIN_FRAMES" = 521676
test "$EXPECTED_CHAIN_TRAIN_FRAMES" = 1237652
validate_static_inputs

RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
[[ "$RESTART_COUNT" =~ ^[0-9]+$ ]]
test "${SLURM_JOB_NUM_NODES:?}" = 1
test "${SLURM_NTASKS:?}" = "$EXPECTED_WORLD_SIZE"
test "${SLURM_CPUS_PER_TASK:?}" -ge 8
test "${SLURM_JOB_PARTITION:?}" = hoffman-lab
test "${SLURM_JOB_ACCOUNT:?}" = hoffman-lab
test -n "${CUDA_VISIBLE_DEVICES:-}"
IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
test "${#visible_gpus[@]}" = "$EXPECTED_WORLD_SIZE"

RUN_DIR=$EXP_ROOT/runs/dp_a40_seed42
PROVENANCE_DIR=$EXP_ROOT/provenance/full/dp_a40/job_${SLURM_JOB_ID:?}
TRAIN_LOG=$RUN_DIR/training.log
mkdir -p "$EXP_ROOT/runs" "$EXP_ROOT/provenance/full/dp_a40"
if test "$RESTART_COUNT" = 0; then
  test ! -e "$RUN_DIR"
  test ! -e "$PROVENANCE_DIR"
  mkdir "$RUN_DIR" "$PROVENANCE_DIR"
  printf '%s\n' "$SLURM_JOB_ID" > "$RUN_DIR/.owner_job_id"
  WANDB_RESUME=never
else
  test -d "$RUN_DIR"
  test -d "$PROVENANCE_DIR"
  test "$(sed -n '1p' "$RUN_DIR/.owner_job_id")" = "$SLURM_JOB_ID"
  WANDB_RESUME=allow
fi

copy_once_and_verify() {
  local source=$1 destination=$2
  if test -e "$destination"; then
    cmp -s "$source" "$destination"
  else
    cp "$source" "$destination"
  fi
}

nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
  --query-gpu=index,name,uuid,memory.total,driver_version \
  --format=csv,noheader > "$PROVENANCE_DIR/gpu_attempt_${RESTART_COUNT}.txt"
test "$(wc -l < "$PROVENANCE_DIR/gpu_attempt_${RESTART_COUNT}.txt")" = "$EXPECTED_WORLD_SIZE"
test "$(grep -ci A40 "$PROVENANCE_DIR/gpu_attempt_${RESTART_COUNT}.txt")" = "$EXPECTED_WORLD_SIZE"

copy_once_and_verify "$LAUNCHER" "$PROVENANCE_DIR/launcher.sbatch"
copy_once_and_verify "$SMOKE_LAUNCHER" "$PROVENANCE_DIR/smoke_launcher.sbatch"
copy_once_and_verify "$SMOKE_GATE_VERIFIER" "$PROVENANCE_DIR/verify_newdata_dp_a40_full_smoke_gate.py"
copy_once_and_verify "$NORM_ARTIFACT" "$PROVENANCE_DIR/norm_stats.json"
copy_once_and_verify "$DP_SMOKE_RESULT" "$PROVENANCE_DIR/dp_SMOKE_RESULT.json"
copy_once_and_verify "$U_INVENTORY" "$PROVENANCE_DIR/usocket_inventory.txt"
copy_once_and_verify "$U_EPISODE_METADATA" "$PROVENANCE_DIR/usocket_episode_metadata.tsv"
copy_once_and_verify "$CHAIN_BASE_INVENTORY" "$PROVENANCE_DIR/chain_base_inventory.txt"
copy_once_and_verify "$CHAIN_BASE_EPISODE_METADATA" "$PROVENANCE_DIR/chain_base_episode_metadata.tsv"
copy_once_and_verify "$CHAIN_GEN_INVENTORY" "$PROVENANCE_DIR/chain_gen_inventory.txt"
copy_once_and_verify "$CHAIN_GEN_EPISODE_METADATA" "$PROVENANCE_DIR/chain_gen_episode_metadata.tsv"
environment_manifest > "$PROVENANCE_DIR/python_environment_attempt_${RESTART_COUNT}.txt"
test "$(sha256sum "$PROVENANCE_DIR/python_environment_attempt_${RESTART_COUNT}.txt" | awk '{print $1}')" = "$EXPECTED_ENV_SHA"
git -C "$REPO" log -1 --format=fuller > "$PROVENANCE_DIR/git_commit_attempt_${RESTART_COUNT}.txt"
git -C "$REPO" status --porcelain=v1 --untracked-files=all > "$PROVENANCE_DIR/git_status_attempt_${RESTART_COUNT}.txt"
"$SLURM_BIN/scontrol" show job -dd "$SLURM_JOB_ID" > "$PROVENANCE_DIR/slurm_job_attempt_${RESTART_COUNT}.txt"
"$PY_ENV/bin/python" - "$PROVENANCE_DIR/slurm_job_attempt_${RESTART_COUNT}.txt" long 2-00:00:00 1 <<'PY'
import pathlib
import re
import sys

path, expected_qos, expected_time, expected_requeue = sys.argv[1:]
fields = dict(re.findall(r"([A-Za-z][A-Za-z0-9/]*)=([^\s]+)", pathlib.Path(path).read_text()))
assert fields["QOS"] == expected_qos
assert fields["TimeLimit"] == expected_time
assert fields["Requeue"] == expected_requeue
assert "mem=128G" in fields["ReqTRES"].split(",")
PY
verify_smoke_gate > "$PROVENANCE_DIR/smoke_gate_attempt_${RESTART_COUNT}_before.json"

source "$PY_ENV/bin/activate"
cd "$REPO"
export PYTHONPATH=$REPO
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
export WANDB_SILENT=true

COMMON_OVERRIDES=(
  --config-name=train_zarr_cartesian
  "+experiment=$EXPERIMENT"
  mode=train
  "hydra.run.dir=$RUN_DIR"
  "++paths.root_dir=$RUN_DIR"
  "paths.output_dir=$RUN_DIR"
  "paths.work_dir=$REPO"
  launch_params.gpus_per_node=2
  launch_params.nodes=1
  trainer.max_steps=240000
  trainer.limit_train_batches=1.0
  trainer.val_check_interval=10000
  trainer.limit_val_batches=0
  trainer.num_sanity_val_steps=0
  trainer.log_every_n_steps=1
  logger.wandb.offline=false
  "logger.wandb.project=$WANDB_PROJECT"
  "logger.wandb.entity=$WANDB_ENTITY"
  "logger.wandb.group=$WANDB_GROUP"
  "logger.wandb.id=$WANDB_ID"
  "++logger.wandb.name=$WANDB_ID"
  "++logger.wandb.resume=$WANDB_RESUME"
  norm_stats.norm_mode=minmax
  norm_stats.reduce_all_but_last=true
  norm_stats.sample_frac=1.0
  norm_stats.save_cache_dir=null
  "norm_stats.precomputed_norm_path=$NORM_ARTIFACT"
)

RESUME_CKPT=
if test "$RESTART_COUNT" != 0; then
  shopt -s nullglob
  RESUME_CANDIDATES=("$RUN_DIR"/hpc_ckpt_*.ckpt)
  TERMINAL_CANDIDATES=("$RUN_DIR"/checkpoints/final/*.ckpt)
  shopt -u nullglob
  RESUME_CANDIDATES+=("${TERMINAL_CANDIDATES[@]}")
  LAST_CKPT=$RUN_DIR/checkpoints/last.ckpt
  if test -s "$LAST_CKPT"; then
    RESUME_CANDIDATES+=("$LAST_CKPT")
  fi
  test "${#RESUME_CANDIDATES[@]}" -gt 0
  RESUME_CKPT=$(python - "${RESUME_CANDIDATES[@]}" <<'PY'
import pathlib
import sys

paths = [pathlib.Path(raw) for raw in sys.argv[1:]]
assert paths
for path in paths:
    assert path.is_file() and path.stat().st_size > 0, path
print(max(paths, key=lambda path: (path.stat().st_mtime_ns, str(path))))
PY
)
  test -s "$RESUME_CKPT"
  printf 'Selected newest requeue checkpoint: %s\n' "$RESUME_CKPT"
  COMMON_OVERRIDES+=("ckpt_path=$RESUME_CKPT")
fi

RESOLVED_CONFIG=$PROVENANCE_DIR/resolved_config_attempt_${RESTART_COUNT}.yaml
python -m egomimic.trainHydra "${COMMON_OVERRIDES[@]}" --cfg job --resolve > "$RESOLVED_CONFIG"
python - "$RESOLVED_CONFIG" "$NORM_ARTIFACT" "$RUN_DIR" "$REPO" \
  "$WANDB_ID" "$WANDB_RESUME" "$RESUME_CKPT" "$EXCLUDED_CHAIN_EPISODE" <<'PY'
import pathlib
import sys

from omegaconf import OmegaConf

(
    path,
    norm_artifact,
    run_dir,
    repo,
    wandb_id,
    wandb_resume,
    resume_ckpt,
    excluded_chain_episode,
) = sys.argv[1:]
cfg = OmegaConf.load(path)
model = cfg.model.robomimic_model
domains = {"pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper"}
expected_chain_filter = (
    "lambda row: row.get('episode_hash') != " f"'{excluded_chain_episode}'"
)

assert cfg.seed == 42
assert cfg.mode == "train"
assert str(cfg.paths.root_dir) == run_dir
assert str(cfg.paths.output_dir) == run_dir
assert str(cfg.paths.work_dir) == repo
assert cfg.trainer.precision == "bf16"
assert cfg.trainer.strategy == "ddp"
assert cfg.trainer.devices == 2
assert cfg.trainer.num_nodes == 1
assert cfg.trainer.sync_batchnorm is True
assert cfg.trainer.max_steps == 240_000
assert cfg.trainer.limit_train_batches == 1.0
assert cfg.trainer.val_check_interval == 10_000
assert cfg.trainer.limit_val_batches == 0
assert cfg.trainer.num_sanity_val_steps == 0
assert cfg.trainer.log_every_n_steps == 1
assert cfg.trainer.accumulate_grad_batches == 1
assert cfg.trainer.get("gradient_clip_val") is None
assert cfg.launch_params.gpus_per_node == 2
assert cfg.launch_params.nodes == 1
assert cfg.logger.wandb.offline is False
assert cfg.logger.wandb.project == "pushshapes-flow-transfer"
assert cfg.logger.wandb.entity == "rl2-group"
assert cfg.logger.wandb.group == "flow_transfer_newdata3719_cotrain_h16_20260828"
assert cfg.logger.wandb.id == wandb_id
assert cfg.logger.wandb.name == wandb_id
assert cfg.logger.wandb.resume == wandb_resume
if resume_ckpt:
    assert wandb_resume == "allow"
    assert pathlib.Path(str(cfg.ckpt_path)).resolve() == pathlib.Path(resume_ckpt).resolve()
else:
    assert wandb_resume == "never"
    assert cfg.ckpt_path is None

assert cfg.model.enable_grad_norm is False
assert cfg.model.train_metrics_on_step is True
assert cfg.model.optimizer.lr == 3.0e-5
assert cfg.model.optimizer.weight_decay == 1.0e-4
assert cfg.model.scheduler.max_steps == 240_000
assert cfg.model.scheduler.warmup_steps == 3_000
assert cfg.model.scheduler.warmup_start_factor == 0.1
assert cfg.model.scheduler.eta_min == 3.0e-6
assert cfg.model.get("scheduler_interval", "step") == "step"
assert cfg.model.get("scheduler_frequency", 1) == 1
assert cfg.norm_stats.norm_mode == "minmax"
assert cfg.norm_stats.reduce_all_but_last is True
assert cfg.norm_stats.sample_frac == 1.0
assert cfg.norm_stats.save_cache_dir is None
assert str(cfg.norm_stats.precomputed_norm_path) == norm_artifact

recovery = cfg.callbacks.model_checkpoint
assert recovery._target_ == "lightning.pytorch.callbacks.ModelCheckpoint"
assert str(recovery.dirpath).endswith("/checkpoints")
assert recovery.get("monitor") is None
assert recovery.every_n_epochs is None
assert recovery.every_n_train_steps is None
assert recovery.train_time_interval._target_ == "datetime.timedelta"
assert recovery.train_time_interval.hours == 1
assert recovery.save_top_k == 1
assert recovery.save_last is True
terminal = cfg.callbacks.terminal_checkpoint
assert terminal._target_ == "lightning.pytorch.callbacks.ModelCheckpoint"
assert str(terminal.dirpath).endswith("/checkpoints/final")
assert terminal.filename == "step-{step}"
assert terminal.monitor is None
assert terminal.every_n_train_steps == 240_000
assert terminal.every_n_epochs is None
assert terminal.train_time_interval is None
assert terminal.save_top_k == 1
assert terminal.save_last is False
assert terminal.save_on_train_epoch_end is False
assert terminal.save_on_exception is False
assert terminal.save_weights_only is False
assert terminal.auto_insert_metric_name is False
assert terminal.enable_version_counter is False
assert not any(name.startswith("best_") for name in cfg.callbacks)

assert model.action_horizon == 16
assert set(model.domains) == domains
assert set(cfg.data.train_datasets) == domains
assert set(cfg.data.valid_datasets) == domains
for domain in domains:
    assert cfg.data.train_dataloader_params[domain].batch_size == 32
    assert cfg.data.train_dataloader_params[domain].batch_size * 2 == 64
    assert cfg.data.valid_dataloader_params[domain].batch_size == 16
for split in ("train_datasets", "valid_datasets"):
    chain = cfg.data[split].pushshapes_sim_chain_gripper
    assert chain.filters._target_ == "egomimic.rldb.filters.DatasetFilter"
    assert list(chain.filters.filter_lambdas) == [expected_chain_filter]
chain = cfg.data.train_datasets.pushshapes_sim_chain_gripper.resolver
assert list(chain.folder_paths) == [
    "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_3000_v2",
    "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_gen",
]

assert wandb_id == "ft_cotrain_newdata3719_dp_h16_s42_world2_a40x2_20260828"
stage = model.stages[1]
assert stage.action_horizon == 16
assert set(stage.policies) == domains
assert all(
    policy.noise_scheduler.prediction_type == "epsilon"
    for policy in stage.policies.values()
)

assert "arc_length" not in pathlib.Path(path).read_text().lower()
PY

"$SRUN" --kill-on-bad-exit=1 --unbuffered \
  python -m egomimic.trainHydra "${COMMON_OVERRIDES[@]}" \
  2>&1 | tee -a "$TRAIN_LOG"

FULL_RESULT_TMP=$RUN_DIR/FULL_RESULT.json.tmp
test ! -e "$RUN_DIR/FULL_RESULT.json"
python - "$RUN_DIR" "$RESOLVED_CONFIG" "$EXPECTED_MAX_STEPS" "$WANDB_ID" \
  "$EXPECTED_HEAD" "$EXPECTED_LAUNCHER_SHA" "$EXPECTED_NORM_SHA" \
  "$EXPECTED_DP_SMOKE_RESULT_SHA" \
  "$FULL_RESULT_TMP" <<'PY'
import hashlib
import json
import math
import pathlib
import sys

import torch
from omegaconf import OmegaConf

from egomimic.scripts.verify_training_smoke import read_successful_wandb_exit_code

(
    run_dir_arg,
    config_arg,
    expected_step_arg,
    wandb_id,
    expected_head,
    launcher_sha,
    norm_sha,
    dp_smoke_sha,
    output_arg,
) = sys.argv[1:]
run_dir = pathlib.Path(run_dir_arg)
config_path = pathlib.Path(config_arg)
expected_step = int(expected_step_arg)
terminal_dir = run_dir / "checkpoints" / "final"
expected_checkpoint = terminal_dir / f"step-{expected_step}.ckpt"
assert sorted(terminal_dir.glob("*.ckpt")) == [expected_checkpoint]
checkpoint = torch.load(expected_checkpoint, map_location="cpu", weights_only=False)
assert int(checkpoint["global_step"]) == expected_step
assert checkpoint.get("state_dict")
assert checkpoint.get("optimizer_states")
assert len(checkpoint.get("lr_schedulers", [])) == 1
scheduler_state = checkpoint["lr_schedulers"][0]
assert int(scheduler_state["last_epoch"]) == expected_step
optimizer_lrs = [
    float(group["lr"])
    for state in checkpoint["optimizer_states"]
    for group in state["param_groups"]
]
assert optimizer_lrs and all(math.isfinite(value) for value in optimizer_lrs)
assert all(math.isclose(value, 3.0e-6, rel_tol=1.0e-4) for value in optimizer_lrs)
norm_state = checkpoint["hyper_parameters"]["norm_stats_state"]
assert norm_state["norm_mode"] == "minmax"
assert norm_state["reduce_all_but_last"] is True
assert tuple(norm_state["shapes"][19]["actions"]) == (16, 4)
assert tuple(norm_state["shapes"][20]["actions"]) == (16, 6)
cfg = OmegaConf.load(config_path)
assert cfg.trainer.limit_val_batches == 0
assert cfg.data.train_dataloader_params.pushshapes_sim_u_socket.batch_size * 2 == 64
assert cfg.data.train_dataloader_params.pushshapes_sim_chain_gripper.batch_size * 2 == 64
wandb_streams = sorted(run_dir.glob(f"wandb/run-*/run-{wandb_id}.wandb"))
assert wandb_streams and all(path.stat().st_size > 0 for path in wandb_streams)
final_wandb_stream = max(wandb_streams, key=lambda path: path.stat().st_mtime_ns)
wandb_exit_code = read_successful_wandb_exit_code(final_wandb_stream)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


record = {
    "status": "PASS",
    "arm": "dp",
    "hardware": "2xa40",
    "repo_head": expected_head,
    "launcher_sha256": launcher_sha,
    "norm_sha256": norm_sha,
    "dp_smoke_result_sha256": dp_smoke_sha,
    "world_size": 2,
    "global_batch_per_domain": {
        "pushshapes_sim_u_socket": 64,
        "pushshapes_sim_chain_gripper": 64,
    },
    "validation_enabled": False,
    "wandb_id": wandb_id,
    "global_step": expected_step,
    "wandb_exit_code": wandb_exit_code,
    "final_wandb_stream": str(final_wandb_stream),
    "final_wandb_stream_sha256": digest(final_wandb_stream),
    "terminal_checkpoint": str(expected_checkpoint),
    "terminal_checkpoint_sha256": digest(expected_checkpoint),
    "scheduler_last_epoch": int(scheduler_state["last_epoch"]),
    "optimizer_lrs": optimizer_lrs,
    "resolved_config": str(config_path),
    "resolved_config_sha256": digest(config_path),
    "wandb_streams": [str(path) for path in wandb_streams],
    "wandb_stream_sha256": [digest(path) for path in wandb_streams],
}
pathlib.Path(output_arg).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, sort_keys=True))
PY

# A full PASS is durable only if source, data, norm, and the semantic DP A40
# smoke gate are still byte-identical after training.
validate_static_inputs
verify_smoke_gate > "$PROVENANCE_DIR/smoke_gate_attempt_${RESTART_COUNT}_after.json"
cmp -s \
  "$PROVENANCE_DIR/smoke_gate_attempt_${RESTART_COUNT}_before.json" \
  "$PROVENANCE_DIR/smoke_gate_attempt_${RESTART_COUNT}_after.json"
mv "$FULL_RESULT_TMP" "$RUN_DIR/FULL_RESULT.json"

printf '[full] ALL_PASS arm=dp hardware=a40 head=%s world_size=%s wandb_id=%s run=%s\n' \
  "$EXPECTED_HEAD" "$EXPECTED_WORLD_SIZE" "$WANDB_ID" "$RUN_DIR"
