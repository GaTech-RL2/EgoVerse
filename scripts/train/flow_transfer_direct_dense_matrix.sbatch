#!/bin/bash
# Resource requests are supplied by sbatch. The default Skynet profile uses
# one A40 for BC and two L40S for obstacle cotrain. The PACE profile is
# cotrain-only and uses one typed A100 or H200 while preserving global batch.
# This file is shared by smoke and full jobs.

set -Eeuo pipefail

ARM=${ARM:?submit with ARM}
MODE=${MODE:?submit with MODE=smoke or MODE=full}
GPUS_EXPECTED=${GPUS_EXPECTED:?submit with GPUS_EXPECTED}
EXPECTED_HEAD=${EXPECTED_HEAD:?submit with EXPECTED_HEAD}
EXPECTED_LAUNCHER_SHA=${EXPECTED_LAUNCHER_SHA:?submit with EXPECTED_LAUNCHER_SHA}
NORM_ARTIFACT=${NORM_ARTIFACT:?submit with an exact norm_stats.json path}
EXPECTED_NORM_SHA=${EXPECTED_NORM_SHA:?submit with the exact norm artifact SHA256}

CLUSTER_PROFILE=${CLUSTER_PROFILE:-skynet_world2}
case "$CLUSTER_PROFILE" in
  skynet_world2)
    REPO=${FLOW_TRANSFER_REPO:-/coc/flash7/paphiwetsa3/worktrees/flow-transfer-direct-dense-obstacle-dp-schedulefix-20260826}
    PY_ENV=${FLOW_TRANSFER_PY_ENV:-/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv}
    EXPECTED_ENV_SHA=${EXPECTED_ENV_SHA:-8fd1504c955756adf8167f7bd34fc1a09cb844d268898b3ac30693e99ac60e87}
    EXP_ROOT=${FLOW_TRANSFER_EXP_ROOT:-/coc/flash7/paphiwetsa3/experiments/flow_transfer_direct_dense_obstacle3k_cotrain_world2_normfix_20260826}
    U_DATA=${FLOW_TRANSFER_U_DATA:-/coc/flash7/paphiwetsa3/datasets/Tsim_v2/u_socket_3000_v2_clean}
    export CHAIN_OBSTACLE_ROOT=${CHAIN_OBSTACLE_ROOT:-/coc/flash7/paphiwetsa3/experiments/chain_gripper_obstacle_mimicgen_128_20260826/output_3000_balanced_v1}
    SLURM_BIN=${SLURM_BIN:-/opt/slurm/Ubuntu-20.04/24.11.0/bin}
    ;;
  pace_world1)
    REPO=${FLOW_TRANSFER_REPO:?PACE profile requires FLOW_TRANSFER_REPO}
    PY_ENV=${FLOW_TRANSFER_PY_ENV:?PACE profile requires FLOW_TRANSFER_PY_ENV}
    EXPECTED_ENV_SHA=${EXPECTED_ENV_SHA:?PACE profile requires EXPECTED_ENV_SHA}
    EXP_ROOT=${FLOW_TRANSFER_EXP_ROOT:?PACE profile requires FLOW_TRANSFER_EXP_ROOT}
    U_DATA=${FLOW_TRANSFER_U_DATA:?PACE profile requires FLOW_TRANSFER_U_DATA}
    export CHAIN_OBSTACLE_ROOT=${CHAIN_OBSTACLE_ROOT:?PACE profile requires CHAIN_OBSTACLE_ROOT}
    SLURM_BIN=${SLURM_BIN:-/opt/slurm/current/bin}
    ;;
  *)
    printf 'Unknown CLUSTER_PROFILE=%s\n' "$CLUSTER_PROFILE" >&2
    exit 64
    ;;
esac
LAUNCHER=$REPO/scripts/train/flow_transfer_direct_dense_matrix.sbatch
OBSTACLE_AUDIT=$CHAIN_OBSTACLE_ROOT/audit_report.json
OBSTACLE_MANIFEST=$CHAIN_OBSTACLE_ROOT/subset_manifest.json
EXPECTED_OBSTACLE_AUDIT_SHA=1f7f341b2929b1f0d4e9178cc832fd0818c82c2d6086c644634bc4b56f4978dc
EXPECTED_OBSTACLE_MANIFEST_SHA=b5c9385b3b100023552601b050a5b1e6c8bf8cf270d2936f8f9373f57484b6b6
EXPECTED_OBSTACLE_INVENTORY_SHA=1c1433966f1da12ce15380daf4cd4a47213fb92c31738546a992cc4457f4cc1d
EXPECTED_U_COUNT=2999
EXPECTED_U_SHA=fb6e107b585d6a5fae9e9ea4186f441b625a680a36d5691469d9c4125cd9a0fa
EXPECTED_U_TRAIN_FRAMES=521676
EXPECTED_CHAIN_TRAIN_FRAMES=1272946
WANDB_GROUP=flow_transfer_direct_dense_obstacle3k_normfix_20260826
export PATH=$SLURM_BIN:$PATH
SRUN=$SLURM_BIN/srun

case "$ARM" in
  bc_usocket_latent)
    EXPERIMENT=pusht/pipeline_sampler_usocket_dense_medium
    BASE_WANDB_ID=ft_bc_usocket_latent_dense_m96_s42_obstacle3k_normfix_20260826
    ACTION_HORIZON=100
    EXPECTED_GPU_MODEL=A40
    EXPECTED_PARTITION=hoffman-lab
    EXPECTED_ACCOUNT=hoffman-lab
    ;;
  bc_usocket_dp)
    EXPERIMENT=pusht/pipeline_diffusion_usocket_h16
    BASE_WANDB_ID=ft_bc_usocket_dp_s42_obstacle3k_normfix_20260826
    ACTION_HORIZON=16
    EXPECTED_GPU_MODEL=A40
    EXPECTED_PARTITION=hoffman-lab
    EXPECTED_ACCOUNT=hoffman-lab
    ;;
  bc_chain_latent)
    EXPERIMENT=pusht/pipeline_sampler_chain_gripper_obstacle_points_dense_medium
    BASE_WANDB_ID=ft_bc_chain_obstacle3k_latent_dense_m96_s42_normfix_20260826
    ACTION_HORIZON=100
    EXPECTED_GPU_MODEL=A40
    EXPECTED_PARTITION=hoffman-lab
    EXPECTED_ACCOUNT=hoffman-lab
    ;;
  bc_chain_dp)
    EXPERIMENT=pusht/pipeline_diffusion_chain_gripper_obstacle_points_h16
    BASE_WANDB_ID=ft_bc_chain_obstacle3k_dp_s42_normfix_20260826
    ACTION_HORIZON=16
    EXPECTED_GPU_MODEL=A40
    EXPECTED_PARTITION=hoffman-lab
    EXPECTED_ACCOUNT=hoffman-lab
    ;;
  cotrain_obstacle_latent)
    EXPERIMENT=pusht/pipeline_sampler_usocket_chain_obstacle_dense_medium
    BASE_WANDB_ID=ft_cotrain_obstacle3k_latent_dense_m96_s42_world2_normfix_20260826
    ACTION_HORIZON=100
    EXPECTED_GPU_MODEL=L40S
    EXPECTED_PARTITION=rl2-lab
    EXPECTED_ACCOUNT=rl2-lab
    ;;
  cotrain_obstacle_dp)
    EXPERIMENT=pusht/pipeline_diffusion_usocket_chain_obstacle_h16
    BASE_WANDB_ID=ft_cotrain_obstacle3k_dp_s42_world2_normfix_20260826
    ACTION_HORIZON=16
    EXPECTED_GPU_MODEL=L40S
    EXPECTED_PARTITION=rl2-lab
    EXPECTED_ACCOUNT=rl2-lab
    ;;
  *)
    printf 'Unknown ARM=%s\n' "$ARM" >&2
    exit 64
    ;;
esac

if test "$CLUSTER_PROFILE" = pace_world1; then
  case "$ARM" in
    cotrain_obstacle_latent)
      BASE_WANDB_ID=ft_cotrain_obstacle3k_latent_dense_m96_s42_world1_pace_normfix_20260826
      ;;
    cotrain_obstacle_dp)
      BASE_WANDB_ID=ft_cotrain_obstacle3k_dp_s42_world1_pace_normfix_20260826
      ;;
    *)
      printf 'PACE profile is cotrain-only; unsupported ARM=%s\n' "$ARM" >&2
      exit 64
      ;;
  esac
  EXPECTED_GPU_MODEL=${PACE_GPU_MODEL:?PACE profile requires PACE_GPU_MODEL=A100 or H200}
  EXPECTED_PARTITION=${PACE_PARTITION:?PACE profile requires PACE_PARTITION=gpu-a100 or gpu-h200}
  EXPECTED_ACCOUNT=gts-dxu345-rl2
  test "$GPUS_EXPECTED" = 1
  case "$EXPECTED_GPU_MODEL:$EXPECTED_PARTITION" in
    A100:gpu-a100 | H200:gpu-h200) ;;
    *)
      printf 'Mismatched PACE GPU/partition: %s / %s\n' \
        "$EXPECTED_GPU_MODEL" "$EXPECTED_PARTITION" >&2
      exit 64
      ;;
  esac
fi

if test "$ACTION_HORIZON" = 100; then
  EXPECTED_NORM_ARM=cotrain_obstacle_latent
else
  test "$ACTION_HORIZON" = 16
  EXPECTED_NORM_ARM=cotrain_obstacle_dp
fi

case "$MODE" in
  smoke)
    WANDB_ID=${BASE_WANDB_ID}_smoke_job_${SLURM_JOB_ID}
    RUN_DIR=$EXP_ROOT/smokes/$ARM/job_${SLURM_JOB_ID}
    EXPECTED_MAX_STEPS=3200
    ;;
  full)
    WANDB_ID=$BASE_WANDB_ID
    RUN_DIR=$EXP_ROOT/runs/${ARM}_seed42
    EXPECTED_MAX_STEPS=240000
    ;;
  *)
    printf 'Unknown MODE=%s\n' "$MODE" >&2
    exit 64
    ;;
esac
PROVENANCE_DIR=$EXP_ROOT/provenance/$MODE/$ARM/job_${SLURM_JOB_ID}
NORM_JOB_DIR=$(dirname "$(dirname "$(dirname "$NORM_ARTIFACT")")")
NORM_VALIDATION=$NORM_JOB_DIR/validation.json

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
test -x "$SRUN"
test "$(environment_manifest | sha256sum | awk '{print $1}')" = "$EXPECTED_ENV_SHA"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$EXPECTED_LAUNCHER_SHA"
test "$(sha256sum "$OBSTACLE_AUDIT" | awk '{print $1}')" = "$EXPECTED_OBSTACLE_AUDIT_SHA"
test "$(sha256sum "$OBSTACLE_MANIFEST" | awk '{print $1}')" = "$EXPECTED_OBSTACLE_MANIFEST_SHA"
test -s "$NORM_ARTIFACT"
test -s "$NORM_VALIDATION"
test "$(stat -c '%a' "$NORM_ARTIFACT")" = 444
test "$(sha256sum "$NORM_ARTIFACT" | awk '{print $1}')" = "$EXPECTED_NORM_SHA"

"$PY_ENV/bin/python" - "$NORM_ARTIFACT" "$NORM_VALIDATION" \
  "$EXPECTED_NORM_SHA" "$EXPECTED_U_TRAIN_FRAMES" \
  "$EXPECTED_CHAIN_TRAIN_FRAMES" \
  "$EXPECTED_NORM_ARM" "$ACTION_HORIZON" <<'PY'
import hashlib
import json
import pathlib
import sys

artifact_path = pathlib.Path(sys.argv[1])
validation_path = pathlib.Path(sys.argv[2])
expected_sha = sys.argv[3]
expected_frames = {"19": int(sys.argv[4]), "20": int(sys.argv[5])}
expected_arm = sys.argv[6]
expected_horizon = int(sys.argv[7])

artifact_bytes = artifact_path.read_bytes()
assert hashlib.sha256(artifact_bytes).hexdigest() == expected_sha
payload = json.loads(artifact_bytes)
validation = json.loads(validation_path.read_text())
assert validation["status"] == "PASS"
assert validation["arm"] == expected_arm
assert validation["action_horizon"] == expected_horizon
# The validated artifact may be materialized on another cluster. Its immutable
# content hash and semantic metadata remain authoritative; retain the original
# absolute path in the copied sidecar as provenance.
assert pathlib.Path(validation["artifact"]).name == artifact_path.name
assert validation["sha256"] == expected_sha
expected_active_sha = validation["active_minmax_sha256"]
assert len(expected_active_sha) == 64
int(expected_active_sha, 16)
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
    assert item["seed"] == 42
    assert item["max_samples"] is None
assert set(payload["stats"]) == {"19", "20"}
assert len(payload["stats"]["19"]["state_agent_obj"]["min"]) == 6
assert len(payload["stats"]["20"]["state_agent_obj"]["min"]) == 6
assert len(payload["stats"]["19"]["actions"]["min"]) == 4
assert len(payload["stats"]["20"]["actions"]["min"]) == 6
active = {
    embodiment: {
        key: {
            name: payload["stats"][embodiment][key][name]
            for name in ("min", "max")
        }
        for key in sorted(payload["stats"][embodiment])
    }
    for embodiment in sorted(payload["stats"])
}
active_bytes = json.dumps(active, sort_keys=True, separators=(",", ":")).encode()
assert hashlib.sha256(active_bytes).hexdigest() == expected_active_sha
PY

mkdir -p "$PROVENANCE_DIR" "$EXP_ROOT/runs" "$EXP_ROOT/smokes" "$(dirname "$RUN_DIR")"
if test "${SLURM_RESTART_COUNT:-0}" = 0; then
  test ! -e "$RUN_DIR"
  mkdir "$RUN_DIR"
  printf '%s\n' "$SLURM_JOB_ID" > "$RUN_DIR/.owner_job_id"
else
  test "$(sed -n '1p' "$RUN_DIR/.owner_job_id")" = "$SLURM_JOB_ID"
fi

inventory() {
  local root=$1 expected_count=$2 expected_sha=$3 output=$4
  # Accept physical Zarr directories and symlinks that resolve to directories.
  # The clean U-Socket corpus is intentionally published as verified symlinks.
  find "$root" -mindepth 1 -maxdepth 1 -xtype d -name '*.zarr' -printf '%f\n' | LC_ALL=C sort > "$output"
  test "$(wc -l < "$output")" = "$expected_count"
  test "$(sha256sum "$output" | awk '{print $1}')" = "$expected_sha"
}
inventory "$U_DATA" "$EXPECTED_U_COUNT" "$EXPECTED_U_SHA" "$PROVENANCE_DIR/usocket_inventory.txt"

: > "$PROVENANCE_DIR/chain_obstacle_inventory.txt"
for level_number in $(seq 1 30); do
  level=$(printf '%02d' "$level_number")
  level_root=$CHAIN_OBSTACLE_ROOT/level_${level}/chain_gripper/T
  count=$(find "$level_root" -mindepth 1 -maxdepth 1 -xtype d -name '*.zarr' | wc -l)
  test "$count" = 100
  find "$level_root" -mindepth 1 -maxdepth 1 -xtype d -name '*.zarr' -printf "level_${level}/%f\n" | LC_ALL=C sort >> "$PROVENANCE_DIR/chain_obstacle_inventory.txt"
done
test "$(wc -l < "$PROVENANCE_DIR/chain_obstacle_inventory.txt")" = 3000
test "$(sha256sum "$PROVENANCE_DIR/chain_obstacle_inventory.txt" | awk '{print $1}')" = "$EXPECTED_OBSTACLE_INVENTORY_SHA"
cmp -s "$CHAIN_OBSTACLE_ROOT/inventory.txt" "$PROVENANCE_DIR/chain_obstacle_inventory.txt"

copy_once_and_verify() {
  local source=$1 destination=$2
  if test -e "$destination"; then
    cmp -s "$source" "$destination"
  else
    cp "$source" "$destination"
  fi
}
copy_once_and_verify "$OBSTACLE_AUDIT" "$PROVENANCE_DIR/obstacle_audit_report.json"
copy_once_and_verify "$OBSTACLE_MANIFEST" "$PROVENANCE_DIR/obstacle_subset_manifest.json"
copy_once_and_verify "$LAUNCHER" "$PROVENANCE_DIR/launcher.sbatch"
copy_once_and_verify "$NORM_ARTIFACT" "$PROVENANCE_DIR/norm_stats.json"
copy_once_and_verify "$NORM_VALIDATION" "$PROVENANCE_DIR/norm_validation.json"
environment_manifest > "$PROVENANCE_DIR/python_environment.txt"
sha256sum "$PROVENANCE_DIR/python_environment.txt" > "$PROVENANCE_DIR/python_environment.sha256"
sha256sum "$PROVENANCE_DIR/norm_stats.json" > "$PROVENANCE_DIR/norm_stats.sha256"
git -C "$REPO" log -1 --format=fuller > "$PROVENANCE_DIR/git_commit.txt"
git -C "$REPO" status --porcelain=v1 --untracked-files=all > "$PROVENANCE_DIR/git_status.txt"
$SLURM_BIN/scontrol show job -dd "$SLURM_JOB_ID" > "$PROVENANCE_DIR/slurm_job.txt"
test "${SLURM_JOB_NUM_NODES:?}" = 1
test "${SLURM_NTASKS:?}" = "$GPUS_EXPECTED"
test "${SLURM_CPUS_PER_TASK:?}" -ge 8
test "${SLURM_JOB_PARTITION:?}" = "$EXPECTED_PARTITION"
test "${SLURM_JOB_ACCOUNT:?}" = "$EXPECTED_ACCOUNT"
test -n "${CUDA_VISIBLE_DEVICES:-}"
IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
test "${#visible_gpus[@]}" = "$GPUS_EXPECTED"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader > "$PROVENANCE_DIR/gpu.txt"
test "$(wc -l < "$PROVENANCE_DIR/gpu.txt")" = "$GPUS_EXPECTED"
test "$(grep -ci "$EXPECTED_GPU_MODEL" "$PROVENANCE_DIR/gpu.txt")" = "$GPUS_EXPECTED"

RUNTIME_ENV=$PROVENANCE_DIR/runtime_env
mkdir -p "$RUNTIME_ENV"
if test -e "$RUNTIME_ENV/emimic"; then
  test "$(readlink -f "$RUNTIME_ENV/emimic")" = "$(readlink -f "$PY_ENV")"
else
  ln -s "$PY_ENV" "$RUNTIME_ENV/emimic"
fi
cd "$RUNTIME_ENV"
source emimic/bin/activate
cd "$REPO"
export PYTHONPATH=$REPO
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true
if test "${SLURM_RESTART_COUNT:-0}" = 0; then
  WANDB_RESUME=never
else
  WANDB_RESUME=allow
fi

COMMON_OVERRIDES=(
  --config-name=train_zarr_cartesian
  "+experiment=$EXPERIMENT"
  mode=train
  "hydra.run.dir=$RUN_DIR"
  "++paths.root_dir=$RUN_DIR"
  "paths.output_dir=$RUN_DIR"
  "paths.work_dir=$REPO"
  "launch_params.gpus_per_node=$GPUS_EXPECTED"
  launch_params.nodes=1
  logger.wandb.project=pushshapes-flow-transfer
  logger.wandb.entity=rl2-group
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

# Keep the cotrain global batch at 64 examples per embodiment on either the
# original two-rank Skynet profile or the one-rank PACE profile.
if [[ "$ARM" == cotrain_* ]]; then
  test $((64 % GPUS_EXPECTED)) = 0
  LOCAL_BATCH_PER_DOMAIN=$((64 / GPUS_EXPECTED))
  COMMON_OVERRIDES+=(
    "data.train_dataloader_params.pushshapes_sim_u_socket.batch_size=$LOCAL_BATCH_PER_DOMAIN"
    "data.train_dataloader_params.pushshapes_sim_chain_gripper.batch_size=$LOCAL_BATCH_PER_DOMAIN"
  )
fi
if [[ "$ARM" == bc_usocket_* || "$ARM" == cotrain_* ]]; then
  COMMON_OVERRIDES+=(
    "data.train_datasets.pushshapes_sim_u_socket.resolver.folder_path=$U_DATA"
    "data.valid_datasets.pushshapes_sim_u_socket.resolver.folder_path=$U_DATA"
  )
fi

if test "$MODE" = smoke; then
  COMMON_OVERRIDES+=(
    trainer.max_steps=3200
    trainer.limit_train_batches=3200
    trainer.val_check_interval=1000
    trainer.limit_val_batches=1
    callbacks.model_checkpoint.every_n_train_steps=1600
    callbacks.model_checkpoint.train_time_interval=null
    callbacks.model_checkpoint.save_on_train_epoch_end=false
    ++logger.csv._target_=lightning.pytorch.loggers.CSVLogger
    "++logger.csv.save_dir=$RUN_DIR"
    ++logger.csv.name=csv
    ++logger.csv.version=metrics
    ++logger.csv.flush_logs_every_n_steps=1
  )
fi

# Select the newest recovery checkpoint by nanosecond mtime. This prefers a
# fresh signal-written HPC checkpoint after normal preemption, but avoids
# rolling back to a stale HPC file when a later hard failure leaves a newer
# hourly last.ckpt.
if test "${SLURM_RESTART_COUNT:-0}" != 0; then
  shopt -s nullglob
  RESUME_CANDIDATES=("$RUN_DIR"/hpc_ckpt_*.ckpt)
  shopt -u nullglob
  LAST_CKPT=$RUN_DIR/checkpoints/last.ckpt
  if test -s "$LAST_CKPT"; then
    RESUME_CANDIDATES+=("$LAST_CKPT")
  fi
  if test "${#RESUME_CANDIDATES[@]}" != 0; then
    RESUME_CKPT=$(python - "${RESUME_CANDIDATES[@]}" <<'PY'
import pathlib
import sys

paths = [pathlib.Path(raw) for raw in sys.argv[1:]]
for path in paths:
    assert path.is_file() and path.stat().st_size > 0, path
newest = max(paths, key=lambda path: (path.stat().st_mtime_ns, str(path)))
print(newest)
PY
)
    test -s "$RESUME_CKPT"
    printf 'Selected requeue checkpoint: %s\n' "$RESUME_CKPT"
    COMMON_OVERRIDES+=("ckpt_path=$RESUME_CKPT")
  fi
fi

RESOLVED_CONFIG=$PROVENANCE_DIR/resolved_config.yaml
python -m egomimic.trainHydra "${COMMON_OVERRIDES[@]}" --cfg job --resolve > "$RESOLVED_CONFIG"
python - "$RESOLVED_CONFIG" "$ARM" "$MODE" "$GPUS_EXPECTED" \
  "$WANDB_ID" "$NORM_ARTIFACT" "$WANDB_RESUME" "$RUN_DIR" "$REPO" <<'PY'
import pathlib
import sys

from omegaconf import OmegaConf

(
    path,
    arm,
    mode,
    expected_gpus,
    wandb_id,
    norm_artifact,
    wandb_resume,
    run_dir,
    repo,
) = sys.argv[1:]
cfg = OmegaConf.load(path)
model = cfg.model.robomimic_model
expected_gpus = int(expected_gpus)

assert cfg.seed == 42
assert cfg.mode == "train"
assert str(cfg.paths.root_dir) == run_dir
assert str(cfg.paths.output_dir) == run_dir
assert str(cfg.paths.work_dir) == repo
assert cfg.trainer.precision == "bf16"
assert cfg.trainer.strategy == "ddp"
assert cfg.trainer.devices == expected_gpus
assert cfg.trainer.num_nodes == 1
assert cfg.trainer.sync_batchnorm is True
assert cfg.trainer.get("gradient_clip_val") is None
assert cfg.trainer.accumulate_grad_batches == 1
assert cfg.launch_params.gpus_per_node == expected_gpus
assert cfg.launch_params.nodes == 1
assert cfg.logger.wandb.project == "pushshapes-flow-transfer"
assert cfg.logger.wandb.id == wandb_id
assert cfg.logger.wandb.resume == wandb_resume
assert cfg.norm_stats.reduce_all_but_last is True
assert cfg.norm_stats.norm_mode == "minmax"
assert cfg.norm_stats.sample_frac == 1.0
assert cfg.norm_stats.save_cache_dir is None
assert str(cfg.norm_stats.precomputed_norm_path) == norm_artifact
assert cfg.model.enable_grad_norm is False
assert cfg.model.optimizer.lr == 1.0e-4
assert cfg.model.optimizer.weight_decay == 1.0e-4
assert cfg.model.scheduler.max_steps == 240_000
assert cfg.model.scheduler.warmup_steps == 3_000
assert cfg.model.scheduler.warmup_start_factor == 0.1
assert cfg.model.scheduler.eta_min == 1.0e-5
assert model._target_ == "egomimic.pipeline.algo.PipelineAlgo"
terminal = cfg.callbacks.terminal_checkpoint
assert terminal._target_ == "lightning.pytorch.callbacks.ModelCheckpoint"
assert str(terminal.dirpath).endswith("/checkpoints/final")
assert terminal.filename == "step-{step}"
assert terminal.monitor is None
assert terminal.every_n_train_steps == cfg.trainer.max_steps
assert terminal.every_n_epochs is None
assert terminal.train_time_interval is None
assert terminal.save_top_k == 1
assert terminal.save_last is False
assert terminal.save_on_train_epoch_end is False
assert terminal.save_on_exception is False
assert terminal.save_weights_only is False
assert terminal.auto_insert_metric_name is False
assert terminal.enable_version_counter is False

if mode == "smoke":
    assert cfg.trainer.max_steps == 3_200
    assert cfg.trainer.limit_train_batches == 3_200
    assert cfg.trainer.val_check_interval == 1_000
    assert cfg.trainer.limit_val_batches == 1
    assert cfg.callbacks.model_checkpoint.every_n_train_steps == 1_600
    assert cfg.callbacks.model_checkpoint.train_time_interval is None
    assert cfg.callbacks.model_checkpoint.save_on_train_epoch_end is False
    assert cfg.logger.csv._target_ == "lightning.pytorch.loggers.CSVLogger"
    assert cfg.logger.csv.name == "csv"
    assert cfg.logger.csv.version == "metrics"
    assert cfg.logger.csv.flush_logs_every_n_steps == 1
else:
    assert mode == "full"
    assert cfg.trainer.max_steps == 240_000
    assert cfg.trainer.limit_train_batches == 1.0
    assert cfg.trainer.val_check_interval == 10_000
    assert cfg.trainer.limit_val_batches == 0

yaml = pathlib.Path(path).read_text()
assert "arc_length" not in yaml.lower()
assert "actions.points" not in yaml

latent = arm.endswith("latent")
cotrain = arm.startswith("cotrain_")
usocket_only = arm.startswith("bc_usocket_")
expected_domains = (
    ["pushshapes_sim_u_socket"]
    if usocket_only
    else ["pushshapes_sim_chain_gripper"]
    if arm.startswith("bc_chain_")
    else ["pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper"]
)
expected_best = {}
if "pushshapes_sim_u_socket" in expected_domains:
    expected_best["best_usocket_checkpoint"] = (
        "best_usocket",
        "Valid/emb19_actions_action_mse",
    )
if "pushshapes_sim_chain_gripper" in expected_domains:
    expected_best["best_chain_checkpoint"] = (
        "best_chain",
        "Valid/emb20_actions_action_mse",
    )
actual_best = {name for name in cfg.callbacks if name.startswith("best_")}
assert actual_best == set(expected_best)
for name, (directory, monitor) in expected_best.items():
    callback = cfg.callbacks[name]
    assert callback._target_ == "lightning.pytorch.callbacks.ModelCheckpoint"
    assert str(callback.dirpath).endswith(f"/checkpoints/{directory}")
    assert callback.filename == "best-step-{step}"
    assert callback.monitor == monitor
    assert callback.mode == "min"
    assert callback.every_n_epochs == 1
    assert callback.every_n_train_steps is None
    assert callback.train_time_interval is None
    assert callback.save_top_k == 1
    assert callback.save_last is False
    assert callback.save_on_train_epoch_end is False
    assert callback.save_on_exception is False
    assert callback.save_weights_only is False
    assert callback.auto_insert_metric_name is False
    assert callback.enable_version_counter is False
assert len(model.domains) == len(expected_domains)
assert set(model.domains) == set(expected_domains)
assert set(cfg.data.train_datasets) == set(expected_domains)
assert set(cfg.data.valid_datasets) == set(expected_domains)
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
for domain in expected_domains:
    assert not cotrain or 64 % expected_gpus == 0
    expected_train_batch = 64 // expected_gpus if cotrain else 64
    assert cfg.data.train_dataloader_params[domain].batch_size == expected_train_batch
    assert cfg.data.valid_dataloader_params[domain].batch_size == 16
global_batch_per_domain = {
    domain: cfg.data.train_dataloader_params[domain].batch_size * expected_gpus
    for domain in expected_domains
}
assert set(global_batch_per_domain.values()) == {64}
assert sum(global_batch_per_domain.values()) == (128 if cotrain else 64)

if latent:
    assert model.action_horizon == 100
    assert model.stages[1]._target_.endswith("GaussianLatentNoise")
    assert model.stages[1].latent_dim == 96
    sampler = model.stages[2]
    assert sampler._target_.endswith("MultiJActionSampler")
    assert sampler.action_horizon == 100
    assert sampler.latent_dim == 96
    assert sampler.decoder_hidden_dim == 512
    assert sampler.denoiser_hidden_dim == 384
    assert sampler.denoising_module.hidden_dim == 384
    assert sampler.denoising_module.act_dim == 96
    assert sampler.denoising_module.act_seq == 100
    assert sampler.denoising_module.nblocks == 16
    assert sampler.gradient_accumulation_steps == 1
    if "pushshapes_sim_u_socket" in expected_domains:
        assert sampler.action_dims.pushshapes_sim_u_socket == 4
    if "pushshapes_sim_chain_gripper" in expected_domains:
        assert sampler.action_dims.pushshapes_sim_chain_gripper == 6
else:
    assert model.action_horizon == 16
    stage = model.stages[1]
    assert stage._target_.endswith("MultiDomainDiffusionPolicyStage")
    for domain, width in (("pushshapes_sim_u_socket", 4), ("pushshapes_sim_chain_gripper", 6)):
        if domain not in expected_domains:
            continue
        policy = stage.policies[domain]
        assert policy.model._target_.endswith("ConditionalUnet1D")
        assert policy.model.input_dim == width
        assert policy.noise_scheduler.prediction_type == "epsilon"

if cotrain or arm.startswith("bc_chain_"):
    chain = cfg.data.train_datasets.pushshapes_sim_chain_gripper.resolver
    assert chain._target_.endswith("LocalEpisodeResolverManyWithEmbodimentOverride")
    assert len(chain.folder_paths) == 30
    assert str(chain.folder_paths[0]).endswith("/level_01/chain_gripper/T")
    assert str(chain.folder_paths[-1]).endswith("level_30/chain_gripper/T")
PY

python - "$OBSTACLE_AUDIT" "$OBSTACLE_MANIFEST" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
manifest = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert report["status"] == "PASS"
assert report["levels"] == list(range(1, 31))
assert report["target_total_per_level"] == 100
assert report["total_episodes"] == 3000
assert report["total_frames"] == 1272946
assert sum(item["total"] for item in report["reports"]) == 3000
assert sum(item["manual"] for item in report["reports"]) == 480
assert sum(item["generated"] for item in report["reports"]) == 2520
assert len(manifest["episodes"]) == 3000
assert manifest["total_frames"] == 1272946
assert manifest["target_per_level"] == 100
PY

TRAIN_LOG=$RUN_DIR/training.log
"$SRUN" --kill-on-bad-exit=1 --unbuffered \
  python -m egomimic.trainHydra "${COMMON_OVERRIDES[@]}" 2>&1 | tee -a "$TRAIN_LOG"

python - "$RUN_DIR" "$PROVENANCE_DIR" "$MODE" "$EXPECTED_MAX_STEPS" \
  "$GPUS_EXPECTED" "$TRAIN_LOG" "$ARM" "$ACTION_HORIZON" "$WANDB_ID" <<'PY'
import csv
import hashlib
import json
import math
import pathlib
import sys

import torch
from omegaconf import OmegaConf

run_dir = pathlib.Path(sys.argv[1])
provenance_dir = pathlib.Path(sys.argv[2])
mode = sys.argv[3]
expected_step = int(sys.argv[4])
expected_world_size = int(sys.argv[5])
train_log = pathlib.Path(sys.argv[6])
arm = sys.argv[7]
action_horizon = int(sys.argv[8])
wandb_id = sys.argv[9]

terminal_dir = run_dir / "checkpoints" / "final"
expected_checkpoint = terminal_dir / f"step-{expected_step}.ckpt"
checkpoint_files = sorted(terminal_dir.glob("*.ckpt"))
assert checkpoint_files == [expected_checkpoint], checkpoint_files
checkpoint = torch.load(expected_checkpoint, map_location="cpu", weights_only=False)
assert int(checkpoint["global_step"]) == expected_step
assert checkpoint.get("state_dict")
assert checkpoint.get("optimizer_states")
assert checkpoint["hyper_parameters"]["enable_grad_norm"] is False
assert len(checkpoint.get("lr_schedulers", [])) == 1
scheduler_state = checkpoint["lr_schedulers"][0]
assert int(scheduler_state["last_epoch"]) == expected_step
optimizer_lrs = [
    float(group["lr"])
    for state in checkpoint["optimizer_states"]
    for group in state["param_groups"]
]
assert optimizer_lrs and all(math.isfinite(value) for value in optimizer_lrs)
norm_state = checkpoint["hyper_parameters"]["norm_stats_state"]
assert norm_state["norm_mode"] == "minmax"
assert norm_state["reduce_all_but_last"] is True
expected_shapes = {}
if "usocket" in arm or arm.startswith("cotrain_"):
    expected_shapes[19] = (action_horizon, 4)
if "chain" in arm or arm.startswith("cotrain_"):
    expected_shapes[20] = (action_horizon, 6)
assert set(norm_state["shapes"]) == set(expected_shapes)
for embodiment, expected_shape in expected_shapes.items():
    assert tuple(norm_state["shapes"][embodiment]["actions"]) == expected_shape
checkpoint_sha = hashlib.sha256(expected_checkpoint.read_bytes()).hexdigest()

resolved_cfg = OmegaConf.load(provenance_dir / "resolved_config.yaml")
assert all(
    float(dataset.valid_ratio) == 0.0
    for dataset in resolved_cfg.data.train_datasets.values()
)
assert all(
    dataset.mode == "train"
    for dataset in resolved_cfg.data.train_datasets.values()
)
assert all(
    float(dataset.valid_ratio) == 0.02
    for dataset in resolved_cfg.data.valid_datasets.values()
)
assert all(
    dataset.mode == "valid"
    for dataset in resolved_cfg.data.valid_datasets.values()
)
if mode == "smoke":
    assert resolved_cfg.trainer.limit_val_batches == 1
else:
    assert mode == "full"
    assert resolved_cfg.trainer.limit_val_batches == 0
global_batch_per_domain = {
    domain: int(params.batch_size) * expected_world_size
    for domain, params in resolved_cfg.data.train_dataloader_params.items()
}
assert set(global_batch_per_domain.values()) == {64}
total_global_batch = sum(global_batch_per_domain.values())
assert total_global_batch == (128 if len(global_batch_per_domain) == 2 else 64)

expected_best = {}
if 19 in expected_shapes:
    expected_best["best_usocket"] = "Valid/emb19_actions_action_mse"
if 20 in expected_shapes:
    expected_best["best_chain"] = "Valid/emb20_actions_action_mse"
best_results = {}
for directory, monitor in expected_best.items():
    files = sorted((run_dir / "checkpoints" / directory).glob("*.ckpt"))
    if mode == "full":
        assert not files, (directory, files)
        continue
    assert mode == "smoke"
    assert len(files) == 1, (directory, files)
    best_path = files[0]
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    best_step = int(best_checkpoint["global_step"])
    assert 0 < best_step <= expected_step
    assert best_step in {1_000, 2_000, 3_000}, best_step
    state_key = (
        "ModelCheckpoint{'monitor': '"
        + monitor
        + "', 'mode': 'min', 'every_n_train_steps': 0, "
        + "'every_n_epochs': 1, 'train_time_interval': None}"
    )
    callback_states = best_checkpoint.get("callbacks", {})
    assert state_key in callback_states, list(callback_states)
    callback_state = callback_states[state_key]
    assert callback_state["monitor"] == monitor
    assert pathlib.Path(callback_state["dirpath"]).resolve() == best_path.parent.resolve()
    assert pathlib.Path(callback_state["best_model_path"]).resolve() == best_path.resolve()
    best_score = float(callback_state["best_model_score"])
    current_score = float(callback_state["current_score"])
    kth_value = float(callback_state["kth_value"])
    assert math.isfinite(best_score)
    assert math.isfinite(current_score)
    assert math.isfinite(kth_value)
    assert math.isclose(current_score, best_score)
    assert math.isclose(kth_value, best_score)
    assert set(callback_state["best_k_models"]) == {str(best_path)}
    assert math.isclose(
        float(callback_state["best_k_models"][str(best_path)]),
        best_score,
    )
    assert pathlib.Path(callback_state["kth_best_model_path"]).resolve() == best_path.resolve()
    assert callback_state["last_model_path"] == ""
    best_results[directory] = {
        "checkpoint": str(best_path),
        "checkpoint_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest(),
        "global_step": best_step,
        "monitor": monitor,
        "best_score": best_score,
        "callback_state_key": state_key,
    }

result = {
    "status": "PASS",
    "mode": mode,
    "expected_world_size": expected_world_size,
    "terminal_checkpoint": str(expected_checkpoint),
    "terminal_checkpoint_sha256": checkpoint_sha,
    "global_step": int(checkpoint["global_step"]),
    "state_tensor_count": len(checkpoint["state_dict"]),
    "optimizer_state_count": len(checkpoint["optimizer_states"]),
    "optimizer_lrs": optimizer_lrs,
    "scheduler_last_epoch": int(scheduler_state["last_epoch"]),
    "global_batch_per_domain": global_batch_per_domain,
    "total_global_batch": total_global_batch,
    "best_checkpoints": best_results,
    "gradient_clipping_enabled": False,
    "training_uses_all_episodes": True,
    "validation_enabled": mode == "smoke",
}

if mode == "smoke":
    assert all(9.9e-5 <= value <= 1.0e-4 for value in optimizer_lrs), optimizer_lrs
    metrics_path = run_dir / "csv" / "metrics" / "metrics.csv"
    assert metrics_path.is_file(), metrics_path
    with metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows

    metric_values = {}
    validation_steps = set()
    for row in rows:
        for key, raw_value in row.items():
            if not raw_value or key in {"epoch", "step"}:
                continue
            value = float(raw_value)
            assert math.isfinite(value), (key, raw_value)
            metric_values.setdefault(key, []).append(value)
            if key.startswith("Valid/") and "mse" in key.lower():
                validation_steps.add(int(float(row["step"])))

    train_losses = {
        key: values
        for key, values in metric_values.items()
        if key.startswith("Train/") and "loss" in key.lower()
    }
    validation_mse = {
        key: values
        for key, values in metric_values.items()
        if key.startswith("Valid/") and "mse" in key.lower()
    }
    grad_norms = {
        key: values
        for key, values in metric_values.items()
        if key.startswith("Train/") and "grad_norm" in key.lower()
    }
    assert train_losses, sorted(metric_values)
    assert validation_mse, sorted(metric_values)
    assert validation_steps and max(validation_steps) >= 2_999, validation_steps
    assert not grad_norms, grad_norms
    for best in best_results.values():
        monitor = best["monitor"]
        assert monitor in validation_mse, (monitor, sorted(validation_mse))
        assert math.isclose(
            best["best_score"],
            min(validation_mse[monitor]),
            rel_tol=1.0e-5,
            abs_tol=1.0e-8,
        ), (best, validation_mse[monitor])

    log_text = train_log.read_text(errors="replace")
    assert f"max_steps={expected_step}" in log_text
    assert "[GRAD_NORM_SPIKE]" not in log_text
    for rank in range(expected_world_size):
        assert f"Rank {rank} on fit start" in log_text

    wandb_files = sorted(run_dir.glob("wandb/run-*/run-*.wandb"))
    assert wandb_files, wandb_files
    assert all(path.name == f"run-{wandb_id}.wandb" for path in wandb_files)
    assert all(path.stat().st_size > 0 for path in wandb_files)
    result.update(
        {
            "metrics_csv": str(metrics_path),
            "train_loss_metrics": train_losses,
            "validation_mse_metrics": validation_mse,
            "validation_steps": sorted(validation_steps),
            "gradient_norm_metrics": grad_norms,
            "gradient_clipping_enabled": False,
            "wandb_files": [str(path) for path in wandb_files],
        }
    )

output = provenance_dir / "run_validation.json"
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
PY
