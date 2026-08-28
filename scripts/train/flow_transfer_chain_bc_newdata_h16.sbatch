#!/bin/bash
# Fail-closed ChainGripper-only BC launcher for the frozen base-plus-generated
# H16 corpus. The sbatch caller must request exactly one A40 on hoffman-lab.
# Full training is accepted only with an exact, immutable semantic smoke result.

set -Eeuo pipefail

ARM=${ARM:?set ARM=latent_h16 or dp}
MODE=${MODE:?set MODE=smoke or full}
EXPECTED_HEAD=${EXPECTED_HEAD:?set the exact clean 40-character commit SHA}
EXPECTED_LAUNCHER_SHA=${EXPECTED_LAUNCHER_SHA:?set the exact launcher SHA256}

CHAIN_BASE_INVENTORY=${CHAIN_BASE_INVENTORY:?set the immutable base inventory path}
EXPECTED_CHAIN_BASE_INVENTORY_SHA=${EXPECTED_CHAIN_BASE_INVENTORY_SHA:?set its SHA256}
CHAIN_BASE_EPISODE_METADATA=${CHAIN_BASE_EPISODE_METADATA:?set the immutable base metadata TSV}
EXPECTED_CHAIN_BASE_EPISODE_METADATA_SHA=${EXPECTED_CHAIN_BASE_EPISODE_METADATA_SHA:?set its SHA256}
CHAIN_GEN_INVENTORY=${CHAIN_GEN_INVENTORY:?set the immutable generated inventory path}
EXPECTED_CHAIN_GEN_INVENTORY_SHA=${EXPECTED_CHAIN_GEN_INVENTORY_SHA:?set its SHA256}
CHAIN_GEN_EPISODE_METADATA=${CHAIN_GEN_EPISODE_METADATA:?set the immutable generated metadata TSV}
EXPECTED_CHAIN_GEN_EPISODE_METADATA_SHA=${EXPECTED_CHAIN_GEN_EPISODE_METADATA_SHA:?set its SHA256}

NORM_ARTIFACT=${NORM_ARTIFACT:?set the exact shared H16 norm_stats.json path}
EXPECTED_NORM_SHA=${EXPECTED_NORM_SHA:?set the exact norm artifact SHA256}
NORM_VALIDATION=${NORM_VALIDATION:?set the exact shared H16 validation.json path}
EXPECTED_NORM_VALIDATION_SHA=${EXPECTED_NORM_VALIDATION_SHA:?set its SHA256}

REPO=/coc/flash7/paphiwetsa3/worktrees/flow-transfer-chain-bc-newdata-h16-20260828
PY_ENV=/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv
EXPECTED_ENV_SHA=8fd1504c955756adf8167f7bd34fc1a09cb844d268898b3ac30693e99ac60e87
EXP_ROOT=/coc/flash7/paphiwetsa3/experiments/flow_transfer_chain_bc_newdata_h16_20260828
LAUNCHER=$REPO/scripts/train/flow_transfer_chain_bc_newdata_h16.sbatch
VERIFIER=$REPO/egomimic/scripts/verify_training_smoke.py
CHAIN_BASE_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_3000_v2
CHAIN_GEN_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_gen
EXCLUDED_EPISODE=episode_T_chain_gripper_obs7_000050
EXCLUDED_FOLDER=${EXCLUDED_EPISODE}.zarr
EXPECTED_CHAIN_BASE_COUNT=3000
EXPECTED_CHAIN_BASE_FRAMES=931061
EXPECTED_CHAIN_GEN_COUNT=720
EXPECTED_CHAIN_GEN_SOURCE_FRAMES=309709
EXPECTED_CHAIN_GEN_EFFECTIVE_COUNT=719
EXPECTED_CHAIN_GEN_EFFECTIVE_FRAMES=306591
EXPECTED_EXCLUDED_FRAMES=3118
EXPECTED_CHAIN_EFFECTIVE_COUNT=3719
EXPECTED_CHAIN_EFFECTIVE_FRAMES=1237652
EXPECTED_WORLD_SIZE=1
EXPECTED_LOCAL_BATCH=64
WANDB_PROJECT=pushshapes-flow-transfer
WANDB_GROUP=flow_transfer_chain_newdata3719_bc_h16_20260828
SLURM_BIN=/opt/slurm/Ubuntu-20.04/24.11.0/bin
export PATH=$SLURM_BIN:$PATH
SRUN=$SLURM_BIN/srun

case "$ARM" in
  latent_h16)
    EXPERIMENT=pusht/pipeline_sampler_chain_gripper_newdata_points_dense_medium_h16
    CONFIG_NAME=flow_transfer_bc_chain_newdata_latent_dense_medium_h16
    FULL_WANDB_ID=ft_bc_chain_newdata3719_latent_dense_m96_h16_s42_20260828
    ;;
  dp)
    EXPERIMENT=pusht/pipeline_diffusion_chain_gripper_newdata_points_h16
    CONFIG_NAME=flow_transfer_bc_chain_newdata_dp_h16
    FULL_WANDB_ID=ft_bc_chain_newdata3719_dp_h16_s42_20260828
    ;;
  *)
    printf 'Unknown ARM=%s; expected latent_h16 or dp\n' "$ARM" >&2
    exit 64
    ;;
esac

case "$MODE" in
  smoke)
    test "${SLURM_RESTART_COUNT:-0}" = 0
    WANDB_ID=ft_bc_chain_newdata3719_${ARM}_smoke_job_${SLURM_JOB_ID:?}
    WANDB_RESUME=never
    RUN_DIR=$EXP_ROOT/smokes/$ARM/job_${SLURM_JOB_ID}
    ;;
  full)
    SMOKE_RESULT=${SMOKE_RESULT:?full mode requires the exact semantic smoke result}
    EXPECTED_SMOKE_RESULT_SHA=${EXPECTED_SMOKE_RESULT_SHA:?full mode requires its SHA256}
    WANDB_ID=$FULL_WANDB_ID
    if test "${SLURM_RESTART_COUNT:-0}" = 0; then
      WANDB_RESUME=never
    else
      WANDB_RESUME=allow
    fi
    RUN_DIR=$EXP_ROOT/runs/$ARM
    ;;
  *)
    printf 'Unknown MODE=%s; expected smoke or full\n' "$MODE" >&2
    exit 64
    ;;
esac

is_sha256() {
  [[ "$1" =~ ^[0-9a-f]{64}$ ]]
}

is_git_sha() {
  [[ "$1" =~ ^[0-9a-f]{40}$ ]]
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
  local metadata=$5 metadata_sha=$6 expected_count=$7 expected_frames=$8

  test -d "$root"
  test -s "$inventory"
  test -s "$metadata"
  test "$(stat -c '%a' "$inventory")" = 444
  test "$(stat -c '%a' "$metadata")" = 444
  test "$(sha256sum "$inventory" | awk '{print $1}')" = "$inventory_sha"
  test "$(sha256sum "$metadata" | awk '{print $1}')" = "$metadata_sha"

  "$PY_ENV/bin/python" - \
    "$label" "$root" "$inventory" "$metadata" \
    "$expected_count" "$expected_frames" <<'PY'
import hashlib
import json
import pathlib
import sys

label, root_arg, inventory_arg, metadata_arg, count_arg, frames_arg = sys.argv[1:]
root = pathlib.Path(root_arg)
inventory = pathlib.Path(inventory_arg)
metadata = pathlib.Path(metadata_arg)
expected_count = int(count_arg)
expected_frames = int(frames_arg)

episodes = sorted(
    (path for path in root.iterdir() if path.is_dir() and path.name.endswith(".zarr")),
    key=lambda path: path.name,
)
names = [path.name for path in episodes]
assert len(names) == expected_count, (label, len(names), expected_count)
assert len(names) == len(set(names)), (label, "duplicate episode names")
expected_inventory = "".join(f"{name}\n" for name in names).encode()
assert inventory.read_bytes() == expected_inventory, (label, "inventory drift")

rows = []
total_frames = 0
for episode in episodes:
    raw = (episode / "zarr.json").read_bytes()
    payload = json.loads(raw)
    frames = payload["attributes"]["total_frames"]
    assert isinstance(frames, int) and not isinstance(frames, bool) and frames > 0
    total_frames += frames
    rows.append(f"{episode.name}\t{frames}\t{hashlib.sha256(raw).hexdigest()}\n")
assert total_frames == expected_frames, (label, total_frames, expected_frames)
assert metadata.read_text() == "".join(rows), (label, "metadata drift")
PY
}

validate_chain_contract() {
  validate_inventory \
    chain_base "$CHAIN_BASE_DATA" \
    "$CHAIN_BASE_INVENTORY" "$EXPECTED_CHAIN_BASE_INVENTORY_SHA" \
    "$CHAIN_BASE_EPISODE_METADATA" \
    "$EXPECTED_CHAIN_BASE_EPISODE_METADATA_SHA" \
    "$EXPECTED_CHAIN_BASE_COUNT" "$EXPECTED_CHAIN_BASE_FRAMES"
  validate_inventory \
    chain_gen "$CHAIN_GEN_DATA" \
    "$CHAIN_GEN_INVENTORY" "$EXPECTED_CHAIN_GEN_INVENTORY_SHA" \
    "$CHAIN_GEN_EPISODE_METADATA" \
    "$EXPECTED_CHAIN_GEN_EPISODE_METADATA_SHA" \
    "$EXPECTED_CHAIN_GEN_COUNT" "$EXPECTED_CHAIN_GEN_SOURCE_FRAMES"

  "$PY_ENV/bin/python" - \
    "$CHAIN_BASE_EPISODE_METADATA" "$CHAIN_GEN_EPISODE_METADATA" \
    "$EXCLUDED_FOLDER" "$EXPECTED_EXCLUDED_FRAMES" \
    "$EXPECTED_CHAIN_GEN_EFFECTIVE_COUNT" \
    "$EXPECTED_CHAIN_GEN_EFFECTIVE_FRAMES" \
    "$EXPECTED_CHAIN_EFFECTIVE_COUNT" \
    "$EXPECTED_CHAIN_EFFECTIVE_FRAMES" <<'PY'
import pathlib
import sys

(
    base_arg,
    gen_arg,
    excluded,
    excluded_frames_arg,
    gen_effective_count_arg,
    gen_effective_frames_arg,
    combined_count_arg,
    combined_frames_arg,
) = sys.argv[1:]


def load(path_arg):
    output = {}
    for row in pathlib.Path(path_arg).read_text().splitlines():
        name, frames, metadata_sha = row.split("\t")
        assert len(metadata_sha) == 64
        int(metadata_sha, 16)
        output[name] = int(frames)
    return output


base = load(base_arg)
gen = load(gen_arg)
assert excluded not in base
assert gen.get(excluded) == int(excluded_frames_arg)
effective_gen = {name: frames for name, frames in gen.items() if name != excluded}
assert len(effective_gen) == int(gen_effective_count_arg)
assert sum(effective_gen.values()) == int(gen_effective_frames_arg)
assert len(base) + len(effective_gen) == int(combined_count_arg)
assert sum(base.values()) + sum(effective_gen.values()) == int(combined_frames_arg)
PY
}

validate_norm_contract() {
  test -s "$NORM_ARTIFACT"
  test -s "$NORM_VALIDATION"
  test "$(stat -c '%a' "$NORM_ARTIFACT")" = 444
  test "$(stat -c '%a' "$NORM_VALIDATION")" = 444
  test "$(sha256sum "$NORM_ARTIFACT" | awk '{print $1}')" = "$EXPECTED_NORM_SHA"
  test "$(sha256sum "$NORM_VALIDATION" | awk '{print $1}')" = "$EXPECTED_NORM_VALIDATION_SHA"

  "$PY_ENV/bin/python" - \
    "$NORM_ARTIFACT" "$NORM_VALIDATION" "$EXPECTED_NORM_SHA" \
    "$EXPECTED_CHAIN_BASE_FRAMES" "$EXPECTED_CHAIN_GEN_SOURCE_FRAMES" \
    "$EXPECTED_CHAIN_GEN_EFFECTIVE_FRAMES" "$EXPECTED_CHAIN_EFFECTIVE_COUNT" \
    "$EXPECTED_CHAIN_EFFECTIVE_FRAMES" "$EXCLUDED_EPISODE" <<'PY'
import hashlib
import json
import math
import pathlib
import sys

(
    artifact_arg,
    validation_arg,
    expected_sha,
    base_frames_arg,
    gen_source_frames_arg,
    gen_effective_frames_arg,
    effective_count_arg,
    effective_frames_arg,
    excluded_episode,
) = sys.argv[1:]
artifact_path = pathlib.Path(artifact_arg).resolve()
raw = artifact_path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == expected_sha
artifact = json.loads(raw)
validation = json.loads(pathlib.Path(validation_arg).read_text())

effective_frames = int(effective_frames_arg)
assert validation["status"] == "PASS"
assert set(validation["shared_by"]) == {"dp_h16", "latent_dense_h16"}
assert validation["action_horizon"] == 16
assert pathlib.Path(validation["artifact"]).resolve() == artifact_path
assert validation["sha256"] == expected_sha
assert validation["chain_effective_train_episodes"] == int(effective_count_arg)
assert validation["chain_gen_source_frames"] == int(gen_source_frames_arg)
assert validation["chain_gen_effective_train_episodes"] == 719
assert validation["excluded_chain_gen_episode"] == excluded_episode
assert validation["per_source_train_frames"]["chain_base"] == int(base_frames_arg)
assert validation["per_source_train_frames"]["chain_gen"] == int(gen_effective_frames_arg)
assert validation["per_embodiment_train_frames"]["20"] == effective_frames
assert validation["sample_frac"] == 1.0

assert artifact["norm_mode"] == "minmax"
assert artifact["reduce_all_but_last"] is True
assert "20" in artifact["stats"]
metadata = artifact["norm_run_metadata"]["embodiments"]["20"]
assert metadata["dataset_size"] == effective_frames
assert metadata["sampled_frames"] == effective_frames
assert metadata["sample_frac"] == 1.0
for key, width in {"state_agent_obj": 6, "actions": 6}.items():
    stats = artifact["stats"]["20"][key]
    for values in stats.values():
        assert len(values) == width
        assert all(math.isfinite(float(value)) for value in values)
PY
}

validate_semantic_smoke() {
  test -s "$SMOKE_RESULT"
  test "$(stat -c '%a' "$SMOKE_RESULT")" = 444
  test "$(sha256sum "$SMOKE_RESULT" | awk '{print $1}')" = "$EXPECTED_SMOKE_RESULT_SHA"

  "$PY_ENV/bin/python" - \
    "$SMOKE_RESULT" "$EXP_ROOT" "$ARM" "$EXPERIMENT" "$CONFIG_NAME" \
    "$EXPECTED_HEAD" "$EXPECTED_LAUNCHER_SHA" "$EXPECTED_NORM_SHA" \
    "$EXPECTED_NORM_VALIDATION_SHA" "$FULL_WANDB_ID" \
    "$EXPECTED_CHAIN_BASE_INVENTORY_SHA" \
    "$EXPECTED_CHAIN_BASE_EPISODE_METADATA_SHA" \
    "$EXPECTED_CHAIN_GEN_INVENTORY_SHA" \
    "$EXPECTED_CHAIN_GEN_EPISODE_METADATA_SHA" <<'PY'
import hashlib
import json
import math
import pathlib
import sys

(
    result_arg,
    root_arg,
    arm,
    experiment,
    config_name,
    expected_head,
    launcher_sha,
    norm_sha,
    norm_validation_sha,
    full_wandb_id,
    base_inventory_sha,
    base_metadata_sha,
    gen_inventory_sha,
    gen_metadata_sha,
) = sys.argv[1:]
result_path = pathlib.Path(result_arg).resolve()
expected_root = (pathlib.Path(root_arg) / "smokes" / arm).resolve()
result_path.relative_to(expected_root)
assert result_path.name == "SMOKE_RESULT.json"
result = json.loads(result_path.read_text())

assert result["status"] == "passed"
assert result["launcher_status"] == "PASS"
assert result["arm"] == arm
assert result["experiment"] == experiment
assert result["config_name"] == config_name
assert result["repo_head"] == expected_head
assert result["launcher_sha256"] == launcher_sha
assert result["norm_sha256"] == norm_sha
assert result["norm_validation_sha256"] == norm_validation_sha
assert result["full_wandb_id"] == full_wandb_id
assert result["world_size"] == 1
assert result["gpu_model"] == "A40"
assert result["global_batch"] == 64
assert result["precision"] == "bf16"
assert result["global_step"] == 2
assert result["scheduler_last_epoch"] == 2
assert result["required_embodiments"] == [20]
assert result["dense_training_steps"] == [0, 1]
assert result["validation_trainer_global_step"] >= 1
assert result["wandb_exit_code"] == 0
assert result["chain_effective_episodes"] == 3719
assert result["chain_effective_frames"] == 1237652
assert result["inventory_sha256"] == {
    "chain_base": base_inventory_sha,
    "chain_base_metadata": base_metadata_sha,
    "chain_gen": gen_inventory_sha,
    "chain_gen_metadata": gen_metadata_sha,
}
assert result["offline"] is True
assert result["wandb_project"] == "pushshapes-flow-transfer"

for key in ("config", "checkpoint", "wandb_stream"):
    path = pathlib.Path(result[key])
    assert path.is_file(), path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == result[f"{key}_sha256"]
assert all(math.isfinite(float(value)) for value in result["validation_metrics"].values())
PY
}

test -x "$PY_ENV/bin/python"
test -x "$SRUN"
test -x "$SLURM_BIN/scontrol"
test -f "$VERIFIER"
is_git_sha "$EXPECTED_HEAD"
for digest in \
  "$EXPECTED_LAUNCHER_SHA" \
  "$EXPECTED_CHAIN_BASE_INVENTORY_SHA" \
  "$EXPECTED_CHAIN_BASE_EPISODE_METADATA_SHA" \
  "$EXPECTED_CHAIN_GEN_INVENTORY_SHA" \
  "$EXPECTED_CHAIN_GEN_EPISODE_METADATA_SHA" \
  "$EXPECTED_NORM_SHA" \
  "$EXPECTED_NORM_VALIDATION_SHA"; do
  is_sha256 "$digest"
done
if test "$MODE" = full; then
  is_sha256 "$EXPECTED_SMOKE_RESULT_SHA"
fi

test "$((EXPECTED_CHAIN_GEN_SOURCE_FRAMES - EXPECTED_CHAIN_GEN_EFFECTIVE_FRAMES))" = "$EXPECTED_EXCLUDED_FRAMES"
test "$((EXPECTED_CHAIN_BASE_FRAMES + EXPECTED_CHAIN_GEN_EFFECTIVE_FRAMES))" = "$EXPECTED_CHAIN_EFFECTIVE_FRAMES"
test "$((EXPECTED_CHAIN_BASE_COUNT + EXPECTED_CHAIN_GEN_EFFECTIVE_COUNT))" = "$EXPECTED_CHAIN_EFFECTIVE_COUNT"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$EXPECTED_LAUNCHER_SHA"
test "$(environment_manifest | sha256sum | awk '{print $1}')" = "$EXPECTED_ENV_SHA"

test "${SLURM_JOB_NUM_NODES:?}" = 1
test "${SLURM_NTASKS:?}" = "$EXPECTED_WORLD_SIZE"
test "${SLURM_CPUS_PER_TASK:?}" -ge 8
test "${SLURM_JOB_PARTITION:?}" = hoffman-lab
test "${SLURM_JOB_ACCOUNT:?}" = hoffman-lab
SLURM_JOB_SUMMARY=$("$SLURM_BIN/scontrol" show job -o "$SLURM_JOB_ID")
if test "$MODE" = full; then
  grep -Eq '(^| )TimeLimit=2-00:00:00( |$)' <<< "$SLURM_JOB_SUMMARY"
fi
test -n "${CUDA_VISIBLE_DEVICES:-}"
IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
test "${#visible_gpus[@]}" = "$EXPECTED_WORLD_SIZE"

validate_chain_contract
validate_norm_contract
if test "$MODE" = full; then
  validate_semantic_smoke
fi

restart_count=${SLURM_RESTART_COUNT:-0}
if test "$MODE" = smoke || test "$restart_count" = 0; then
  test ! -e "$RUN_DIR"
else
  test -d "$RUN_DIR"
fi
mkdir -p "$RUN_DIR"
PROVENANCE_DIR=$RUN_DIR/provenance/restart_${restart_count}
test ! -e "$PROVENANCE_DIR"
mkdir -p "$PROVENANCE_DIR"
TRAIN_LOG=$RUN_DIR/training_restart_${restart_count}.log

nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
  --query-gpu=index,name,uuid,memory.total,driver_version \
  --format=csv,noheader > "$PROVENANCE_DIR/gpu.txt"
test "$(wc -l < "$PROVENANCE_DIR/gpu.txt")" = 1
test "$(grep -ciE '(^|[ ,])A40([ ,]|$)' "$PROVENANCE_DIR/gpu.txt")" = 1

environment_manifest > "$PROVENANCE_DIR/python_environment.txt"
sha256sum "$PROVENANCE_DIR/python_environment.txt" \
  > "$PROVENANCE_DIR/python_environment.sha256"
cp "$LAUNCHER" "$PROVENANCE_DIR/launcher.sbatch"
cp "$VERIFIER" "$PROVENANCE_DIR/verify_training_smoke.py"
cp "$NORM_ARTIFACT" "$PROVENANCE_DIR/norm_stats.json"
cp "$NORM_VALIDATION" "$PROVENANCE_DIR/norm_validation.json"
cp "$CHAIN_BASE_INVENTORY" "$PROVENANCE_DIR/chain_base_inventory.txt"
cp "$CHAIN_BASE_EPISODE_METADATA" "$PROVENANCE_DIR/chain_base_episode_metadata.tsv"
cp "$CHAIN_GEN_INVENTORY" "$PROVENANCE_DIR/chain_gen_inventory.txt"
cp "$CHAIN_GEN_EPISODE_METADATA" "$PROVENANCE_DIR/chain_gen_episode_metadata.tsv"
if test "$MODE" = full; then
  cp "$SMOKE_RESULT" "$PROVENANCE_DIR/semantic_smoke_result.json"
fi
git -C "$REPO" log -1 --format=fuller > "$PROVENANCE_DIR/git_commit.txt"
git -C "$REPO" status --porcelain=v1 --untracked-files=all \
  > "$PROVENANCE_DIR/git_status.txt"
"$SLURM_BIN/scontrol" show job -dd "$SLURM_JOB_ID" \
  > "$PROVENANCE_DIR/slurm_job.txt"
sha256sum "$PROVENANCE_DIR"/* > "$PROVENANCE_DIR/artifacts.sha256"

source "$PY_ENV/bin/activate"
cd "$REPO"
export PYTHONPATH=$REPO
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true

COMMON_OVERRIDES=(
  --config-name=train_zarr_cartesian
  "+experiment=$EXPERIMENT"
  mode=train
  "hydra.run.dir=$RUN_DIR"
  "++paths.root_dir=$RUN_DIR"
  "paths.output_dir=$RUN_DIR"
  "paths.work_dir=$REPO"
  ckpt_path=null
  launch_params.gpus_per_node=1
  launch_params.nodes=1
  trainer.precision=bf16
  trainer.accumulate_grad_batches=1
  trainer.log_every_n_steps=1
  trainer.num_sanity_val_steps=0
  callbacks.model_checkpoint.every_n_epochs=null
  callbacks.model_checkpoint.save_last=true
  callbacks.model_checkpoint.save_top_k=1
  callbacks.model_checkpoint.save_on_train_epoch_end=false
  logger.wandb.project=$WANDB_PROJECT
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

if test "$MODE" = smoke; then
  export WANDB_MODE=offline
  COMMON_OVERRIDES+=(
    trainer.max_steps=2
    trainer.limit_train_batches=2
    trainer.val_check_interval=1
    trainer.limit_val_batches=1
    callbacks.model_checkpoint.every_n_train_steps=2
    callbacks.model_checkpoint.train_time_interval=null
    callbacks.terminal_checkpoint.every_n_train_steps=1
    logger.wandb.offline=true
  )
else
  unset WANDB_MODE
  COMMON_OVERRIDES+=(
    trainer.max_steps=240000
    trainer.limit_train_batches=1.0
    trainer.val_check_interval=10000
    trainer.limit_val_batches=0
    callbacks.model_checkpoint.every_n_train_steps=null
    logger.wandb.offline=false
  )
fi

if test "$MODE" = full && test "$restart_count" != 0; then
  shopt -s nullglob
  resume_candidates=("$RUN_DIR"/hpc_ckpt_*.ckpt)
  shopt -u nullglob
  if test -s "$RUN_DIR/checkpoints/last.ckpt"; then
    resume_candidates+=("$RUN_DIR/checkpoints/last.ckpt")
  fi
  if test "${#resume_candidates[@]}" = 0; then
    printf 'Refusing restart without a recovery checkpoint in %s\n' "$RUN_DIR" >&2
    exit 66
  fi
  RESUME_CKPT=$("$PY_ENV/bin/python" - "${resume_candidates[@]}" <<'PY'
import pathlib
import sys

paths = [pathlib.Path(raw) for raw in sys.argv[1:]]
for path in paths:
    assert path.is_file() and path.stat().st_size > 0, path
print(max(paths, key=lambda path: (path.stat().st_mtime_ns, str(path))))
PY
)
  COMMON_OVERRIDES+=("ckpt_path=$RESUME_CKPT")
fi

RESOLVED_CONFIG=$PROVENANCE_DIR/resolved_config.yaml
python -m egomimic.trainHydra "${COMMON_OVERRIDES[@]}" \
  --cfg job --resolve > "$RESOLVED_CONFIG"

python - \
  "$RESOLVED_CONFIG" "$ARM" "$MODE" "$EXPERIMENT" "$CONFIG_NAME" \
  "$NORM_ARTIFACT" "$RUN_DIR" "$WANDB_ID" "$WANDB_RESUME" \
  "$EXCLUDED_EPISODE" <<'PY'
import pathlib
import sys

from hydra.utils import instantiate
from omegaconf import OmegaConf

(
    path,
    arm,
    mode,
    experiment,
    config_name,
    norm_artifact,
    run_dir,
    wandb_id,
    wandb_resume,
    excluded_episode,
) = sys.argv[1:]
cfg = OmegaConf.load(path)
model = cfg.model.robomimic_model
expected_filter = (
    "lambda row: row.get('episode_hash') != " f"'{excluded_episode}'"
)

assert cfg.name == config_name
assert cfg.seed == 42
assert cfg.mode == "train"
assert str(cfg.paths.root_dir) == run_dir
assert str(cfg.paths.output_dir) == run_dir
assert cfg.trainer.precision == "bf16"
assert cfg.trainer.strategy == "ddp"
assert cfg.trainer.devices == 1
assert cfg.trainer.num_nodes == 1
assert cfg.trainer.accumulate_grad_batches == 1
assert cfg.trainer.log_every_n_steps == 1
assert cfg.trainer.num_sanity_val_steps == 0
assert cfg.trainer.get("gradient_clip_val") is None
assert cfg.launch_params.gpus_per_node == 1
assert cfg.launch_params.nodes == 1
assert cfg.model.enable_grad_norm is False
assert cfg.model.train_metrics_on_step is True
assert cfg.model.train_metrics_on_epoch is True
assert cfg.model.optimizer.lr == 3.0e-5
assert cfg.model.optimizer.weight_decay == 1.0e-4
assert cfg.model.scheduler.max_steps == 240_000
assert cfg.model.scheduler.warmup_steps == 3_000
assert cfg.model.scheduler.warmup_start_factor == 0.1
assert cfg.model.scheduler.eta_min == 3.0e-6
assert cfg.logger.wandb.project == "pushshapes-flow-transfer"
assert cfg.logger.wandb.group == "flow_transfer_chain_newdata3719_bc_h16_20260828"
assert cfg.logger.wandb.id == wandb_id
assert cfg.logger.wandb.resume == wandb_resume
assert cfg.norm_stats.norm_mode == "minmax"
assert cfg.norm_stats.reduce_all_but_last is True
assert cfg.norm_stats.sample_frac == 1.0
assert cfg.norm_stats.save_cache_dir is None
assert str(cfg.norm_stats.precomputed_norm_path) == norm_artifact
assert model.action_horizon == 16
assert list(model.domains) == ["pushshapes_sim_chain_gripper"]
assert set(cfg.data.train_datasets) == {"pushshapes_sim_chain_gripper"}
assert set(cfg.data.valid_datasets) == {"pushshapes_sim_chain_gripper"}
assert cfg.data.train_dataloader_params.pushshapes_sim_chain_gripper.batch_size == 64
assert cfg.data.valid_dataloader_params.pushshapes_sim_chain_gripper.batch_size == 16
assert cfg.evaluator._target_.endswith("HumanRobotOverlayEval")

for split_name, expected_mode, expected_ratio in (
    ("train_datasets", "train", 0.0),
    ("valid_datasets", "valid", 0.02),
):
    dataset = cfg.data[split_name].pushshapes_sim_chain_gripper
    assert dataset.mode == expected_mode
    assert float(dataset.valid_ratio) == expected_ratio
    assert list(dataset.resolver.folder_paths) == [
        "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_3000_v2",
        "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_gen",
    ]
    assert dataset.resolver.key_map.action_horizon == 16
    assert dataset.resolver.key_map.action_zarr_key == "actions"
    assert dataset.resolver.transform_list._target_.endswith(
        "get_chain_gripper_point_transform_list"
    )
    assert float(dataset.resolver.transform_list.world_size) == 512.0
    assert list(dataset.filters.filter_lambdas) == [expected_filter]

checkpoint = cfg.callbacks.model_checkpoint
terminal = cfg.callbacks.terminal_checkpoint
assert checkpoint.save_last is True
assert checkpoint.save_top_k == 1
assert checkpoint.save_on_train_epoch_end is False
assert terminal.save_top_k == 1
assert terminal.save_last is False
assert instantiate(checkpoint).state_key != instantiate(terminal).state_key

if mode == "smoke":
    assert cfg.trainer.max_steps == 2
    assert cfg.trainer.limit_train_batches == 2
    assert cfg.trainer.val_check_interval == 1
    assert cfg.trainer.limit_val_batches == 1
    assert checkpoint.every_n_train_steps == 2
    assert checkpoint.train_time_interval is None
    assert terminal.every_n_train_steps == 1
    assert cfg.logger.wandb.offline is True
else:
    assert mode == "full"
    assert cfg.trainer.max_steps == 240_000
    assert cfg.trainer.limit_train_batches == 1.0
    assert cfg.trainer.val_check_interval == 10_000
    assert cfg.trainer.limit_val_batches == 0
    assert checkpoint.every_n_train_steps is None
    assert checkpoint.train_time_interval._target_ == "datetime.timedelta"
    assert checkpoint.train_time_interval.hours == 1
    assert terminal.every_n_train_steps == cfg.trainer.max_steps
    assert cfg.logger.wandb.offline is False

resolved = pathlib.Path(path).read_text().lower()
assert "arc_length" not in resolved
assert "action_encoder" not in resolved
if arm == "latent_h16":
    assert model.stages[1].action_horizon == 16
    assert model.stages[1].latent_dim == 96
    sampler = model.stages[2]
    assert sampler.action_horizon == 16
    assert sampler.action_dims.pushshapes_sim_chain_gripper == 6
    assert sampler.latent_dim == 96
    assert sampler.decoder_hidden_dim == 512
    assert sampler.denoiser_hidden_dim == 384
    assert sampler.denoising_module.act_seq == 16
    assert sampler.denoising_module.nblocks == 16
else:
    assert arm == "dp"
    stage = model.stages[1]
    assert stage.action_horizon == 16
    policy = stage.policies.pushshapes_sim_chain_gripper
    assert policy.action_horizon == 16
    assert policy.infer_ac_dims.pushshapes_sim_chain_gripper == 6
    assert policy.noise_scheduler.prediction_type == "epsilon"
PY

"$SRUN" --ntasks=1 --gpus-per-task=1 --kill-on-bad-exit=1 --unbuffered \
  python -m egomimic.trainHydra "${COMMON_OVERRIDES[@]}" \
  2>&1 | tee "$TRAIN_LOG"

if grep -E 'Traceback|CUDA out of memory|OutOfMemoryError|ProcessExitedException' \
  "$TRAIN_LOG"; then
  printf 'Fatal signature found in combined training stdout/stderr\n' >&2
  exit 70
fi

validate_chain_contract
validate_norm_contract
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$EXPECTED_LAUNCHER_SHA"

if test "$MODE" = smoke; then
  python "$VERIFIER" "$RUN_DIR" \
    --required-embodiments 20 \
    --expected-head "$EXPECTED_HEAD" \
    --expected-world-size "$EXPECTED_WORLD_SIZE" \
    --dry-run
  python "$VERIFIER" "$RUN_DIR" \
    --required-embodiments 20 \
    --expected-head "$EXPECTED_HEAD" \
    --expected-world-size "$EXPECTED_WORLD_SIZE"

  python - \
    "$RUN_DIR/SMOKE_RESULT.json" "$ARM" "$EXPERIMENT" "$CONFIG_NAME" \
    "$EXPECTED_LAUNCHER_SHA" "$EXPECTED_NORM_SHA" \
    "$EXPECTED_NORM_VALIDATION_SHA" "$FULL_WANDB_ID" \
    "$EXPECTED_CHAIN_BASE_INVENTORY_SHA" \
    "$EXPECTED_CHAIN_BASE_EPISODE_METADATA_SHA" \
    "$EXPECTED_CHAIN_GEN_INVENTORY_SHA" \
    "$EXPECTED_CHAIN_GEN_EPISODE_METADATA_SHA" <<'PY'
import json
import pathlib
import sys

(
    result_arg,
    arm,
    experiment,
    config_name,
    launcher_sha,
    norm_sha,
    norm_validation_sha,
    full_wandb_id,
    base_inventory_sha,
    base_metadata_sha,
    gen_inventory_sha,
    gen_metadata_sha,
) = sys.argv[1:]
path = pathlib.Path(result_arg)
payload = json.loads(path.read_text())
payload.update(
    {
        "launcher_status": "PASS",
        "arm": arm,
        "experiment": experiment,
        "config_name": config_name,
        "launcher_sha256": launcher_sha,
        "norm_sha256": norm_sha,
        "norm_validation_sha256": norm_validation_sha,
        "full_wandb_id": full_wandb_id,
        "offline": True,
        "wandb_project": "pushshapes-flow-transfer",
        "gpu_model": "A40",
        "global_batch": 64,
        "chain_effective_episodes": 3719,
        "chain_effective_frames": 1237652,
        "excluded_episode": "episode_T_chain_gripper_obs7_000050",
        "inventory_sha256": {
            "chain_base": base_inventory_sha,
            "chain_base_metadata": base_metadata_sha,
            "chain_gen": gen_inventory_sha,
            "chain_gen_metadata": gen_metadata_sha,
        },
    }
)
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
  chmod 0444 "$RUN_DIR/SMOKE_RESULT.json"
  SMOKE_SHA=$(sha256sum "$RUN_DIR/SMOKE_RESULT.json" | awk '{print $1}')
  printf '[smoke] ALL_PASS arm=%s result=%s sha256=%s\n' \
    "$ARM" "$RUN_DIR/SMOKE_RESULT.json" "$SMOKE_SHA"
else
  python - "$RUN_DIR" <<'PY'
import pathlib
import sys

import torch

run_dir = pathlib.Path(sys.argv[1])
checkpoints = list((run_dir / "checkpoints" / "final").glob("*.ckpt"))
assert checkpoints, "terminal checkpoint is missing"
matches = []
for path in checkpoints:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("global_step", -1)) == 240_000:
        matches.append(path)
assert len(matches) == 1, matches
print(f"[full] terminal checkpoint: {matches[0]}")
PY
  printf '[full] ALL_PASS arm=%s head=%s run=%s\n' \
    "$ARM" "$EXPECTED_HEAD" "$RUN_DIR"
fi
