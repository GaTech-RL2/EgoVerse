#!/bin/bash
# DP-only A40 smoke for the new-data Flow Transfer H16 cotrain run.
# This is the hardware/world-size gate for the matching 2xA40 full launcher.
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a40:2
#SBATCH --open-mode=append

set -Eeuo pipefail

EXPECTED_HEAD=${EXPECTED_HEAD:?set the exact clean 40-character commit SHA}
EXPECTED_LAUNCHER_SHA=${EXPECTED_LAUNCHER_SHA:?set the exact launcher SHA256}
NORM_ARTIFACT=${NORM_ARTIFACT:?set the exact norm_stats.json path}
EXPECTED_NORM_SHA=${EXPECTED_NORM_SHA:?set the exact norm artifact SHA256}

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
EXPECTED_CHAIN_TRAIN_FRAMES=${EXPECTED_CHAIN_TRAIN_FRAMES:?set the combined Chain frame count}

REPO=/coc/flash7/paphiwetsa3/worktrees/flow-transfer-cotrain-newdata-h16-full-20260828
PY_ENV=/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv
EXPECTED_ENV_SHA=8fd1504c955756adf8167f7bd34fc1a09cb844d268898b3ac30693e99ac60e87
EXP_ROOT=/coc/flash7/paphiwetsa3/experiments/flow_transfer_newdata3719_cotrain_h16_20260828
LAUNCHER=$REPO/scripts/train/flow_transfer_newdata_h16_world2_dp_a40_smoke.sbatch
VERIFIER=$REPO/egomimic/scripts/verify_training_smoke.py
U_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/u_socket_3000_v2_clean
CHAIN_BASE_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_3000_v2
CHAIN_GEN_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_gen
EXCLUDED_CHAIN_EPISODE=episode_T_chain_gripper_obs7_000050
EXCLUDED_CHAIN_FOLDER=${EXCLUDED_CHAIN_EPISODE}.zarr
EXCLUDED_CHAIN_FRAMES=3118
EXPECTED_WORLD_SIZE=2
SLURM_BIN=/opt/slurm/Ubuntu-20.04/24.11.0/bin
export PATH=$SLURM_BIN:$PATH
SRUN=$SLURM_BIN/srun
EXPERIMENT=pusht/pipeline_diffusion_usocket_chain_newdata_h16

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
        normalized = name.lower().replace("_", "-")
        entries.add(f"{normalized}=={distribution.version}")
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
    (
        path
        for path in root.iterdir()
        if path.name.endswith(".zarr") and path.is_dir()
    ),
    key=lambda path: path.name,
)
names = [path.name for path in episodes]
assert len(names) == expected_count, (label, len(names), expected_count)
assert len(names) == len(set(names)), f"{label} has duplicate episode names"
inventory_bytes = "".join(f"{name}\n" for name in names).encode()
assert inventory_path.read_bytes() == inventory_bytes, (
    label,
    "live inventory differs from pinned artifact",
)

metadata_rows = []
for episode in episodes:
    zarr_json = episode / "zarr.json"
    raw = zarr_json.read_bytes()
    payload = json.loads(raw)
    frames = payload["attributes"]["total_frames"]
    assert isinstance(frames, int) and not isinstance(frames, bool) and frames > 0, (
        label,
        episode.name,
        frames,
    )
    metadata_rows.append(
        f"{episode.name}\t{frames}\t{hashlib.sha256(raw).hexdigest()}\n"
    )
metadata_bytes = "".join(metadata_rows).encode()
assert metadata_path.read_bytes() == metadata_bytes, (
    label,
    "live episode metadata differs from pinned artifact",
)
PY
}

validate_all_inventories() {
  validate_inventory \
    usocket "$U_DATA" "$U_INVENTORY" \
    "$EXPECTED_U_INVENTORY_SHA" \
    "$U_EPISODE_METADATA" "$EXPECTED_U_EPISODE_METADATA_SHA" \
    "$EXPECTED_U_COUNT"
  validate_inventory \
    chain_base "$CHAIN_BASE_DATA" "$CHAIN_BASE_INVENTORY" \
    "$EXPECTED_CHAIN_BASE_INVENTORY_SHA" \
    "$CHAIN_BASE_EPISODE_METADATA" \
    "$EXPECTED_CHAIN_BASE_EPISODE_METADATA_SHA" \
    "$EXPECTED_CHAIN_BASE_COUNT"
  validate_inventory \
    chain_gen "$CHAIN_GEN_DATA" "$CHAIN_GEN_INVENTORY" \
    "$EXPECTED_CHAIN_GEN_INVENTORY_SHA" \
    "$CHAIN_GEN_EPISODE_METADATA" \
    "$EXPECTED_CHAIN_GEN_EPISODE_METADATA_SHA" \
    "$EXPECTED_CHAIN_GEN_COUNT"
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

test -x "$PY_ENV/bin/python"
test -x "$SRUN"
test -f "$VERIFIER"
test "${#EXPECTED_HEAD}" = 40
is_sha256 "$EXPECTED_LAUNCHER_SHA"
is_sha256 "$EXPECTED_NORM_SHA"
is_sha256 "$EXPECTED_U_INVENTORY_SHA"
is_sha256 "$EXPECTED_U_EPISODE_METADATA_SHA"
is_sha256 "$EXPECTED_CHAIN_BASE_INVENTORY_SHA"
is_sha256 "$EXPECTED_CHAIN_BASE_EPISODE_METADATA_SHA"
is_sha256 "$EXPECTED_CHAIN_GEN_INVENTORY_SHA"
is_sha256 "$EXPECTED_CHAIN_GEN_EPISODE_METADATA_SHA"
is_positive_integer "$EXPECTED_U_COUNT"
is_positive_integer "$EXPECTED_CHAIN_BASE_COUNT"
is_positive_integer "$EXPECTED_CHAIN_GEN_COUNT"
is_positive_integer "$EXPECTED_U_TRAIN_FRAMES"
is_positive_integer "$EXPECTED_CHAIN_TRAIN_FRAMES"
test "$EXPECTED_U_COUNT" = 2999
test "$EXPECTED_CHAIN_BASE_COUNT" = 3000
test "$EXPECTED_CHAIN_GEN_COUNT" = 720
test "$EXPECTED_U_TRAIN_FRAMES" = 521676
test "$EXPECTED_CHAIN_TRAIN_FRAMES" = 1237652
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$EXPECTED_LAUNCHER_SHA"
test "$(environment_manifest | sha256sum | awk '{print $1}')" = "$EXPECTED_ENV_SHA"
test -s "$NORM_ARTIFACT"
test "$(stat -c '%a' "$NORM_ARTIFACT")" = 444
test "$(sha256sum "$NORM_ARTIFACT" | awk '{print $1}')" = "$EXPECTED_NORM_SHA"
validate_all_inventories

"$PY_ENV/bin/python" - \
  "$NORM_ARTIFACT" "$EXPECTED_NORM_SHA" \
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

test "${SLURM_RESTART_COUNT:-0}" = 0
test "${SLURM_JOB_NUM_NODES:?}" = 1
test "${SLURM_NTASKS:?}" = "$EXPECTED_WORLD_SIZE"
test "${SLURM_CPUS_PER_TASK:?}" -ge 8
test "${SLURM_JOB_PARTITION:?}" = hoffman-lab
test "${SLURM_JOB_ACCOUNT:?}" = hoffman-lab
test -n "${CUDA_VISIBLE_DEVICES:-}"
IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
test "${#visible_gpus[@]}" = "$EXPECTED_WORLD_SIZE"

RUN_DIR=$EXP_ROOT/smokes/dp_a40/job_${SLURM_JOB_ID:?}
PROVENANCE_DIR=$EXP_ROOT/provenance/smoke/dp_a40/job_${SLURM_JOB_ID}
TRAIN_LOG=$RUN_DIR/training.log
WANDB_ID=ft_cotrain_newdata3719_dp_h16_s42_world2_a40_smoke_job_${SLURM_JOB_ID}
test ! -e "$RUN_DIR"
test ! -e "$PROVENANCE_DIR"
mkdir -p "$RUN_DIR" "$PROVENANCE_DIR"

nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
  --query-gpu=index,name,uuid,memory.total,driver_version \
  --format=csv,noheader > "$PROVENANCE_DIR/gpu.txt"
test "$(wc -l < "$PROVENANCE_DIR/gpu.txt")" = "$EXPECTED_WORLD_SIZE"
test "$(grep -ci A40 "$PROVENANCE_DIR/gpu.txt")" = "$EXPECTED_WORLD_SIZE"

environment_manifest > "$PROVENANCE_DIR/python_environment.txt"
sha256sum "$PROVENANCE_DIR/python_environment.txt" \
  > "$PROVENANCE_DIR/python_environment.sha256"
cp "$LAUNCHER" "$PROVENANCE_DIR/launcher.sbatch"
cp "$VERIFIER" "$PROVENANCE_DIR/verify_training_smoke.py"
cp "$NORM_ARTIFACT" "$PROVENANCE_DIR/norm_stats.json"
cp "$U_INVENTORY" "$PROVENANCE_DIR/usocket_inventory.txt"
cp "$U_EPISODE_METADATA" "$PROVENANCE_DIR/usocket_episode_metadata.tsv"
cp "$CHAIN_BASE_INVENTORY" "$PROVENANCE_DIR/chain_base_inventory.txt"
cp "$CHAIN_BASE_EPISODE_METADATA" \
  "$PROVENANCE_DIR/chain_base_episode_metadata.tsv"
cp "$CHAIN_GEN_INVENTORY" "$PROVENANCE_DIR/chain_gen_inventory.txt"
cp "$CHAIN_GEN_EPISODE_METADATA" \
  "$PROVENANCE_DIR/chain_gen_episode_metadata.tsv"
git -C "$REPO" log -1 --format=fuller > "$PROVENANCE_DIR/git_commit.txt"
git -C "$REPO" status --porcelain=v1 --untracked-files=all \
  > "$PROVENANCE_DIR/git_status.txt"
"$SLURM_BIN/scontrol" show job -dd "$SLURM_JOB_ID" \
  > "$PROVENANCE_DIR/slurm_job.txt"
"$PY_ENV/bin/python" - "$PROVENANCE_DIR/slurm_job.txt" short 02:00:00 0 <<'PY'
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
sha256sum \
  "$PROVENANCE_DIR/launcher.sbatch" \
  "$PROVENANCE_DIR/verify_training_smoke.py" \
  "$PROVENANCE_DIR/norm_stats.json" \
  "$PROVENANCE_DIR/usocket_inventory.txt" \
  "$PROVENANCE_DIR/usocket_episode_metadata.tsv" \
  "$PROVENANCE_DIR/chain_base_inventory.txt" \
  "$PROVENANCE_DIR/chain_base_episode_metadata.tsv" \
  "$PROVENANCE_DIR/chain_gen_inventory.txt" \
  "$PROVENANCE_DIR/chain_gen_episode_metadata.tsv" \
  > "$PROVENANCE_DIR/artifacts.sha256"

source "$PY_ENV/bin/activate"
cd "$REPO"
export PYTHONPATH=$REPO
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
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
  trainer.max_steps=2
  trainer.limit_train_batches=2
  trainer.val_check_interval=1
  trainer.limit_val_batches=1
  trainer.num_sanity_val_steps=0
  trainer.log_every_n_steps=1
  callbacks.model_checkpoint.every_n_train_steps=2
  callbacks.model_checkpoint.train_time_interval=null
  callbacks.model_checkpoint.save_on_train_epoch_end=false
  callbacks.terminal_checkpoint.every_n_train_steps=1
  logger.wandb.offline=true
  logger.wandb.project=pushshapes-flow-transfer
  logger.wandb.entity=rl2-group
  logger.wandb.group=flow_transfer_newdata3719_cotrain_h16_20260828
  "logger.wandb.id=$WANDB_ID"
  "++logger.wandb.name=$WANDB_ID"
  ++logger.wandb.resume=never
  norm_stats.norm_mode=minmax
  norm_stats.reduce_all_but_last=true
  norm_stats.sample_frac=1.0
  norm_stats.save_cache_dir=null
  "norm_stats.precomputed_norm_path=$NORM_ARTIFACT"
)

RESOLVED_CONFIG=$PROVENANCE_DIR/resolved_config.yaml
python -m egomimic.trainHydra "${COMMON_OVERRIDES[@]}" \
  --cfg job --resolve > "$RESOLVED_CONFIG"
python - \
  "$RESOLVED_CONFIG" "$NORM_ARTIFACT" "$RUN_DIR" \
  "$EXCLUDED_CHAIN_EPISODE" "$WANDB_ID" <<'PY'
import pathlib
import sys

from hydra.utils import instantiate
from omegaconf import OmegaConf

path, norm_artifact, run_dir, excluded_chain_episode, wandb_id = sys.argv[1:]
cfg = OmegaConf.load(path)
model = cfg.model.robomimic_model
expected_chain_filter = (
    "lambda row: row.get('episode_hash') != "
    f"'{excluded_chain_episode}'"
)

assert cfg.seed == 42
assert cfg.mode == "train"
assert str(cfg.paths.root_dir) == run_dir
assert str(cfg.paths.output_dir) == run_dir
assert cfg.trainer.precision == "bf16"
assert cfg.trainer.strategy == "ddp"
assert cfg.trainer.devices == 2
assert cfg.trainer.num_nodes == 1
assert cfg.trainer.sync_batchnorm is True
assert cfg.trainer.max_steps == 2
assert cfg.trainer.limit_train_batches == 2
assert cfg.trainer.val_check_interval == 1
assert cfg.trainer.limit_val_batches == 1
assert cfg.trainer.num_sanity_val_steps == 0
assert cfg.trainer.log_every_n_steps == 1
assert cfg.trainer.accumulate_grad_batches == 1
assert cfg.trainer.get("gradient_clip_val") is None
assert cfg.launch_params.gpus_per_node == 2
assert cfg.launch_params.nodes == 1
assert cfg.logger.wandb.offline is True
assert cfg.logger.wandb.project == "pushshapes-flow-transfer"
assert cfg.logger.wandb.entity == "rl2-group"
assert cfg.logger.wandb.group == "flow_transfer_newdata3719_cotrain_h16_20260828"
assert cfg.logger.wandb.id == wandb_id
assert cfg.logger.wandb.name == wandb_id
assert cfg.logger.wandb.resume == "never"
assert cfg.model.enable_grad_norm is False
assert cfg.model.train_metrics_on_step is True
assert cfg.model.optimizer.lr == 3.0e-5
assert cfg.model.scheduler.max_steps == 240_000
assert cfg.model.scheduler.warmup_steps == 3_000
assert cfg.model.scheduler.warmup_start_factor == 0.1
assert cfg.model.scheduler.eta_min == 3.0e-6
assert cfg.norm_stats.norm_mode == "minmax"
assert cfg.norm_stats.reduce_all_but_last is True
assert cfg.norm_stats.sample_frac == 1.0
assert cfg.norm_stats.save_cache_dir is None
assert str(cfg.norm_stats.precomputed_norm_path) == norm_artifact
assert cfg.callbacks.model_checkpoint.every_n_train_steps == 2
assert cfg.callbacks.model_checkpoint.train_time_interval is None
assert cfg.callbacks.model_checkpoint.save_last is True
assert cfg.callbacks.terminal_checkpoint.every_n_train_steps == 1
recovery_callback = instantiate(cfg.callbacks.model_checkpoint)
terminal_callback = instantiate(cfg.callbacks.terminal_checkpoint)
assert recovery_callback.state_key != terminal_callback.state_key
assert model.action_horizon == 16
assert set(model.domains) == {
    "pushshapes_sim_u_socket",
    "pushshapes_sim_chain_gripper",
}
assert set(cfg.data.train_datasets) == set(model.domains)
assert set(cfg.data.valid_datasets) == set(model.domains)
assert all(
    dataset.mode == "train" and float(dataset.valid_ratio) == 0.0
    for dataset in cfg.data.train_datasets.values()
)
assert all(
    dataset.mode == "valid" and float(dataset.valid_ratio) == 0.02
    for dataset in cfg.data.valid_datasets.values()
)
for split in ("train_datasets", "valid_datasets"):
    chain_dataset = cfg.data[split].pushshapes_sim_chain_gripper
    filters = chain_dataset.filters
    assert filters._target_ == "egomimic.rldb.filters.DatasetFilter"
    assert list(filters.filter_lambdas) == [expected_chain_filter]
for domain in model.domains:
    assert cfg.data.train_dataloader_params[domain].batch_size == 32
    assert cfg.data.valid_dataloader_params[domain].batch_size == 16
chain = cfg.data.train_datasets.pushshapes_sim_chain_gripper.resolver
assert list(chain.folder_paths) == [
    "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_3000_v2",
    "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_gen",
]
assert cfg.evaluator._target_.endswith("HumanRobotOverlayEval")
assert "arc_length" not in pathlib.Path(path).read_text().lower()

stage = model.stages[1]
assert stage.action_horizon == 16
assert set(stage.policies) == set(model.domains)
assert all(
    policy.noise_scheduler.prediction_type == "epsilon"
    for policy in stage.policies.values()
)
PY

"$SRUN" --kill-on-bad-exit=1 --unbuffered \
  python -m egomimic.trainHydra "${COMMON_OVERRIDES[@]}" \
  2>&1 | tee "$TRAIN_LOG"

python "$VERIFIER" "$RUN_DIR" \
  --required-embodiments 19,20 \
  --expected-head "$EXPECTED_HEAD" \
  --expected-world-size "$EXPECTED_WORLD_SIZE" \
  --dry-run

# A live append or artifact mutation during the smoke invalidates the result.
validate_all_inventories
test "$(sha256sum "$NORM_ARTIFACT" | awk '{print $1}')" = "$EXPECTED_NORM_SHA"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$EXPECTED_LAUNCHER_SHA"

# Persist PASS only after the data, norm artifact, source tree, and launcher
# have all survived the postflight unchanged.
python "$VERIFIER" "$RUN_DIR" \
  --required-embodiments 19,20 \
  --expected-head "$EXPECTED_HEAD" \
  --expected-world-size "$EXPECTED_WORLD_SIZE"

printf '[smoke] ALL_PASS arm=dp hardware=a40 head=%s world_size=%s run=%s\n' \
  "$EXPECTED_HEAD" "$EXPECTED_WORLD_SIZE" "$RUN_DIR"
