#!/bin/bash
# Reusable smoke/full launcher for capacity-controlled U-Socket + ChainGripper
# H16 Diffusion Policy. The run bundle fixes every contract other than MODE.

set -Eeuo pipefail

MODE=${MODE:?set MODE=smoke or MODE=full}
EXPECTED_HEAD=${EXPECTED_HEAD:?set exact clean source commit}
EXPECTED_LAUNCHER_SHA=${EXPECTED_LAUNCHER_SHA:?set launcher SHA256}
EXPECTED_SPLIT_SHA=${EXPECTED_SPLIT_SHA:?set split manifest SHA256}
NORM_ARTIFACT=${NORM_ARTIFACT:?set train-only normalization artifact}
EXPECTED_NORM_SHA=${EXPECTED_NORM_SHA:?set normalization SHA256}
EXPECTED_SLURM_ACCOUNT=${EXPECTED_SLURM_ACCOUNT:?set exact authorized Slurm account}
EXPECTED_SLURM_PARTITION=${EXPECTED_SLURM_PARTITION:?set exact Slurm partition}
EXPECTED_GPU_MODEL=${EXPECTED_GPU_MODEL:?set nvidia-smi model substring, e.g. L40S or A40}
SMOKE_GATE=${SMOKE_GATE:-}
NORM_JOB_ROOT=$(dirname "$(dirname "$(dirname "$NORM_ARTIFACT")")")
NORM_VALIDATION=$NORM_JOB_ROOT/validation_recovered.json

REPO=/coc/flash7/paphiwetsa3/worktrees/flow-transfer-dp20m-newdata-h16-20260830
PY_ENV=/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv
EXP_ROOT=/coc/flash7/paphiwetsa3/experiments/flow_transfer_dp20m_newdata_val01_h16_20260830
LAUNCHER=$REPO/scripts/train/flow_transfer_dp_capacity_skynet_l40sx2.sbatch
VERIFIER=$REPO/egomimic/scripts/verify_training_smoke.py
SPLIT_MANIFEST=/coc/flash7/paphiwetsa3/experiments/flow_transfer_dp20m_newdata_val01_h16_20260830/preflight_20260830/dataset_split/split_manifest.json
EXPERIMENT=pusht/pipeline_diffusion_usocket_chain_newdata_val01_h16_20m
BASE_WANDB_ID=ft_dp20m_cotrain_newdata3719_h16_val01_s42_world2_20260831
WANDB_GROUP=flow_transfer_dp_capacity_val01_20260831
SLURM_BIN=/opt/slurm/Ubuntu-20.04/24.11.0/bin
export PATH=$SLURM_BIN:$PATH

assert_sha() {
  local expected=$1 path=$2
  test -s "$path"
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
}

case "$MODE" in
  smoke)
    test -z "$SMOKE_GATE"
    RUN_DIR=$EXP_ROOT/smokes/job_${SLURM_JOB_ID:?}
    WANDB_ID=${BASE_WANDB_ID}_smoke_${SLURM_JOB_ID}
    ;;
  full)
    SMOKE_GATE=${SMOKE_GATE:?full mode requires SMOKE_GATE}
    test -s "$SMOKE_GATE"
    "$PY_ENV/bin/python" - "$SMOKE_GATE" "$EXPECTED_HEAD" <<'PY'
import json, pathlib, sys
x = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert x["status"] == "passed"
assert x["repo_head"] == sys.argv[2]
assert x["global_step"] == 2 and x["world_size"] == 2
assert x["strict_reload"] is True
assert x["required_exact_metrics"] is True
PY
    RUN_DIR=$EXP_ROOT/runs/dp20m_seed42
    WANDB_ID=$BASE_WANDB_ID
    ;;
  *)
    printf 'Unknown MODE=%s\n' "$MODE" >&2
    exit 64
    ;;
esac

PROVENANCE_DIR=$EXP_ROOT/provenance/$MODE/job_${SLURM_JOB_ID:?}
test ! -e "$RUN_DIR"
test ! -e "$PROVENANCE_DIR"
mkdir -p "$RUN_DIR" "$PROVENANCE_DIR"

test "${SLURM_RESTART_COUNT:-0}" = 0
test "${SLURM_JOB_ACCOUNT:?}" = "$EXPECTED_SLURM_ACCOUNT"
test "${SLURM_JOB_PARTITION:?}" = "$EXPECTED_SLURM_PARTITION"
test "${SLURM_NTASKS:?}" = 2
test "${SLURM_JOB_NUM_NODES:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" -ge 8
GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader)
test "$(printf '%s\n' "$GPU_NAMES" | sed '/^[[:space:]]*$/d' | wc -l)" = 2
test "$(printf '%s\n' "$GPU_NAMES" | grep -Fci -- "$EXPECTED_GPU_MODEL")" = 2
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
assert_sha "$EXPECTED_LAUNCHER_SHA" "$LAUNCHER"
assert_sha "$EXPECTED_SPLIT_SHA" "$SPLIT_MANIFEST"
assert_sha "$EXPECTED_NORM_SHA" "$NORM_ARTIFACT"
test -s "$NORM_VALIDATION"
test -x "$PY_ENV/bin/python"
test -f "$VERIFIER"

cp "$LAUNCHER" "$SPLIT_MANIFEST" "$NORM_ARTIFACT" "$NORM_VALIDATION" \
  "$PROVENANCE_DIR/"
cp "$(dirname "$SPLIT_MANIFEST")"/*.tsv "$PROVENANCE_DIR/"
git -C "$REPO" log -1 --format=fuller > "$PROVENANCE_DIR/git_commit.txt"
git -C "$REPO" status --porcelain=v1 --untracked-files=all \
  > "$PROVENANCE_DIR/git_status.txt"
nvidia-smi -q > "$PROVENANCE_DIR/nvidia_smi_q.txt"
"$SLURM_BIN/scontrol" show job -dd "$SLURM_JOB_ID" \
  > "$PROVENANCE_DIR/slurm_job.txt"

source "$PY_ENV/bin/activate"
cd "$REPO"
export PYTHONPATH=$REPO
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true
export WANDB_MODE=online

OVERRIDES=(
  --config-name=train_zarr_cartesian
  "+experiment=$EXPERIMENT"
  mode=train
  ckpt_path=null
  "hydra.run.dir=$RUN_DIR"
  "++paths.root_dir=$RUN_DIR"
  "paths.output_dir=$RUN_DIR"
  "paths.work_dir=$REPO"
  launch_params.gpus_per_node=2
  launch_params.nodes=1
  trainer.devices=2
  trainer.num_nodes=1
  trainer.strategy=ddp
  trainer.sync_batchnorm=true
  trainer.accumulate_grad_batches=1
  trainer.gradient_clip_val=null
  trainer.log_every_n_steps=1
  data.train_dataloader_params.pushshapes_sim_u_socket.batch_size=32
  data.train_dataloader_params.pushshapes_sim_chain_gripper.batch_size=32
  data.valid_dataloader_params.pushshapes_sim_u_socket.batch_size=16
  data.valid_dataloader_params.pushshapes_sim_chain_gripper.batch_size=16
  "norm_stats.precomputed_norm_path=$NORM_ARTIFACT"
  norm_stats.save_cache_dir=null
  logger.wandb.offline=false
  logger.wandb.entity=rl2-group
  logger.wandb.project=pushshapes-flow-transfer
  "logger.wandb.group=$WANDB_GROUP"
  "logger.wandb.id=$WANDB_ID"
  "+logger.wandb.name=$WANDB_ID"
  +logger.wandb.resume=never
  "logger.wandb.tags=[flow-transfer,diffusion-policy,dp20m,h16,val01-seed42,fresh-start,batch64-per-embodiment,checkpoint-20k,skynet,gpu-auto,world2,chain-gen-frozen719]"
)

if test "$MODE" = smoke; then
  OVERRIDES+=(
    trainer.max_steps=2
    trainer.limit_train_batches=2
    trainer.val_check_interval=1
    trainer.limit_val_batches=1
    trainer.num_sanity_val_steps=0
    callbacks.model_checkpoint.every_n_train_steps=2
  )
else
  OVERRIDES+=(
    trainer.max_steps=240000
    trainer.limit_train_batches=1.0
    trainer.val_check_interval=10000
    trainer.limit_val_batches=8
    trainer.num_sanity_val_steps=0
    callbacks.model_checkpoint.every_n_train_steps=20000
  )
fi

RESOLVED=$PROVENANCE_DIR/resolved_config.yaml
python -m egomimic.trainHydra "${OVERRIDES[@]}" --cfg job --resolve > "$RESOLVED"

python - "$RESOLVED" "$MODE" "$NORM_ARTIFACT" "$WANDB_ID" <<'PY'
import math
import sys

from hydra.utils import instantiate
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
mode, norm_path, wandb_id = sys.argv[2:]
assert cfg.seed == 42 and cfg.mode == "train" and cfg.ckpt_path is None
assert cfg.trainer.precision == "bf16"
assert cfg.trainer.devices == 2 and cfg.trainer.num_nodes == 1
assert cfg.trainer.strategy == "ddp" and cfg.trainer.sync_batchnorm is True
assert cfg.trainer.accumulate_grad_batches == 1
assert cfg.trainer.gradient_clip_val is None
assert cfg.model.optimizer._target_ == "torch.optim.AdamW"
assert math.isclose(cfg.model.optimizer.lr, 3e-5)
assert math.isclose(cfg.model.optimizer.weight_decay, 1e-4)
assert cfg.model.scheduler.max_steps == 240000
assert cfg.model.scheduler.warmup_steps == 3000
assert math.isclose(cfg.model.scheduler.warmup_start_factor, 0.1)
assert math.isclose(cfg.model.scheduler.eta_min, 3e-6)
assert str(cfg.norm_stats.precomputed_norm_path) == norm_path
assert cfg.logger.wandb.id == cfg.logger.wandb.name == wandb_id
assert cfg.logger.wandb.resume == "never"
assert cfg.data.valid_combined_mode == "max_size"
domains = {"pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper"}
for domain in domains:
    train = cfg.data.train_datasets[domain]
    valid = cfg.data.valid_datasets[domain]
    assert train.mode == "train" and valid.mode == "valid"
    assert float(train.valid_ratio) == float(valid.valid_ratio) == 0.01
    assert cfg.data.train_dataloader_params[domain].batch_size == 32
pipeline = cfg.model.robomimic_model
assert pipeline.action_horizon == 16
assert pipeline.stages[0].n_obs_steps == 1
assert pipeline.stages[1].condition_input_dim == 67
assert set(pipeline.stages[1].policies) == domains
for domain, action_dim in {"pushshapes_sim_u_socket": 4, "pushshapes_sim_chain_gripper": 6}.items():
    policy = pipeline.stages[1].policies[domain]
    assert policy.action_horizon == 16
    assert policy.model.input_dim == action_dim
    assert list(policy.model.down_dims) == [72, 144, 288]
    assert policy.noise_scheduler.prediction_type == "epsilon"
encoder = instantiate(pipeline.stages[0])
diffusion = instantiate(pipeline.stages[1])
total = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in diffusion.parameters())
assert total == 19635818, total
assert cfg.evaluator.exact_epoch_metrics is True
assert cfg.evaluator.deterministic_seed == 42
checkpoint = cfg.callbacks.model_checkpoint
assert checkpoint.save_top_k == -1 and checkpoint.save_last is True
assert checkpoint.every_n_epochs is None and checkpoint.train_time_interval is None
assert checkpoint.auto_insert_metric_name is False
assert checkpoint.every_n_train_steps == (2 if mode == "smoke" else 20000)
PY

sha256sum "$RESOLVED" > "$PROVENANCE_DIR/resolved_config.sha256"
TRAIN_LOG=$RUN_DIR/training.log
"$SLURM_BIN/srun" --kill-on-bad-exit=1 --unbuffered \
  python -m egomimic.trainHydra "${OVERRIDES[@]}" 2>&1 | tee "$TRAIN_LOG"

assert_sha "$EXPECTED_LAUNCHER_SHA" "$LAUNCHER"
assert_sha "$EXPECTED_SPLIT_SHA" "$SPLIT_MANIFEST"
assert_sha "$EXPECTED_NORM_SHA" "$NORM_ARTIFACT"

if test "$MODE" = smoke; then
  python "$VERIFIER" "$RUN_DIR" --required-embodiments 19,20 \
    --expected-head "$EXPECTED_HEAD" --expected-world-size 2
  CUDA_VISIBLE_DEVICES= python - "$RUN_DIR/checkpoints/last.ckpt" <<'PY'
import json
import pathlib
import sys

from egomimic.pl_utils.pl_model import ModelWrapper

path = pathlib.Path(sys.argv[1]).resolve(strict=True)
# This checkpoint was created by this exact smoke from the pinned clean source.
# PyTorch 2.6 defaults torch.load to weights_only=True, which rejects Lightning's
# trusted OmegaConf hyperparameter metadata before strict state-dict checks run.
model = ModelWrapper.load_from_checkpoint(
    str(path), map_location="cpu", strict=True, weights_only=False
)
total = sum(parameter.numel() for parameter in model.parameters())
assert total == 19635818, total
record = {"status": "PASS", "strict": True, "parameters": total, "checkpoint": str(path)}
path.with_name("strict_reload.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
  python - "$RUN_DIR/SMOKE_RESULT.json" "$RUN_DIR/checkpoints/strict_reload.json" <<'PY'
import json
import pathlib
import sys

result_path, reload_path = map(pathlib.Path, sys.argv[1:])
result = json.loads(result_path.read_text())
reload = json.loads(reload_path.read_text())
assert reload["status"] == "PASS" and reload["strict"] is True
result["strict_reload"] = True
result["parameter_count"] = reload["parameters"]
result["required_exact_metrics"] = True
temporary = result_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
temporary.replace(result_path)
PY
  printf '[smoke] ALL_PASS job=%s wandb=%s\n' "$SLURM_JOB_ID" \
    "https://wandb.ai/rl2-group/pushshapes-flow-transfer/runs/$WANDB_ID"
else
  test -s "$RUN_DIR/checkpoints/last.ckpt"
  printf '[full] ALL_PASS job=%s wandb=%s\n' "$SLURM_JOB_ID" \
    "https://wandb.ai/rl2-group/pushshapes-flow-transfer/runs/$WANDB_ID"
fi
