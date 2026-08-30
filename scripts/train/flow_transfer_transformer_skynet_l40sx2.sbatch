#!/bin/bash
# Canonical preflight/smoke/full launcher for the Flow Transfer Transformer
# r02 cg32 multi-J training-unroll family. Submit on one Skynet node with two
# tasks and two matching GPUs.

set -Eeuo pipefail

report_error() {
  local status=$?
  local line=${BASH_LINENO[0]:-$LINENO}
  printf '[%s] FAIL arm=%s line=%s status=%s command=%q\n' \
    "${MODE:-unset}" "${ARM:-unset}" "$line" "$status" "$BASH_COMMAND" >&2
  exit "$status"
}
trap report_error ERR

MODE=${MODE:?set MODE=preflight, smoke, or full}
ARM=${ARM:?set ARM=r02_l8_cg32_multij64}
EXPECTED_HEAD=${EXPECTED_HEAD:?set exact clean source commit}
EXPECTED_LAUNCHER_SHA=${EXPECTED_LAUNCHER_SHA:?set launcher SHA256}
EXPECTED_SPLIT_SHA=${EXPECTED_SPLIT_SHA:?set split manifest SHA256}
EXPECTED_RUN_BUNDLE_SHA=${EXPECTED_RUN_BUNDLE_SHA:-}
SMOKE_GATE=${SMOKE_GATE:-}
EXPECTED_SMOKE_GATE_SHA=${EXPECTED_SMOKE_GATE_SHA:-}

REPO=/coc/flash7/paphiwetsa3/worktrees/flow-transfer-transformer3-multij64-20260831
PY_ENV=/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv
EXP_ROOT=/coc/flash7/paphiwetsa3/experiments/flow_transfer_transformer3_cg32_jmix2468163264_051010101550_a40x2_20260831
LAUNCHER=$REPO/scripts/train/flow_transfer_transformer_skynet_l40sx2.sbatch
VERIFIER=$REPO/egomimic/scripts/verify_training_smoke.py
RUN_BUNDLE=$EXP_ROOT/RUN_BUNDLE_PROVENANCE.json
SUPERSEDED_BUNDLE=/coc/flash7/paphiwetsa3/experiments/flow_transfer_transformer3_cg32_jmix2468163264_051010101550_20260831/RUN_BUNDLE_PROVENANCE.json
SUPERSEDED_BUNDLE_SHA=0faef0cc302c0ebccedeb3f70418df821c7d0b85d8a6b63e8824b7e4c8f6dcbb
DATA_FILE=$REPO/egomimic/hydra_configs/data/pusht/flow_transfer_usocket_chain_newdata_val01_h16_per_emb_proprio.yaml
NORM_ARTIFACT=/coc/flash7/paphiwetsa3/experiments/flow_transfer_temporal_h8_l8_world2_smokes_20260829/norm_artifacts/per_emb_proprio_h16/fast_merge_20260829/artifact/norm_stats/norm_stats.json
SPLIT_MANIFEST=$EXP_ROOT/source_artifacts/splits/seed42_val01_skynet.json
U_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/u_socket_3000_v2_clean
CHAIN_BASE_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_3000_v2
CHAIN_GEN_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_gen_flow_transfer_frozen719_20260829
EXCLUDED_CHAIN_EPISODE=episode_T_chain_gripper_obs7_000050.zarr
EXPECTED_NORM_SHA=d6abc08a3315cd172bd6c03ded0346ae24d69f2ce1744e133fe1768f0feefa96
EXPECTED_STAGE_SHA=caf081c136f42318e172007edc91e0c0073235c74dd0ad4ee8e8d2da43e43a34
EXPECTED_TEST_SHA=23b628b043b66a852e2313a0c1b66e7353f3af6a5542e3afbb218dcf3428d510
EXPECTED_VERIFIER_SHA=28b73fe625cb305605110c7e5188314cc027578a1023751ed1c7449fb953bcd1
EXPECTED_DATA_SHA=93297962d4343f6b085ea929c4fce2e2f03a8101ddd183d42563ea96ed9c3d9f
EXPECTED_MODEL_SHA=4047c44b8367c453da836db41353f320e4bbe8e530b7500c6716f17604aed1ef
EXPECTED_EXPERIMENT_SHA=4fc538e76a0808c0ecbfa70d723da2a003952faca02b09747201c80821de6d78
WANDB_GROUP=flow_transfer_transformer3_cg32_multij_ablation_20260831
EXPECTED_WORLD_SIZE=2
EXPECTED_SLURM_ACCOUNT=${EXPECTED_SLURM_ACCOUNT:?set expected authorized Skynet account}
EXPECTED_SLURM_PARTITION=${EXPECTED_SLURM_PARTITION:?set expected Skynet partition}
EXPECTED_SLURM_QOS=${EXPECTED_SLURM_QOS:?set expected Skynet QoS}
EXPECTED_GPU_MODEL=${EXPECTED_GPU_MODEL:?set observed nvidia-smi GPU model substring}
TRAINING_GPU_MODEL=${TRAINING_GPU_MODEL:?set fixed smoke/full GPU model substring}
SLURM_BIN=/opt/slurm/Ubuntu-20.04/24.11.0/bin
export PATH=$SLURM_BIN:$PATH

case "$ARM" in
  r02_l8_cg32_multij64)
    EXPERIMENT=pusht/pipeline_sampler_usocket_chain_newdata_r02_transformer_cg32_multij64_val01_h16
    MODEL_FILE=$REPO/egomimic/hydra_configs/model/bf/bf_pipeline_sampler_usocket_chain_points_r02_transformer_cg32_multij64_per_emb_proprio_h16.yaml
    EXPERIMENT_FILE=$REPO/egomimic/hydra_configs/experiment/pusht/pipeline_sampler_usocket_chain_newdata_r02_transformer_cg32_multij64_val01_h16.yaml
    EXPECTED_LATENT_DIM=8
    EXPECTED_DENOISER_WIDTH=256
    EXPECTED_USOCKET_DECODER_WIDTH=32
    EXPECTED_CHAIN_DECODER_WIDTH=32
    BASE_WANDB_ID=ft_r02_l8_w256_d8_c128_transformer3_decu32_cg32_jmix2468163264_p051010101550_s42_skynet_a40x2_20260831
    ;;
  *)
    printf 'Unknown ARM=%s\n' "$ARM" >&2
    exit 64
    ;;
esac

case "$MODE" in
  preflight)
    test -z "$SMOKE_GATE$EXPECTED_RUN_BUNDLE_SHA$EXPECTED_SMOKE_GATE_SHA"
    test ! -e "$RUN_BUNDLE"
    RUN_DIR=$EXP_ROOT/preflights/$ARM/job_${SLURM_JOB_ID:?}
    WANDB_ID=${BASE_WANDB_ID}_preflight_job_${SLURM_JOB_ID}
    WANDB_OFFLINE=true
    ;;
  smoke)
    test -z "$SMOKE_GATE$EXPECTED_SMOKE_GATE_SHA"
    test -n "$EXPECTED_RUN_BUNDLE_SHA"
    RUN_DIR=$EXP_ROOT/smokes/$ARM/job_${SLURM_JOB_ID:?}
    WANDB_ID=${BASE_WANDB_ID}_smoke_job_${SLURM_JOB_ID}
    WANDB_OFFLINE=false
    ;;
  full)
    test -n "$EXPECTED_RUN_BUNDLE_SHA"
    SMOKE_GATE=${SMOKE_GATE:?full mode requires SMOKE_GATE}
    EXPECTED_SMOKE_GATE_SHA=${EXPECTED_SMOKE_GATE_SHA:?set smoke-gate SHA256}
    test "$(sha256sum "$SMOKE_GATE" | awk '{print $1}')" = "$EXPECTED_SMOKE_GATE_SHA"
    "$PY_ENV/bin/python" - "$SMOKE_GATE" "$EXPECTED_HEAD" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert payload["status"] == "passed"
assert payload["repo_head"] == sys.argv[2]
assert payload["world_size"] == 2
assert payload["global_step"] == 2020
assert payload["validation_trainer_global_step"] >= 2000
assert payload["required_embodiments"] == [19, 20]
assert payload["strict_reload"] is True
assert payload["wandb_api_visible"] is True
assert payload["required_exact_metrics"] is True
assert payload["observed_post_warmup_unroll_counts"] == {
    "2": 1, "4": 2, "8": 2, "16": 2, "32": 3, "64": 10,
}
PY
    RUN_DIR=$EXP_ROOT/runs/$ARM
    WANDB_ID=$BASE_WANDB_ID
    WANDB_OFFLINE=false
    ;;
  *)
    printf 'Unknown MODE=%s\n' "$MODE" >&2
    exit 64
    ;;
esac

ATTEMPT=${SLURM_RESTART_COUNT:-0}
PROVENANCE_DIR=$EXP_ROOT/provenance/$MODE/$ARM/job_${SLURM_JOB_ID:?}_attempt_${ATTEMPT}
if test "$MODE" != full || test "$ATTEMPT" = 0; then
  test ! -e "$RUN_DIR"
else
  test -d "$RUN_DIR"
fi
test ! -e "$PROVENANCE_DIR"
mkdir -p "$RUN_DIR" "$PROVENANCE_DIR"

assert_sha() {
  local expected=$1 path=$2
  test -s "$path"
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
}

assert_sha "$EXPECTED_LAUNCHER_SHA" "$LAUNCHER"
assert_sha "$EXPECTED_NORM_SHA" "$NORM_ARTIFACT"
assert_sha "$EXPECTED_SPLIT_SHA" "$SPLIT_MANIFEST"
assert_sha "$EXPECTED_STAGE_SHA" "$REPO/egomimic/pipeline/stages_sampler.py"
assert_sha "$EXPECTED_TEST_SHA" "$REPO/tests/test_pipeline_sampler_stage.py"
assert_sha "$EXPECTED_VERIFIER_SHA" "$VERIFIER"
assert_sha "$EXPECTED_DATA_SHA" "$DATA_FILE"
assert_sha "$EXPECTED_MODEL_SHA" "$MODEL_FILE"
assert_sha "$EXPECTED_EXPERIMENT_SHA" "$EXPERIMENT_FILE"
assert_sha "$SUPERSEDED_BUNDLE_SHA" "$SUPERSEDED_BUNDLE"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
if test "$MODE" != preflight; then
  assert_sha "$EXPECTED_RUN_BUNDLE_SHA" "$RUN_BUNDLE"
fi
test -x "$PY_ENV/bin/python"
test -x "$SLURM_BIN/srun"
test -f "$VERIFIER"
test -d "$U_DATA"
test -d "$CHAIN_BASE_DATA"
test -d "$CHAIN_GEN_DATA"
test ! -e "$CHAIN_GEN_DATA/$EXCLUDED_CHAIN_EPISODE"
test "$(find -L "$U_DATA" -mindepth 1 -maxdepth 1 -type d -name '*.zarr' | wc -l)" -eq 2999
test "$(find -L "$CHAIN_BASE_DATA" -mindepth 1 -maxdepth 1 -type d -name '*.zarr' | wc -l)" -eq 3000
test "$(find -L "$CHAIN_GEN_DATA" -mindepth 1 -maxdepth 1 -type d -name '*.zarr' | wc -l)" -eq 719

"$PY_ENV/bin/python" - "$SPLIT_MANIFEST" "$U_DATA" "$CHAIN_BASE_DATA" "$CHAIN_GEN_DATA" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert payload["status"] == "PASS"
assert payload["seed"] == 42
assert payload["valid_ratio"] == 0.01
assert payload["roots"] == {"usocket": sys.argv[2], "chain": [sys.argv[3], sys.argv[4]]}
assert payload["counts"] == {
    "usocket_total": 2999,
    "usocket_train": 2970,
    "usocket_valid": 29,
    "chain_total": 3719,
    "chain_train": 3682,
    "chain_valid": 37,
    "chain_by_source": [3000, 719],
}
assert all(value == 0 for value in payload["overlap"].values())
PY

if test "$MODE" != full; then
  test "$ATTEMPT" = 0
fi
if test "$MODE" != preflight; then
  test "$EXPECTED_GPU_MODEL" = "$TRAINING_GPU_MODEL"
fi
test "${SLURM_JOB_ACCOUNT:?}" = "$EXPECTED_SLURM_ACCOUNT"
test "${SLURM_JOB_PARTITION:?}" = "$EXPECTED_SLURM_PARTITION"
test "${SLURM_JOB_QOS:?}" = "$EXPECTED_SLURM_QOS"
test "${SLURM_NTASKS:?}" = "$EXPECTED_WORLD_SIZE"
test "${SLURM_JOB_NUM_NODES:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" -ge 8
GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader)
test "$(grep -Fci "$EXPECTED_GPU_MODEL" <<< "$GPU_NAMES")" = "$EXPECTED_WORLD_SIZE"

nvidia-smi -q > "$PROVENANCE_DIR/nvidia_smi_q.txt"
cp "$LAUNCHER" "$MODEL_FILE" "$EXPERIMENT_FILE" "$DATA_FILE" \
  "$SPLIT_MANIFEST" "$NORM_ARTIFACT" "$PROVENANCE_DIR/"
if test "$MODE" != preflight; then
  cp "$RUN_BUNDLE" "$PROVENANCE_DIR/"
fi
git -C "$REPO" log -1 --format=fuller > "$PROVENANCE_DIR/git_commit.txt"
git -C "$REPO" status --porcelain=v1 --untracked-files=all \
  > "$PROVENANCE_DIR/git_status.txt"
printf '%s\n' "$GPU_NAMES" > "$PROVENANCE_DIR/gpu_names.txt"
"$PY_ENV/bin/python" --version > "$PROVENANCE_DIR/python_version.txt" 2>&1
"$SLURM_BIN/scontrol" show job -dd "$SLURM_JOB_ID" > "$PROVENANCE_DIR/slurm_job.txt"

cd "$REPO"
source "$PY_ENV/bin/activate"
export PYTHONPATH=$REPO
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
if test "$WANDB_OFFLINE" = true; then
  export WANDB_MODE=offline
else
  export WANDB_MODE=online
fi

RESUME_CKPT=
WANDB_RESUME=never
if test "$MODE" = full && test "$ATTEMPT" != 0; then
  RESUME_CKPT=$("$PY_ENV/bin/python" - "$RUN_DIR" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
candidates = [*root.glob("hpc_ckpt_*.ckpt"), *root.glob("checkpoints/**/*.ckpt")]
candidates = [path for path in candidates if path.is_file()]
assert candidates, root
print(max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path))))
PY
)
  WANDB_RESUME=allow
fi

COMMON_OVERRIDES=(
  --config-name=train_zarr_cartesian
  "+experiment=$EXPERIMENT"
  mode=train
  "ckpt_path=${RESUME_CKPT:-null}"
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
  "callbacks.model_checkpoint.filename='epoch-{epoch}-step-{step}'"
  data.train_dataloader_params.pushshapes_sim_u_socket.batch_size=32
  data.train_dataloader_params.pushshapes_sim_chain_gripper.batch_size=32
  data.valid_dataloader_params.pushshapes_sim_u_socket.batch_size=16
  data.valid_dataloader_params.pushshapes_sim_chain_gripper.batch_size=16
  "data.train_datasets.pushshapes_sim_u_socket.resolver.folder_path=$U_DATA"
  "data.valid_datasets.pushshapes_sim_u_socket.resolver.folder_path=$U_DATA"
  "data.train_datasets.pushshapes_sim_chain_gripper.resolver.folder_paths=[$CHAIN_BASE_DATA,$CHAIN_GEN_DATA]"
  "data.valid_datasets.pushshapes_sim_chain_gripper.resolver.folder_paths=[$CHAIN_BASE_DATA,$CHAIN_GEN_DATA]"
  "norm_stats.precomputed_norm_path=$NORM_ARTIFACT"
  norm_stats.save_cache_dir=null
  "logger.wandb.offline=$WANDB_OFFLINE"
  logger.wandb.entity=rl2-group
  logger.wandb.project=pushshapes-flow-transfer
  "logger.wandb.group=$WANDB_GROUP"
  "logger.wandb.id=$WANDB_ID"
  "logger.wandb.name=$WANDB_ID"
  "logger.wandb.resume=$WANDB_RESUME"
  "logger.wandb.tags=[flow-transfer,transformer-action-decoder,transformer-decoder-3blocks,symmetric-decoder-width,chain-decoder-32,condition-dim-128,$ARM,latent-dense,h16,j8,train-jmix-2-4-8-16-32-64,train-jmix-p05-10-10-10-15-50,val01-seed42,fresh-start,batch64-per-embodiment,checkpoint-20k,skynet,a40x2,world2,chain-gen-frozen719,nccl-p2p-disabled]"
)

if test "$MODE" = smoke; then
  COMMON_OVERRIDES+=(
    trainer.max_steps=2020
    trainer.limit_train_batches=2020
    trainer.val_check_interval=2001
    trainer.limit_val_batches=1
    trainer.num_sanity_val_steps=0
    callbacks.model_checkpoint.monitor=null
    callbacks.model_checkpoint.save_top_k=-1
    callbacks.model_checkpoint.save_last=true
    callbacks.model_checkpoint.every_n_epochs=null
    callbacks.model_checkpoint.every_n_train_steps=2020
    callbacks.model_checkpoint.train_time_interval=null
    callbacks.model_checkpoint.save_on_train_epoch_end=false
  )
else
  COMMON_OVERRIDES+=(
    trainer.max_steps=240000
    trainer.limit_train_batches=1.0
    trainer.val_check_interval=10000
    trainer.limit_val_batches=8
    trainer.num_sanity_val_steps=0
    callbacks.model_checkpoint.monitor=null
    callbacks.model_checkpoint.save_top_k=-1
    callbacks.model_checkpoint.save_last=true
    callbacks.model_checkpoint.every_n_epochs=null
    callbacks.model_checkpoint.every_n_train_steps=20000
    callbacks.model_checkpoint.train_time_interval=null
    callbacks.model_checkpoint.save_on_train_epoch_end=false
  )
fi

RESOLVED_CONFIG=$PROVENANCE_DIR/resolved_config.yaml
"$PY_ENV/bin/python" -m egomimic.trainHydra "${COMMON_OVERRIDES[@]}" \
  --cfg job --resolve > "$RESOLVED_CONFIG"

"$PY_ENV/bin/python" - \
  "$RESOLVED_CONFIG" "$MODE" "$EXPECTED_LATENT_DIM" \
  "$EXPECTED_DENOISER_WIDTH" "$EXPECTED_USOCKET_DECODER_WIDTH" \
  "$EXPECTED_CHAIN_DECODER_WIDTH" "$NORM_ARTIFACT" "$U_DATA" \
  "$CHAIN_BASE_DATA" "$CHAIN_GEN_DATA" "$WANDB_ID" "$WANDB_GROUP" <<'PY'
import math
import pathlib
import sys

from omegaconf import OmegaConf

(
    config_path,
    mode,
    latent_dim,
    denoiser_width,
    usocket_decoder_width,
    chain_decoder_width,
    norm_path,
    u_data,
    chain_base,
    chain_gen,
    wandb_id,
    wandb_group,
) = sys.argv[1:]
cfg = OmegaConf.load(config_path)
latent_dim = int(latent_dim)
denoiser_width = int(denoiser_width)
expected_decoder_widths = {
    "pushshapes_sim_u_socket": int(usocket_decoder_width),
    "pushshapes_sim_chain_gripper": int(chain_decoder_width),
}

assert cfg.seed == 42 and cfg.mode == "train"
if mode != "full":
    assert cfg.ckpt_path is None
assert cfg.trainer.precision == "bf16"
assert cfg.trainer.devices == 2 and cfg.trainer.num_nodes == 1
assert cfg.trainer.strategy == "ddp" and cfg.trainer.sync_batchnorm is True
assert cfg.trainer.accumulate_grad_batches == 1
assert cfg.trainer.gradient_clip_val is None
assert cfg.trainer.log_every_n_steps == 1
assert cfg.model.train_metrics_on_step is True
assert math.isclose(cfg.model.optimizer.lr, 3.0e-5)
assert cfg.model.scheduler.max_steps == 240_000
assert cfg.model.scheduler.warmup_steps == 3_000
assert math.isclose(cfg.model.scheduler.eta_min, 3.0e-6)
assert str(cfg.norm_stats.precomputed_norm_path) == norm_path
assert cfg.norm_stats.save_cache_dir is None
assert cfg.logger.wandb.project == "pushshapes-flow-transfer"
assert cfg.logger.wandb.entity == "rl2-group"
assert cfg.logger.wandb.group == wandb_group
assert cfg.logger.wandb.id == wandb_id
if cfg.ckpt_path is None:
    assert cfg.logger.wandb.resume == "never"
else:
    assert mode == "full" and cfg.logger.wandb.resume == "allow"

domains = {"pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper"}
assert set(cfg.data.train_datasets) == domains
assert set(cfg.data.valid_datasets) == domains
assert cfg.data.valid_combined_mode == "max_size"
for domain in domains:
    train = cfg.data.train_datasets[domain]
    valid = cfg.data.valid_datasets[domain]
    assert train.mode == "train" and valid.mode == "valid"
    assert train.valid_ratio == valid.valid_ratio == 0.01
    assert cfg.data.train_dataloader_params[domain].batch_size == 32
    assert train.resolver.key_map.action_horizon == 16
    assert valid.resolver.key_map.action_horizon == 16
assert str(cfg.data.train_datasets.pushshapes_sim_u_socket.resolver.folder_path) == u_data
assert str(cfg.data.valid_datasets.pushshapes_sim_u_socket.resolver.folder_path) == u_data
for split in ("train_datasets", "valid_datasets"):
    chain = cfg.data[split].pushshapes_sim_chain_gripper.resolver
    assert list(chain.folder_paths) == [chain_base, chain_gen]
    assert chain.transform_list._target_.endswith("get_chain_gripper_point_transform_list")
    usocket = cfg.data[split].pushshapes_sim_u_socket.resolver
    assert usocket.transform_list._target_.endswith("get_usocket_rotvec_action_state_transform_list")

pipeline = cfg.model.robomimic_model
assert pipeline.action_horizon == 16
assert pipeline.rollout_adapters.pushshapes_sim_u_socket._target_.endswith("USocketRotVecRolloutAdapter")
assert pipeline.rollout_adapters.pushshapes_sim_chain_gripper._target_.endswith("ChainGripperPointRolloutAdapter")
proprio = pipeline.stages[0]
assert proprio.projections.pushshapes_sim_u_socket.source_dim == 4
assert proprio.projections.pushshapes_sim_chain_gripper.source_dim == 6
noise = pipeline.stages[2]
sampler = pipeline.stages[3]
decoder_router = pipeline.stages[4]
loss = pipeline.stages[5]
assert noise.num_tokens == 16 and noise.latent_dim == latent_dim
assert sampler.latent_dim == latent_dim
assert sampler.condition_input_dim == sampler.condition_dim == 128
assert sampler.denoiser_hidden_dim == denoiser_width
assert sampler.num_inference_steps == 8
assert sampler.sampling_schedule == {
    1: {1: 0.50, 2: 0.50},
    2001: {2: 0.05, 4: 0.10, 8: 0.10, 16: 0.10, 32: 0.15, 64: 0.50},
}
assert sampler.denoising_module.nblocks == 8
assert sampler.denoising_module.hidden_dim == denoiser_width
assert sampler.denoising_module.cond_dim == 128
assert decoder_router._target_.endswith("PerEmbodimentActionDecoder")
for domain, action_dim in {
    "pushshapes_sim_u_socket": 4,
    "pushshapes_sim_chain_gripper": 6,
}.items():
    decoder = decoder_router.decoders[domain]
    assert decoder._target_.endswith("TransformerActionDecoder")
    assert decoder.latent_dim == latent_dim
    assert decoder.hidden_dim == expected_decoder_widths[domain]
    assert decoder.action_dim == action_dim
    assert decoder.num_blocks == 3 and decoder.num_heads == 4
    assert decoder.ff_ratio == 4 and decoder.output_num_layers == 2
    assert math.isclose(decoder.dropout, 0.1)
assert loss._target_.endswith("NativeActionMSELoss")
assert cfg.evaluator._target_.endswith("HumanRobotOverlayEval")

checkpoint = cfg.callbacks.model_checkpoint
assert checkpoint.monitor is None
assert checkpoint.filename == "epoch-{epoch}-step-{step}"
assert checkpoint.save_last is True
assert checkpoint.save_top_k == -1
assert checkpoint.every_n_epochs is None
assert checkpoint.train_time_interval is None
assert checkpoint.save_on_train_epoch_end is False
if mode == "smoke":
    assert cfg.trainer.max_steps == 2020
    assert cfg.trainer.limit_train_batches == 2020
    assert cfg.trainer.val_check_interval == 2001
    assert cfg.trainer.limit_val_batches == 1
    assert checkpoint.every_n_train_steps == 2020
else:
    assert mode in {"preflight", "full"}
    assert cfg.trainer.max_steps == 240_000
    assert cfg.trainer.limit_train_batches == 1.0
    assert cfg.trainer.val_check_interval == 10_000
    assert cfg.trainer.limit_val_batches == 8
    assert checkpoint.every_n_train_steps == 20_000

assert "arc_length" not in pathlib.Path(config_path).read_text().lower()
PY

sha256sum "$RESOLVED_CONFIG" > "$PROVENANCE_DIR/resolved_config.sha256"
if test "$MODE" = preflight; then
  PREFLIGHT_RESULT=$RUN_DIR/PREFLIGHT_RESULT.json
  "$PY_ENV/bin/python" -m pytest -q \
    tests/test_pipeline_sampler_stage.py \
    tests/test_fixed_validation_loader.py \
    tests/test_human_robot_overlay_eval.py
  "$PY_ENV/bin/python" - \
    "$RESOLVED_CONFIG" "$RUN_BUNDLE" "$PREFLIGHT_RESULT" \
    "$SPLIT_MANIFEST" "$NORM_ARTIFACT" "$LAUNCHER" \
    "$REPO/egomimic/pipeline/stages_sampler.py" \
    "$REPO/tests/test_pipeline_sampler_stage.py" "$MODEL_FILE" \
    "$EXPERIMENT_FILE" "$EXPECTED_HEAD" "$EXP_ROOT" "$BASE_WANDB_ID" \
    "$WANDB_GROUP" "$EXPECTED_SLURM_ACCOUNT" "$EXPECTED_SLURM_PARTITION" \
    "$EXPECTED_SLURM_QOS" "$EXPECTED_GPU_MODEL" \
    "$TRAINING_GPU_MODEL" "$SUPERSEDED_BUNDLE" \
    "$SUPERSEDED_BUNDLE_SHA" <<'PY'
import copy
import hashlib
import json
import pathlib
import shutil
import sys

import hydra
from omegaconf import OmegaConf
import wandb

from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.trainHydra import _instantiate_model_wrapper


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


(
    config_path,
    bundle_path,
    result_path,
    split_path,
    norm_path,
    launcher_path,
    stage_path,
    test_path,
    model_path,
    experiment_path,
    head,
    exp_root,
    wandb_id,
    wandb_group,
    account,
    partition,
    qos,
    gpu_model,
    training_gpu_model,
    superseded_bundle,
    superseded_bundle_sha,
) = sys.argv[1:]
config = OmegaConf.load(config_path)
train_datasets = {
    name: hydra.utils.instantiate(dataset_config)
    for name, dataset_config in config.data.train_datasets.items()
}
norm_stats = MultiDataset(
    state={},
    norm_mode=str(config.norm_stats.norm_mode),
    reduce_all_but_last=bool(config.norm_stats.reduce_all_but_last),
)
norm_stats.populate_from_datasets(train_datasets)
for dataset_name, dataset in train_datasets.items():
    norm_stats.infer_shapes_from_batch(dataset[0], dataset_name)
    norm_config = copy.deepcopy(config.data.train_datasets[dataset_name])
    key_map = OmegaConf.to_container(
        norm_config.resolver.key_map, resolve=False
    )
    key_map["norm_mode"] = True
    norm_config.resolver.key_map = key_map
    norm_dataset = hydra.utils.instantiate(norm_config)
    norm_stats.infer_norm_from_dataset(
        norm_dataset,
        dataset_name,
        sample_frac=float(config.norm_stats.sample_frac),
        num_workers=int(config.norm_stats.num_workers),
        precomputed_norm_path=str(config.norm_stats.precomputed_norm_path),
    )
model = _instantiate_model_wrapper(config, norm_stats)
total_parameters = sum(value.numel() for value in model.parameters())
trainable_parameters = sum(
    value.numel() for value in model.parameters() if value.requires_grad
)
assert total_parameters == trainable_parameters == 45_218_674
del model

split = json.loads(pathlib.Path(split_path).read_text())
assert split["overlap"] == {
    "chain_ids": 0,
    "chain_realpaths": 0,
    "usocket_ids": 0,
    "usocket_realpaths": 0,
}

def list_sha(rows):
    data = "".join(f"{row['id']}\n" for row in rows).encode()
    return hashlib.sha256(data).hexdigest()

episode_list_sha = {
    f"{domain}.{split_name}": list_sha(rows)
    for domain, splits in split["splits"].items()
    for split_name, rows in splits.items()
}
wandb_path = f"rl2-group/pushshapes-flow-transfer/{wandb_id}"
try:
    existing = wandb.Api(timeout=30).run(wandb_path)
except wandb.errors.CommError as exc:
    message = str(exc).lower()
    assert "not found" in message or "could not find run" in message
else:
    raise AssertionError(f"W&B run ID exists: {existing.url} state={existing.state}")

checkpoint_bytes = total_parameters * 18
free_bytes = shutil.disk_usage(pathlib.Path(exp_root).parent).free
bundle = {
    "schema_version": 1,
    "status": "PREFLIGHT_PASS_SMOKE_PENDING",
    "created_date": "2026-08-31",
    "purpose": "Linked r02 Transformer-3 cg32 model with requested post-step-2000 multi-J distribution",
    "parent_run": {
        "wandb": "rl2-group/pushshapes-flow-transfer/ft_r02_l8_w256_d8_c128_transformer3_decu32_cg32_s42_pace_rtxpro6000bx2_20260830",
        "platform": "PACE, 2 x RTX PRO 6000 Blackwell",
    },
    "comparison": {
        "single_intended_model_delta": "training unroll schedule after step 2000",
        "matched": "architecture, data/split, normalization, optimizer, seed, batch, precision, budget, validation/checkpoint cadence, and inference J=8",
        "caveat": "Skynet GPU/runtime differs from the linked PACE Blackwell run",
        "fairness_verdict": "FAIR WITH CAVEATS",
    },
    "replacement": {
        "reason": "User explicitly authorized 2 x A40 to minimize Skynet queue wait",
        "supersedes_bundle": superseded_bundle,
        "supersedes_bundle_sha256": superseded_bundle_sha,
        "semantic_changes": ["scheduler allocation/account", "GPU class", "W&B identity", "output root"],
        "model_data_optimizer_changes": [],
    },
    "source": {
        "repository": str(pathlib.Path(model_path).parents[4]),
        "commit": head,
        "dirty_state": "clean committed Skynet worktree",
        "launcher": launcher_path,
        "critical_sha256": {
            "launcher": sha256(launcher_path),
            "stages_sampler": sha256(stage_path),
            "sampler_tests": sha256(test_path),
            "model_config": sha256(model_path),
            "experiment_config": sha256(experiment_path),
        },
    },
    "resolved_config": {"path": config_path, "sha256": sha256(config_path)},
    "model_contract": {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "denoising_state_dimension": 16 * 8,
        "denoising_state_expression": "GaussianLatentNoise.num_tokens * latent_dim = 16 * 8",
        "latent_dimension": 8,
        "condition_input_dimension": 128,
        "condition_dimension": 128,
        "output_action_dimensions": {"pushshapes_sim_u_socket": 4, "pushshapes_sim_chain_gripper": 6},
        "action_representations": {"pushshapes_sim_u_socket": "x_y_rotvec", "pushshapes_sim_chain_gripper": "agent/object point action"},
        "prediction_horizon": 16,
        "configured_execution_horizon": 16,
        "observation_history": 1,
        "modalities": ["front_img_1_96x96", "per-embodiment proprio projection"],
        "denoiser": "CrossTransformer(width=256, blocks=8, heads=8)",
        "action_decoders": "per-embodiment TransformerActionDecoder(width=32, blocks=3, heads=4)",
        "inference_steps": 8,
        "training_sampling_schedule": {
            "1": {"1": 0.50, "2": 0.50},
            "2001": {"2": 0.05, "4": 0.10, "8": 0.10, "16": 0.10, "32": 0.15, "64": 0.50},
        },
        "post_warmup_mean_unroll_steps": 39.7,
    },
    "optimizer_contract": {
        "optimizer": "AdamW",
        "learning_rate": 3.0e-5,
        "weight_decay": 1.0e-4,
        "scheduler": "warmup_cosine",
        "warmup_steps": 3000,
        "warmup_start_learning_rate": 3.0e-6,
        "minimum_learning_rate": 3.0e-6,
        "gradient_clipping": None,
        "precision": "bf16",
        "microbatch_per_embodiment_per_rank": 32,
        "gradient_accumulation": 1,
        "world_size": 2,
        "effective_global_batch": 128,
        "training_seed": 42,
        "optimizer_step_budget": 240_000,
        "validation_cadence_global_steps": 10_000,
        "validation_batches": 8,
        "resume_precedence": "fresh run; on Slurm requeue restore newest hpc/cadence checkpoint before W&B resume=allow",
    },
    "dataset_contract": {
        "valid_ratio": 0.01,
        "split_seed": 42,
        "split_before_windows": True,
        "source_namespaced_episode_ids": True,
        "counts": split["counts"],
        "inventory_sha256": split["inventory_sha256"],
        "episode_list_sha256": episode_list_sha,
        "overlap": split["overlap"],
        "union_is_complete_corpus": True,
        "split_manifest": split_path,
        "split_manifest_sha256": sha256(split_path),
        "normalization_artifact": norm_path,
        "normalization_sha256": sha256(norm_path),
        "normalization_scope": "training episodes only; reused verified artifact",
    },
    "logging_contract": {
        "entity": "rl2-group",
        "project": "pushshapes-flow-transfer",
        "id": wandb_id,
        "group": wandb_group,
        "resume": "never initially; allow only after Slurm requeue",
        "axis": "trainer/global_step",
        "required_metrics": [
            "Train/MSE", "Train/MSE/pushshapes_sim_u_socket", "Train/MSE/pushshapes_sim_chain_gripper",
            "Valid/MSE", "Valid/MSE/pushshapes_sim_u_socket", "Valid/MSE/pushshapes_sim_chain_gripper",
            "Valid/Native_MSE", "Valid/Native_MSE/pushshapes_sim_u_socket", "Valid/Native_MSE/pushshapes_sim_chain_gripper",
        ],
    },
    "checkpoint_contract": {
        "cadence_steps": 20_000,
        "filename": "epoch-{epoch}-step-{step}",
        "save_top_k": -1,
        "save_last": True,
        "terminal_step": 240_000,
        "slurm_signal_requeue_recovery": True,
        "estimated_checkpoint_bytes": checkpoint_bytes,
        "estimated_total_bytes": checkpoint_bytes * 14,
        "free_bytes": free_bytes,
        "directory": str(pathlib.Path(exp_root) / "runs/r02_l8_cg32_multij64/checkpoints"),
    },
    "resources": {
        "cluster": "Skynet", "account": account, "partition": partition,
        "qos": qos, "gpu": training_gpu_model,
        "preflight_observed_gpu": gpu_model,
        "world_size": 2, "cpus": 16, "memory": "256G",
    },
    "outputs": {
        "experiment_root": exp_root,
        "full_run": str(pathlib.Path(exp_root) / "runs/r02_l8_cg32_multij64"),
    },
    "smoke_gate": {
        "status": "PENDING",
        "requirements": "2020 real optimizer steps, scheduled validation, exact 20-step post-warmup J cycle, finite required W&B metrics, checkpoint, strict reload",
    },
}
assert free_bytes > bundle["checkpoint_contract"]["estimated_total_bytes"] * 2
bundle_target = pathlib.Path(bundle_path)
temporary_bundle = bundle_target.with_suffix(".json.tmp")
temporary_bundle.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
temporary_bundle.replace(bundle_target)

record = {
    "status": "passed",
    "resolved_config": config_path,
    "resolved_config_sha256": sha256(config_path),
    "run_bundle": bundle_path,
    "run_bundle_sha256": sha256(bundle_path),
    "total_parameters": total_parameters,
    "trainable_parameters": trainable_parameters,
    "post_warmup_mean_unroll_steps": 39.7,
    "wandb_id_unique": True,
}
target = pathlib.Path(result_path)
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
temporary.replace(target)
print(json.dumps(record, sort_keys=True))
PY
  test -s "$PREFLIGHT_RESULT" -a -s "$RUN_BUNDLE"
  chmod 444 "$RUN_BUNDLE"
  cp "$RUN_BUNDLE" "$PROVENANCE_DIR/"
  printf '[preflight] ALL_PASS arm=%s job=%s result=%s\n' \
    "$ARM" "$SLURM_JOB_ID" "$PREFLIGHT_RESULT"
  exit 0
fi

TRAIN_LOG=$RUN_DIR/training.log
"$SLURM_BIN/srun" --kill-on-bad-exit=1 --unbuffered \
  "$PY_ENV/bin/python" -m egomimic.trainHydra "${COMMON_OVERRIDES[@]}" \
  2>&1 | tee -a "$TRAIN_LOG"

assert_sha "$EXPECTED_LAUNCHER_SHA" "$LAUNCHER"
assert_sha "$EXPECTED_NORM_SHA" "$NORM_ARTIFACT"
assert_sha "$EXPECTED_SPLIT_SHA" "$SPLIT_MANIFEST"
assert_sha "$EXPECTED_RUN_BUNDLE_SHA" "$RUN_BUNDLE"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"

if test "$MODE" = smoke; then
  "$PY_ENV/bin/python" "$VERIFIER" "$RUN_DIR" \
    --required-embodiments 19,20 \
    --expected-head "$EXPECTED_HEAD" \
    --expected-world-size 2
  test -s "$RUN_DIR/SMOKE_RESULT.json"
  "$PY_ENV/bin/python" - \
    "$RUN_DIR/checkpoints/last.ckpt" "$RUN_BUNDLE" \
    "$RUN_DIR/SMOKE_RESULT.json" \
    "rl2-group/pushshapes-flow-transfer/$WANDB_ID" <<'PY'
import collections
import json
import math
import pathlib
import sys
import time

import wandb

from egomimic.pl_utils.pl_model import ModelWrapper

checkpoint_path, bundle_path, result_path = map(pathlib.Path, sys.argv[1:4])
wandb_path = sys.argv[4]
bundle = json.loads(bundle_path.read_text())
result = json.loads(result_path.read_text())
assert result["global_step"] == 2020

counts = collections.Counter()
for row in result["training_history"]:
    metrics = row["train_metrics"]
    optimizer_step = int(metrics["Train/19_log_optimizer_step"])
    if optimizer_step >= 2001:
        assert metrics["Train/19_log_sampler_unroll_steps"] == metrics[
            "Train/20_log_sampler_unroll_steps"
        ]
        counts[int(metrics["Train/19_log_sampler_unroll_steps"])] += 1
expected_counts = {2: 1, 4: 2, 8: 2, 16: 2, 32: 3, 64: 10}
assert dict(counts) == expected_counts, counts

model = ModelWrapper.load_from_checkpoint(
    str(checkpoint_path.resolve(strict=True)),
    map_location="cpu",
    strict=True,
    weights_only=False,
)
total = sum(parameter.numel() for parameter in model.parameters())
assert total == bundle["model_contract"]["total_parameters"] == 45_218_674
del model

required = bundle["logging_contract"]["required_metrics"]
visible = None
for _ in range(12):
    run = wandb.Api(timeout=30).run(wandb_path)
    summary = dict(run.summary)
    if all(key in summary and math.isfinite(float(summary[key])) for key in required):
        visible = {key: float(summary[key]) for key in required}
        break
    time.sleep(5)
assert visible is not None

result["strict_reload"] = True
result["parameter_count"] = total
result["wandb_api_visible"] = True
result["required_exact_metrics"] = True
result["wandb_path"] = wandb_path
result["wandb_api_metrics"] = visible
result["observed_post_warmup_unroll_counts"] = {
    str(key): value for key, value in expected_counts.items()
}
temporary = result_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
temporary.replace(result_path)
PY
  printf '[smoke] ALL_PASS arm=%s job=%s result=%s\n' \
    "$ARM" "$SLURM_JOB_ID" "$RUN_DIR/SMOKE_RESULT.json"
else
  test -s "$RUN_DIR/checkpoints/last.ckpt"
  printf '[full] ALL_PASS arm=%s job=%s wandb=%s\n' \
    "$ARM" "$SLURM_JOB_ID" "https://wandb.ai/rl2-group/pushshapes-flow-transfer/runs/$WANDB_ID"
fi
