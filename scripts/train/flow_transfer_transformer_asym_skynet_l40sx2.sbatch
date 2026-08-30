#!/bin/bash
# Smoke/full launcher for the asymmetric Flow Transfer Transformer decoders.
# Submit on one Skynet node with two tasks and two L40S GPUs.

set -Eeuo pipefail

report_error() {
  local status=$?
  local line=${BASH_LINENO[0]:-$LINENO}
  printf '[%s] FAIL arm=%s line=%s status=%s command=%q\n' \
    "${MODE:-unset}" "${ARM:-unset}" "$line" "$status" "$BASH_COMMAND" >&2
  exit "$status"
}
trap report_error ERR

MODE=${MODE:?set MODE=smoke or MODE=full}
ARM=${ARM:?set ARM=r01_l4_asym or r02_l8_asym}
EXPECTED_LAUNCHER_SHA=${EXPECTED_LAUNCHER_SHA:?set launcher SHA256}
EXPECTED_SPLIT_SHA=${EXPECTED_SPLIT_SHA:?set split manifest SHA256}
SMOKE_GATE=${SMOKE_GATE:-}

REPO=/coc/flash7/paphiwetsa3/experiments/flow_transfer_transformer3_cond128_decoder_val01_l40sx2_20260830/source_2032a578_transformer3c128
PY_ENV=/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv
EXP_ROOT=/coc/flash7/paphiwetsa3/experiments/flow_transfer_transformer3_cond128_decoder_val01_l40sx2_20260830
LAUNCHER=$REPO/scripts/train/flow_transfer_transformer_asym_skynet_l40sx2.sbatch
VERIFIER=$REPO/egomimic/scripts/verify_training_smoke.py
DATA_FILE=$REPO/egomimic/hydra_configs/data/pusht/flow_transfer_usocket_chain_newdata_val01_h16_per_emb_proprio.yaml
NORM_ARTIFACT=/coc/flash7/paphiwetsa3/experiments/flow_transfer_temporal_h8_l8_world2_smokes_20260829/norm_artifacts/per_emb_proprio_h16/fast_merge_20260829/artifact/norm_stats/norm_stats.json
SPLIT_MANIFEST=$EXP_ROOT/source_artifacts/splits/seed42_val01_skynet.json
U_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/u_socket_3000_v2_clean
CHAIN_BASE_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_3000_v2
CHAIN_GEN_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_gen_flow_transfer_frozen719_20260829
EXCLUDED_CHAIN_EPISODE=episode_T_chain_gripper_obs7_000050.zarr
BASE_REVISION=2032a5781100a0c67b1da8cd1f094cefd586decd
EXPECTED_NORM_SHA=d6abc08a3315cd172bd6c03ded0346ae24d69f2ce1744e133fe1768f0feefa96
EXPECTED_STAGE_SHA=62c2fd766a9dd3a4d17c22ab6135eecd576adea9446a67cfcc21f28443ddadb8
EXPECTED_TEST_SHA=9cd3122b011f45c6ce264da7c83387e7d47331b9e7baefe924360a914fde1dbe
EXPECTED_DATA_SHA=93297962d4343f6b085ea929c4fce2e2f03a8101ddd183d42563ea96ed9c3d9f
EXPECTED_R01_MODEL_SHA=5fb544e6dfbe5cc79019b78d81332c32f4f6a8f0badd8ba52121ab52370e92a4
EXPECTED_R01_EXPERIMENT_SHA=a511717174d555bcf585ed8d06b3a9044ff38dec680f91f2f8a2d7eb250129dc
EXPECTED_R02_MODEL_SHA=32541c82cf6929baf980151fbf1e2350982268345d2e6ce53aa524fc63060010
EXPECTED_R02_EXPERIMENT_SHA=08a4c88bc783c1095e54a4f502b27d608536625cb58ee46fed0fbf9bf65893b8
WANDB_GROUP=flow_transfer_transformer3_asym_decoder_val01_20260830
EXPECTED_WORLD_SIZE=2
EXPECTED_SLURM_ACCOUNT=rl2-lab
EXPECTED_SLURM_PARTITION=rl2-lab
EXPECTED_GPU_MODEL=L40S
SLURM_BIN=/opt/slurm/Ubuntu-20.04/24.11.0/bin
export PATH=$SLURM_BIN:$PATH

case "$ARM" in
  r01_l4_asym)
    EXPERIMENT=pusht/pipeline_sampler_usocket_chain_newdata_r01_l4_transformer_val01_h16
    MODEL_FILE=$REPO/egomimic/hydra_configs/model/bf/bf_pipeline_sampler_usocket_chain_points_r01_l4_transformer_per_emb_proprio_h16.yaml
    EXPERIMENT_FILE=$REPO/egomimic/hydra_configs/experiment/pusht/pipeline_sampler_usocket_chain_newdata_r01_l4_transformer_val01_h16.yaml
    EXPECTED_MODEL_SHA=$EXPECTED_R01_MODEL_SHA
    EXPECTED_EXPERIMENT_SHA=$EXPECTED_R01_EXPERIMENT_SHA
    EXPECTED_LATENT_DIM=4
    EXPECTED_DENOISER_WIDTH=128
    EXPECTED_USOCKET_DECODER_WIDTH=16
    EXPECTED_CHAIN_DECODER_WIDTH=64
    BASE_WANDB_ID=ft_r01_l4_w128_d8_c128_transformer3_decu16_cg64_s42_l40sx2_20260830
    ;;
  r02_l8_asym)
    EXPERIMENT=pusht/pipeline_sampler_usocket_chain_newdata_r02_transformer_val01_h16
    MODEL_FILE=$REPO/egomimic/hydra_configs/model/bf/bf_pipeline_sampler_usocket_chain_points_r02_transformer_per_emb_proprio_h16.yaml
    EXPERIMENT_FILE=$REPO/egomimic/hydra_configs/experiment/pusht/pipeline_sampler_usocket_chain_newdata_r02_transformer_val01_h16.yaml
    EXPECTED_MODEL_SHA=$EXPECTED_R02_MODEL_SHA
    EXPECTED_EXPERIMENT_SHA=$EXPECTED_R02_EXPERIMENT_SHA
    EXPECTED_LATENT_DIM=8
    EXPECTED_DENOISER_WIDTH=256
    EXPECTED_USOCKET_DECODER_WIDTH=32
    EXPECTED_CHAIN_DECODER_WIDTH=64
    BASE_WANDB_ID=ft_r02_l8_w256_d8_c128_transformer3_decu32_cg64_s42_l40sx2_20260830
    ;;
  *)
    printf 'Unknown ARM=%s\n' "$ARM" >&2
    exit 64
    ;;
esac

case "$MODE" in
  smoke)
    test -z "$SMOKE_GATE"
    RUN_DIR=$EXP_ROOT/smokes/$ARM/job_${SLURM_JOB_ID:?}
    WANDB_ID=${BASE_WANDB_ID}_smoke_job_${SLURM_JOB_ID}
    WANDB_OFFLINE=true
    ;;
  full)
    SMOKE_GATE=${SMOKE_GATE:?full mode requires SMOKE_GATE}
    test -s "$SMOKE_GATE"
    "$PY_ENV/bin/python" - "$SMOKE_GATE" "$BASE_REVISION" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert payload["status"] == "passed"
assert payload["repo_head"] == sys.argv[2]
assert payload["world_size"] == 2
assert payload["global_step"] == 2
assert payload["validation_trainer_global_step"] >= 1
assert payload["required_embodiments"] == [19, 20]
PY
    RUN_DIR=$EXP_ROOT/runs/$ARM
    test ! -e "$RUN_DIR"
    WANDB_ID=$BASE_WANDB_ID
    WANDB_OFFLINE=false
    ;;
  *)
    printf 'Unknown MODE=%s\n' "$MODE" >&2
    exit 64
    ;;
esac

PROVENANCE_DIR=$EXP_ROOT/provenance/$MODE/$ARM/job_${SLURM_JOB_ID:?}
test ! -e "$RUN_DIR"
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
assert_sha "$EXPECTED_DATA_SHA" "$DATA_FILE"
assert_sha "$EXPECTED_MODEL_SHA" "$MODEL_FILE"
assert_sha "$EXPECTED_EXPERIMENT_SHA" "$EXPERIMENT_FILE"
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

test "${SLURM_RESTART_COUNT:-0}" = 0
test "${SLURM_JOB_ACCOUNT:?}" = "$EXPECTED_SLURM_ACCOUNT"
test "${SLURM_JOB_PARTITION:?}" = "$EXPECTED_SLURM_PARTITION"
test "${SLURM_NTASKS:?}" = "$EXPECTED_WORLD_SIZE"
test "${SLURM_JOB_NUM_NODES:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" -ge 8
GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader)
test "$(grep -Fci "$EXPECTED_GPU_MODEL" <<< "$GPU_NAMES")" = "$EXPECTED_WORLD_SIZE"

nvidia-smi -q > "$PROVENANCE_DIR/nvidia_smi_q.txt"
cp "$LAUNCHER" "$MODEL_FILE" "$EXPERIMENT_FILE" "$DATA_FILE" \
  "$SPLIT_MANIFEST" "$NORM_ARTIFACT" "$PROVENANCE_DIR/"
printf '%s\n' "$BASE_REVISION" > "$PROVENANCE_DIR/base_revision.txt"
printf '%s\n' "$GPU_NAMES" > "$PROVENANCE_DIR/gpu_names.txt"
"$PY_ENV/bin/python" --version > "$PROVENANCE_DIR/python_version.txt" 2>&1
"$SLURM_BIN/scontrol" show job -dd "$SLURM_JOB_ID" > "$PROVENANCE_DIR/slurm_job.txt"

cd "$REPO"
export PYTHONPATH=$REPO
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true
if test "$WANDB_OFFLINE" = true; then
  export WANDB_MODE=offline
else
  export WANDB_MODE=online
fi

COMMON_OVERRIDES=(
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
  logger.wandb.resume=never
  "logger.wandb.tags=[flow-transfer,transformer-action-decoder,transformer-decoder-3blocks,asymmetric-decoder-width,chain-decoder-64,condition-dim-128,$ARM,latent-dense,h16,j8,val01-seed42,fresh-start,batch64-per-embodiment,checkpoint-20k,skynet,l40sx2,world2,chain-gen-frozen719]"
)

if test "$MODE" = smoke; then
  COMMON_OVERRIDES+=(
    trainer.max_steps=2
    trainer.limit_train_batches=2
    trainer.val_check_interval=1
    trainer.limit_val_batches=1
    trainer.num_sanity_val_steps=0
    callbacks.model_checkpoint.monitor=null
    callbacks.model_checkpoint.save_top_k=-1
    callbacks.model_checkpoint.save_last=true
    callbacks.model_checkpoint.every_n_epochs=null
    callbacks.model_checkpoint.every_n_train_steps=2
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

assert cfg.seed == 42 and cfg.mode == "train" and cfg.ckpt_path is None
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
assert cfg.logger.wandb.resume == "never"

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
assert checkpoint.save_last is True
assert checkpoint.save_top_k == -1
assert checkpoint.every_n_epochs is None
assert checkpoint.train_time_interval is None
assert checkpoint.save_on_train_epoch_end is False
if mode == "smoke":
    assert cfg.trainer.max_steps == 2
    assert cfg.trainer.limit_train_batches == 2
    assert cfg.trainer.val_check_interval == 1
    assert cfg.trainer.limit_val_batches == 1
    assert checkpoint.every_n_train_steps == 2
else:
    assert mode == "full"
    assert cfg.trainer.max_steps == 240_000
    assert cfg.trainer.limit_train_batches == 1.0
    assert cfg.trainer.val_check_interval == 10_000
    assert cfg.trainer.limit_val_batches == 8
    assert checkpoint.every_n_train_steps == 20_000

assert "arc_length" not in pathlib.Path(config_path).read_text().lower()
PY

sha256sum "$RESOLVED_CONFIG" > "$PROVENANCE_DIR/resolved_config.sha256"
TRAIN_LOG=$RUN_DIR/training.log
"$SLURM_BIN/srun" --kill-on-bad-exit=1 --unbuffered \
  "$PY_ENV/bin/python" -m egomimic.trainHydra "${COMMON_OVERRIDES[@]}" \
  2>&1 | tee -a "$TRAIN_LOG"

assert_sha "$EXPECTED_LAUNCHER_SHA" "$LAUNCHER"
assert_sha "$EXPECTED_NORM_SHA" "$NORM_ARTIFACT"
assert_sha "$EXPECTED_SPLIT_SHA" "$SPLIT_MANIFEST"

if test "$MODE" = smoke; then
  "$PY_ENV/bin/python" "$VERIFIER" "$RUN_DIR" \
    --required-embodiments 19,20 \
    --expected-head "$BASE_REVISION" \
    --expected-world-size 2
  test -s "$RUN_DIR/SMOKE_RESULT.json"
  printf '[smoke] ALL_PASS arm=%s job=%s result=%s\n' \
    "$ARM" "$SLURM_JOB_ID" "$RUN_DIR/SMOKE_RESULT.json"
else
  test -s "$RUN_DIR/checkpoints/last.ckpt"
  printf '[full] ALL_PASS arm=%s job=%s wandb=%s\n' \
    "$ARM" "$SLURM_JOB_ID" "https://wandb.ai/rl2-group/pushshapes-flow-transfer/runs/$WANDB_ID"
fi
