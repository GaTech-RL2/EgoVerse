#!/usr/bin/env bash
# PEAR++ driver for either student. It runs only with frozen native admission
# and model/restart qualification receipts; runtime installation is insufficient.
set -euo pipefail
: "${PEAR_SOURCE:?}" "${WORK_DIR:?}" "${RUN:?}" "${MODEL:?}" "${MODEL_REVISION:?}"
: "${EPISODES:?}" "${VALIDATION_EPISODES:?}" "${EFE_PEAR_NATIVE_CONFIG:?}"
: "${MODEL_QUALIFICATION:?}" "${TRAINING_STEPS:?}" "${EFE_PEAR_CURRICULUM:?}" "${TOOL_CONFIG:?}"
GPU=h100
TRAINER_NNODES=${TRAINER_NNODES:-1}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-1}
VERL_ENV=$WORK_DIR/envs/verl
MAX_PROMPT=4096
MAX_RESPONSE=${MAX_RESPONSE:-160000}
MAX_MODEL_LEN=$((MAX_PROMPT + MAX_RESPONSE))
BATCH=128
N=8
SESSIONS_PER_REPLICA=${SESSIONS_PER_REPLICA:-16}
ADAPTER_ROOT=$(cd "$(dirname "$0")" && pwd)
export PEAR_PRIMARY_TRAINING=1
export EFE_PEAR_NATIVE_CONFIG MODEL_REVISION MODEL_QUALIFICATION EFE_PEAR_CURRICULUM TOOL_CONFIG
export PYTHONPATH=$ADAPTER_ROOT${PYTHONPATH:+:$PYTHONPATH}
"$VERL_ENV/bin/python" - <<'PY'
import json, os
from pathlib import Path
import yaml
from efe_pear_curriculum_state import CurriculumState
q=json.loads(Path(os.environ['MODEL_QUALIFICATION']).read_text())
assert q['model_revision']==os.environ['MODEL_REVISION']
assert q['parameter_update']=='full_finetuning' and q['lora_rank']==0
assert q['full_model_replay_max_error'] <= 0.0005
assert q['optimizer_updates']==1 and q['checkpoint_reload_passed'] is True
assert q['primary_training_qualified'] is True and q['on_policy_score_centering_validated'] is True
config=json.loads(Path(os.environ['EFE_PEAR_NATIVE_CONFIG']).read_text())
assert not config.get('qualification_only')
membership=json.loads(Path(config['membership_file']).read_text())
assert membership['training_tasks']
assert all(v['qualification']=='passed' for v in membership['training_tasks'].values())
curriculum=CurriculumState(json.loads(Path(os.environ['EFE_PEAR_CURRICULUM']).read_text()))
tools=yaml.safe_load(Path(os.environ['TOOL_CONFIG']).read_text())['tools']
configured={t['config']['env_class'] for t in tools}
assert {c['env_class'] for c in curriculum.configs.values()} <= configured
PY

cd "$PEAR_SOURCE"
# Preserve our initialization revision: the sourced example assigns its own.
INITIAL_MODEL_REVISION=$MODEL_REVISION
source examples/models/qwen3_8_27b_pear/qwen3_8_27b.sh
MODEL_REVISION=$INITIAL_MODEL_REVISION
MODEL_TYPE=$("$VERL_ENV/bin/python" -c 'import json,pathlib,sys; print(json.loads((pathlib.Path(sys.argv[1])/"config.json").read_text())["model_type"])' "$MODEL")
# PEAR's fused Hopper implementation is dense-only. Its portable training
# kernels and stock vLLM support the MoE; that path needs its own qualification.
ARCHITECTURE_ARGS=()
case "$MODEL_TYPE" in
  qwen3_5) ;;
  qwen3_5_moe)
    ARCHITECTURE_ARGS=(actor_rollout_ref.actor.fsdp_config.use_gpu_backend=false
                       +actor_rollout_ref.actor.optim.override_optimizer_config.foreach=false
                       actor_rollout_ref.rollout.kernels=null)
    ;;
  *) echo "Unsupported primary model architecture: $MODEL_TYPE" >&2; exit 1 ;;
esac
# Override host-memory sizing, retaining the paper's optimizer batch and kernels.
LMCACHE_ARGS=("+actor_rollout_ref.rollout.custom.lmcache={enable: true, log_dir: $RUN/lmcache, l1_size_gb: 96}"
              +actor_rollout_ref.rollout.custom.lmcache.chunk_size=784)
MODEL_ROLLOUT_ARGS=(actor_rollout_ref.rollout.tensor_model_parallel_size="$TP"
  "${TRTLLM_ARGS[@]}" "${LMCACHE_ARGS[@]}" actor_rollout_ref.rollout.standalone_gpu_memory_utilization=0.85)
mkdir -p "$RUN"
export PEAR_LOG_DIR=$RUN/sim
export NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_P2P_DISABLE=1
export LOCAL_DIR=${LOCAL_DIR:-/tmp/efe-pear-local}
source scripts/gpu_process_variables.sh "$VERL_ENV"
export RAY_ADDRESS=${RAY_ADDRESS:-auto}
export VERL_USE_EXTERNAL_MODULES=pear.rollout.server,$MODEL_MODULES
export PYTHONPATH=$ADAPTER_ROOT:$PEAR_SOURCE/pear/utils/sitecustomize:$PEAR_SOURCE:$PEAR_SOURCE/3rdparty/verl/src${PYTHONPATH:+:$PYTHONPATH}
ENV_VAR=+ray_kwargs.ray_init.runtime_env.env_vars
bash 3rdparty/verl/scripts/checkout_verl.sh --check
# EpisodeDataset accepts multiple files, one per native or reasoning tool.
TRAIN_FILES=$("$VERL_ENV/bin/python" -c 'import json,sys; v=sys.argv[1]; print(json.dumps(json.loads(v) if v.startswith("[") else [v]))' "$EPISODES")
VAL_FILES=$("$VERL_ENV/bin/python" -c 'import json,sys; v=sys.argv[1]; print(json.dumps(json.loads(v) if v.startswith("[") else [v]))' "$VALIDATION_EPISODES")
"$VERL_ENV/bin/python" -m efe_pear_multitask_train \
  "${MODEL_ARGS[@]}" "${MODEL_ROLLOUT_ARGS[@]}" "${MODEL_TRAINING_ARGS[@]}" \
  "${ARCHITECTURE_ARGS[@]}" \
  actor_rollout_ref.model.lora_rank=0 \
  "data.train_files=$TRAIN_FILES" "data.val_files=$VAL_FILES" \
  data.max_prompt_length="$MAX_PROMPT" data.max_response_length="$MAX_RESPONSE" data.train_batch_size="$BATCH" \
  actor_rollout_ref.rollout.nnodes="$ROLLOUT_NNODES" actor_rollout_ref.rollout.n_gpus_per_node=8 \
  actor_rollout_ref.rollout.n="$N" actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
  +actor_rollout_ref.rollout.limit_images="$((MAX_MODEL_LEN / 256))" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH" \
  trainer.n_gpus_per_node=8 trainer.nnodes="$TRAINER_NNODES" \
  trainer.total_training_steps="$TRAINING_STEPS" trainer.total_epochs=10000 \
  trainer.experiment_name="$(basename "$RUN")" trainer.default_local_dir="$RUN/ckpt" \
  trainer.validation_data_dir="$RUN/eval" trainer.logger='["console","file","tensorboard"]' \
  run_dir="$RUN" run_sizes.sessions_per_replica="$SESSIONS_PER_REPLICA" \
  "$ENV_VAR.VERL_USE_EXTERNAL_MODULES='$VERL_USE_EXTERNAL_MODULES'" \
  "$ENV_VAR.PYTHONPATH='$PYTHONPATH'" "$ENV_VAR.EFE_PEAR_NATIVE_CONFIG=$EFE_PEAR_NATIVE_CONFIG" \
  "$ENV_VAR.EFE_PEAR_CURRICULUM=$EFE_PEAR_CURRICULUM" \
  "$ENV_VAR.PEAR_PRIMARY_TRAINING=1" "$ENV_VAR.VERL_FILE_LOGGER_PATH=$RUN/metrics.jsonl" \
  "$ENV_VAR.NCCL_ALGO=Ring" "$ENV_VAR.NCCL_PROTO=Simple" "$ENV_VAR.NCCL_P2P_DISABLE=1" \
  "$@" 2>&1 | tee -a "$RUN/train.log"
