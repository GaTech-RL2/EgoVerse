#!/usr/bin/env bash
# Discarded pipeline check: production keeps its separate full-mixture recipe.
set -euo pipefail
: "${PEAR_SOURCE:?}" "${WORK_DIR:?}" "${RUN:?}" "${MODEL:?}" "${ADAPTER_ROOT:?}"
cd "$PEAR_SOURCE"
GPU=h100
MAX_MODEL_LEN=16384
SESSIONS_PER_REPLICA=8
source examples/models/qwen3_8_27b_pear/qwen3_8_27b.sh
VERL_ENV=$WORK_DIR/envs/verl
source scripts/gpu_process_variables.sh "$VERL_ENV"
export VERL_USE_EXTERNAL_MODULES=pear.rollout.server,$MODEL_MODULES
ENV_VAR=+ray_kwargs.ray_init.runtime_env.env_vars
"$VERL_ENV/bin/python" -m efe_pear_onpolicy_check \
  "${MODEL_ARGS[@]}" "${MODEL_TRAINING_ARGS[@]}" \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.actor.fsdp_config.use_gpu_backend=false \
  actor_rollout_ref.actor.fsdp_config.kernels=efe_pear_moe_kernels.FrozenRouterTrainingKernels \
  +actor_rollout_ref.actor.optim.override_optimizer_config.foreach=false \
  actor_rollout_ref.actor.calculate_entropy=true \
  actor_rollout_ref.rollout.kernels=null \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.standalone_gpu_memory_utilization=0.8 \
  '+actor_rollout_ref.rollout.custom.lmcache={enable: false}' \
  "data.train_files=[\"$RUN/train.json\"]" "data.val_files=[\"$RUN/development.json\"]" \
  data.max_prompt_length=8192 data.max_response_length=8192 data.train_batch_size=8 \
  actor_rollout_ref.rollout.nnodes=1 actor_rollout_ref.rollout.n_gpus_per_node=8 \
  actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.max_model_len=16384 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=6 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
  "actor_rollout_ref.rollout.multi_turn.tool_config_path=$RUN/tools.yaml" \
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=$ADAPTER_ROOT/efe_pear_agent_metrics.yaml" \
  +actor_rollout_ref.rollout.limit_images=64 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  trainer.n_gpus_per_node=8 trainer.nnodes=1 \
  trainer.total_training_steps=1 trainer.total_epochs=10000 \
  trainer.experiment_name=rp_jhh_discarded_moe_sc_qualification \
  trainer.default_local_dir=/dev/shm/rp_jhh_discarded_sc_checkpoints \
  trainer.resume_mode=disable trainer.validation_data_dir="$RUN/eval" \
  'trainer.logger=["console","file"]' trainer.rollout_data_dir="$RUN/rollout-records" \
  run_dir="$RUN" run_sizes.sessions_per_replica=8 \
  transfer_queue.backend.SimpleStorage.num_data_storage_units=8 \
  "$ENV_VAR.VERL_USE_EXTERNAL_MODULES='$VERL_USE_EXTERNAL_MODULES'" \
  "$ENV_VAR.PYTHONPATH='$PYTHONPATH'" \
  "$ENV_VAR.EFE_PEAR_REASONING_MEMBERSHIP=$EFE_PEAR_REASONING_MEMBERSHIP" \
  "$ENV_VAR.EFE_PEAR_METRICS_DIR=$EFE_PEAR_METRICS_DIR" \
  "$ENV_VAR.EFE_PEAR_QUALIFICATION_DIR=$EFE_PEAR_QUALIFICATION_DIR" \
  "$ENV_VAR.EFE_PEAR_OPTIMIZER_CHECK_DIR=$EFE_PEAR_OPTIMIZER_CHECK_DIR" \
  "$ENV_VAR.PEAR_PRIMARY_TRAINING='0'" \
  "$ENV_VAR.VERL_FILE_LOGGER_PATH=$RUN/metrics.jsonl" \
  "$ENV_VAR.NCCL_ALGO=Ring" "$ENV_VAR.NCCL_PROTO=Simple" "$ENV_VAR.NCCL_P2P_DISABLE='1'" \
  "$@"
