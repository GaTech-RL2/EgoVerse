#!/bin/bash
# Resource requests are supplied by sbatch because BC uses one L40S and
# obstacle cotrain uses two. This file is shared by smoke and full jobs.

set -Eeuo pipefail

ARM=${ARM:?submit with ARM}
MODE=${MODE:?submit with MODE=smoke or MODE=full}
GPUS_EXPECTED=${GPUS_EXPECTED:?submit with GPUS_EXPECTED}
EXPECTED_HEAD=${EXPECTED_HEAD:?submit with EXPECTED_HEAD}
EXPECTED_LAUNCHER_SHA=${EXPECTED_LAUNCHER_SHA:?submit with EXPECTED_LAUNCHER_SHA}

REPO=/coc/flash7/paphiwetsa3/worktrees/flow-transfer-direct-dense-obstacle-20260826
PY_ENV=/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv
EXP_ROOT=/coc/flash7/paphiwetsa3/experiments/flow_transfer_direct_dense_obstacle_20260826
LAUNCHER=$REPO/scripts/train/flow_transfer_direct_dense_matrix.sbatch
U_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/u_socket_3000_v2_clean
CHAIN_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_3000_v2
export CHAIN_OBSTACLE_ROOT=/coc/flash7/paphiwetsa3/experiments/chain_gripper_obstacle_mimicgen_128_20260826/output_128
OBSTACLE_AUDIT=$CHAIN_OBSTACLE_ROOT/audit_report.json
EXPECTED_OBSTACLE_AUDIT_SHA=a06ce0def2ec62f5ea1491dc0efc7ab451bfd9e4eaaf16b2f7f84537c394f2bb
EXPECTED_U_COUNT=2999
EXPECTED_U_SHA=fb6e107b585d6a5fae9e9ea4186f441b625a680a36d5691469d9c4125cd9a0fa
EXPECTED_CHAIN_COUNT=3000
EXPECTED_CHAIN_SHA=3ca50bf2558a3afbe11784a0b3d6424d3d3d9b1d9a564470618d1b64e2e99ede
WANDB_GROUP=flow_transfer_direct_dense_obstacle_20260826
SLURM_BIN=/opt/slurm/Ubuntu-20.04/24.11.0/bin
export PATH=$SLURM_BIN:$PATH

case "$ARM" in
  bc_usocket_latent)
    EXPERIMENT=pusht/pipeline_sampler_usocket_dense_medium
    BASE_WANDB_ID=ft_bc_usocket_latent_dense_m96_s42_20260826
    ;;
  bc_usocket_dp)
    EXPERIMENT=pusht/pipeline_diffusion_usocket_h16
    BASE_WANDB_ID=ft_bc_usocket_dp_s42_20260826
    ;;
  bc_chain_latent)
    EXPERIMENT=pusht/pipeline_sampler_chain_gripper_points_dense_medium
    BASE_WANDB_ID=ft_bc_chain_latent_dense_m96_s42_20260826
    ;;
  bc_chain_dp)
    EXPERIMENT=pusht/pipeline_diffusion_chain_gripper_points_h16
    BASE_WANDB_ID=ft_bc_chain_dp_s42_20260826
    ;;
  cotrain_obstacle_latent)
    EXPERIMENT=pusht/pipeline_sampler_usocket_chain_obstacle_dense_medium
    BASE_WANDB_ID=ft_cotrain_obstacle_latent_dense_m96_s42_20260826
    ;;
  cotrain_obstacle_dp)
    EXPERIMENT=pusht/pipeline_diffusion_usocket_chain_obstacle_h16
    BASE_WANDB_ID=ft_cotrain_obstacle_dp_s42_20260826
    ;;
  *)
    printf 'Unknown ARM=%s\n' "$ARM" >&2
    exit 64
    ;;
esac

case "$MODE" in
  smoke)
    WANDB_ID=${BASE_WANDB_ID}_smoke
    RUN_DIR=$EXP_ROOT/smokes/$ARM/job_${SLURM_JOB_ID}
    ;;
  full)
    WANDB_ID=$BASE_WANDB_ID
    RUN_DIR=$EXP_ROOT/runs/${ARM}_seed42
    ;;
  *)
    printf 'Unknown MODE=%s\n' "$MODE" >&2
    exit 64
    ;;
esac
PROVENANCE_DIR=$EXP_ROOT/provenance/$MODE/$ARM/job_${SLURM_JOB_ID}

test -x "$PY_ENV/bin/python"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$EXPECTED_LAUNCHER_SHA"
test "$(sha256sum "$OBSTACLE_AUDIT" | awk '{print $1}')" = "$EXPECTED_OBSTACLE_AUDIT_SHA"

mkdir -p "$PROVENANCE_DIR" "$EXP_ROOT/runs" "$EXP_ROOT/smokes" "$(dirname "$RUN_DIR")"
if test "$MODE" = full; then
  if test "${SLURM_RESTART_COUNT:-0}" = 0; then
    test ! -e "$RUN_DIR"
    mkdir "$RUN_DIR"
    printf '%s\n' "$SLURM_JOB_ID" > "$RUN_DIR/.owner_job_id"
  else
    test "$(sed -n '1p' "$RUN_DIR/.owner_job_id")" = "$SLURM_JOB_ID"
  fi
else
  mkdir "$RUN_DIR"
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
inventory "$CHAIN_DATA" "$EXPECTED_CHAIN_COUNT" "$EXPECTED_CHAIN_SHA" "$PROVENANCE_DIR/chain_clean_inventory.txt"

: > "$PROVENANCE_DIR/chain_obstacle_inventory.txt"
for level_number in $(seq 1 30); do
  level=$(printf '%02d' "$level_number")
  level_root=$CHAIN_OBSTACLE_ROOT/level_${level}/chain_gripper/T
  count=$(find "$level_root" -mindepth 1 -maxdepth 1 -type d -name '*.zarr' | wc -l)
  test "$count" = 128
  find "$level_root" -mindepth 1 -maxdepth 1 -type d -name '*.zarr' -printf "level_${level}/%f\n" | LC_ALL=C sort >> "$PROVENANCE_DIR/chain_obstacle_inventory.txt"
done
test "$(wc -l < "$PROVENANCE_DIR/chain_obstacle_inventory.txt")" = 3840

cp "$OBSTACLE_AUDIT" "$PROVENANCE_DIR/obstacle_audit_report.json"
cp "$LAUNCHER" "$PROVENANCE_DIR/launcher.sbatch"
git -C "$REPO" log -1 --format=fuller > "$PROVENANCE_DIR/git_commit.txt"
git -C "$REPO" status --porcelain=v1 --untracked-files=all > "$PROVENANCE_DIR/git_status.txt"
$SLURM_BIN/scontrol show job -dd "$SLURM_JOB_ID" > "$PROVENANCE_DIR/slurm_job.txt"
test -n "${CUDA_VISIBLE_DEVICES:-}"
IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
test "${#visible_gpus[@]}" = "$GPUS_EXPECTED"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader > "$PROVENANCE_DIR/gpu.txt"
test "$(wc -l < "$PROVENANCE_DIR/gpu.txt")" = "$GPUS_EXPECTED"
test "$(grep -ci 'L40S' "$PROVENANCE_DIR/gpu.txt")" = "$GPUS_EXPECTED"

RUNTIME_ENV=$PROVENANCE_DIR/runtime_env
mkdir "$RUNTIME_ENV"
ln -s "$PY_ENV" "$RUNTIME_ENV/emimic"
cd "$RUNTIME_ENV"
source emimic/bin/activate
cd "$REPO"
export PYTHONPATH=$REPO
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true

COMMON_OVERRIDES=(
  --config-name=train_zarr_cartesian
  "+experiment=$EXPERIMENT"
  "hydra.run.dir=$RUN_DIR"
  "launch_params.gpus_per_node=$GPUS_EXPECTED"
  launch_params.nodes=1
  logger.wandb.project=pushshapes-flow-transfer
  logger.wandb.entity=rl2-group
  "logger.wandb.group=$WANDB_GROUP"
  "logger.wandb.id=$WANDB_ID"
  "++logger.wandb.name=$WANDB_ID"
  ++logger.wandb.resume=allow
)

if test "$MODE" = smoke; then
  COMMON_OVERRIDES+=(
    trainer.max_steps=2
    trainer.limit_train_batches=2
    trainer.val_check_interval=1
    trainer.limit_val_batches=1
    norm_stats.sample_frac=0.002
    callbacks.model_checkpoint.every_n_train_steps=1
    callbacks.model_checkpoint.train_time_interval=null
    callbacks.model_checkpoint.save_on_train_epoch_end=false
  )
else
  LAST_CKPT=$RUN_DIR/checkpoints/last.ckpt
  if test -s "$LAST_CKPT"; then
    COMMON_OVERRIDES+=("ckpt_path=$LAST_CKPT")
  fi
fi

RESOLVED_CONFIG=$PROVENANCE_DIR/resolved_config.yaml
python -m egomimic.trainHydra "${COMMON_OVERRIDES[@]}" --cfg job > "$RESOLVED_CONFIG"
python - "$RESOLVED_CONFIG" "$ARM" "$GPUS_EXPECTED" "$WANDB_ID" <<'PY'
import pathlib
import sys

from omegaconf import OmegaConf

path, arm, expected_gpus, wandb_id = sys.argv[1:]
cfg = OmegaConf.load(path)
model = cfg.model.robomimic_model
expected_gpus = int(expected_gpus)

assert cfg.seed == 42
assert cfg.trainer.precision == "bf16"
assert cfg.launch_params.gpus_per_node == expected_gpus
assert cfg.launch_params.nodes == 1
assert cfg.logger.wandb.project == "pushshapes-flow-transfer"
assert cfg.logger.wandb.id == wandb_id
assert cfg.norm_stats.reduce_all_but_last is True
assert cfg.model.optimizer.lr == 1.0e-4
assert cfg.model.optimizer.weight_decay == 1.0e-4
assert model._target_ == "egomimic.pipeline.algo.PipelineAlgo"

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
assert list(model.domains) == expected_domains
assert set(cfg.data.train_datasets) == set(expected_domains)

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

if cotrain:
    chain = cfg.data.train_datasets.pushshapes_sim_chain_gripper.resolver
    assert chain._target_.endswith("LocalEpisodeResolverManyWithEmbodimentOverride")
    assert len(chain.folder_paths) == 31
    assert str(chain.folder_paths[-1]).endswith("level_30/chain_gripper/T")
PY

python - "$OBSTACLE_AUDIT" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert report["status"] == "PASS"
assert report["levels"] == list(range(1, 31))
assert sum(item["total"] for item in report["reports"]) == 3840
assert sum(item["manual"] for item in report["reports"]) == 480
assert sum(item["generated"] for item in report["reports"]) == 3360
PY

python -m egomimic.trainHydra "${COMMON_OVERRIDES[@]}"
