#!/bin/bash
# Real two-GPU optimizer + scheduled-validation smoke for clean Planar V2 DP cotraining.

set -Eeuo pipefail

EXPECTED_HEAD=${EXPECTED_HEAD:?set EXPECTED_HEAD}
EXPECTED_LAUNCHER_SHA=${EXPECTED_LAUNCHER_SHA:?set EXPECTED_LAUNCHER_SHA}
MODE=${MODE:-smoke}
REPO=/coc/flash7/paphiwetsa3/worktrees/planar-v2-dp-cotrain-clean-20260903
PY=/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/python
LAUNCHER=/coc/flash7/paphiwetsa3/scripts/train/planar_v2_dp_cotrain_clean_skynet_v1.sbatch
U_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/u_socket_3000_v2_clean
CHAIN_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_3000_v2
EXP_ROOT=/coc/flash7/paphiwetsa3/experiments/planar_v2_dp_cotrain_clean_20260903
WORLD_SIZE=2
PARAMETER_COUNT=51506981
EXPERIMENT=pusht/planar_v2_cotrain_clean_dp_standard
WANDB_ID=planar-v2-dp-cotrain-clean-smoke-${SLURM_JOB_ID}
SLURM_BIN=/opt/slurm/Ubuntu-20.04/24.11.0/bin
export PATH=$SLURM_BIN:$PATH

trap 'printf "[smoke] FAIL line=%s status=%s command=%q\n" "$LINENO" "$?" "$BASH_COMMAND" >&2' ERR
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$EXPECTED_LAUNCHER_SHA"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain)"
cd "$REPO"
export PYTHONPATH=$REPO HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1

if test "$MODE" = preflight; then
  PREFLIGHT_DIR=$(mktemp -d /tmp/planar-v2-dp-cotrain-clean-preflight.XXXXXX)
  "$PY" scripts/data/materialize_local_episode_split.py \
    --domain pushshapes_sim_u_socket="$U_DATA" \
    --domain pushshapes_sim_chain_gripper="$CHAIN_DATA" \
    --valid-ratio 0.01 --seed 42 --output "$PREFLIGHT_DIR/split.json"
  "$PY" -m egomimic.trainHydra --config-name=train_zarr_cartesian \
    "+experiment=$EXPERIMENT" "++paths.root_dir=$PREFLIGHT_DIR" \
    "paths.output_dir=$PREFLIGHT_DIR" "paths.work_dir=$REPO" \
    norm_stats.sample_frac=1.0 --cfg job --resolve > "$PREFLIGHT_DIR/config.yaml"
  "$PY" - "$PREFLIGHT_DIR/config.yaml" "$PREFLIGHT_DIR/split.json" "$PARAMETER_COUNT" <<'PY'
import json, pathlib, sys
from hydra.utils import instantiate
from omegaconf import OmegaConf
c = OmegaConf.load(sys.argv[1]); split = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert split["status"] == "PASS" and split["cross_domain_train_valid_resolved_path_overlap_count"] == 0
assert set(c.data.train_datasets) == {"pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper"}
assert c.run_provenance.obstacle_data is False and c.norm_stats.sample_frac == 1.0
m = instantiate(c.model.pipeline)
assert sum(p.numel() for p in m.nets.parameters()) == int(sys.argv[3])
print("[preflight] READY clean-only data, exact split, config, graph, and parameters")
PY
  exit 0
fi
test "$MODE" = smoke
RUN_DIR=$EXP_ROOT/smokes/job_${SLURM_JOB_ID:?}
PROVENANCE=$RUN_DIR/provenance
test "${SLURM_NTASKS:?}" = "$WORLD_SIZE"
test "${SLURM_JOB_NUM_NODES:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" -ge 8
test "${SLURM_RESTART_COUNT:-0}" = 0
test ! -e "$RUN_DIR"
mkdir -p "$PROVENANCE"

export WANDB_MODE=online WANDB_SILENT=true
GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader)
test "$(wc -l <<<"$GPU_NAMES")" = "$WORLD_SIZE"
printf '%s\n' "$GPU_NAMES" > "$PROVENANCE/gpu_names.txt"
scontrol show job -dd "$SLURM_JOB_ID" > "$PROVENANCE/slurm_job.txt"
cp "$LAUNCHER" scripts/train/verify_planar_training_smoke.py "$PROVENANCE/"

"$PY" scripts/data/materialize_local_episode_split.py \
  --domain pushshapes_sim_u_socket="$U_DATA" \
  --domain pushshapes_sim_chain_gripper="$CHAIN_DATA" \
  --valid-ratio 0.01 --seed 42 --output "$PROVENANCE/split_audit.json"
"$PY" - "$PROVENANCE/split_audit.json" <<'PY'
import json, pathlib, sys
p = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert p["status"] == "PASS"
assert p["cross_domain_train_valid_resolved_path_overlap_count"] == 0
expected = {
    "pushshapes_sim_u_socket": (2999, 2970, 29),
    "pushshapes_sim_chain_gripper": (3000, 2970, 30),
}
for name, counts in expected.items():
    d = p["domains"][name]
    assert (d["total_count"], d["train_count"], d["valid_count"]) == counts
    assert d["id_overlap_count"] == d["resolved_path_overlap_count"] == 0
    assert d["union_matches_inventory"] is True
PY

OVERRIDES=(
  --config-name=train_zarr_cartesian "+experiment=$EXPERIMENT"
  mode=train ckpt_path=null "hydra.run.dir=$RUN_DIR"
  "++paths.root_dir=$RUN_DIR" "paths.output_dir=$RUN_DIR" "paths.work_dir=$REPO"
  launch_params.gpus_per_node=2 launch_params.nodes=1
  trainer.devices=2 trainer.num_nodes=1 trainer.strategy=ddp trainer.precision=bf16
  trainer.max_steps=2 trainer.limit_train_batches=2 trainer.val_check_interval=1
  trainer.limit_val_batches=1 trainer.num_sanity_val_steps=0 trainer.log_every_n_steps=1
  norm_stats.sample_frac=1.0 norm_stats.save_cache_dir=null
  callbacks.model_checkpoint.monitor=null callbacks.model_checkpoint.save_top_k=-1
  callbacks.model_checkpoint.save_last=true callbacks.model_checkpoint.every_n_epochs=null
  callbacks.model_checkpoint.every_n_train_steps=1
  callbacks.model_checkpoint.train_time_interval=null callbacks.model_checkpoint.save_on_train_epoch_end=false
  "callbacks.model_checkpoint.filename='epoch-{epoch}-step-{step}'"
  logger.wandb.entity=rl2-group logger.wandb.project=pushshapes-planar-v2
  logger.wandb.group=planar-v2-dp-cotrain-clean-20260903
  "logger.wandb.id=$WANDB_ID" "logger.wandb.name=$WANDB_ID" logger.wandb.resume=never
  "logger.wandb.tags=[standard-dp,cotrain,clean-only,no-obstacle,planar-v2,h16,ddim100,smoke,skynet,world2]"
)

"$PY" -m egomimic.trainHydra "${OVERRIDES[@]}" --cfg job --resolve > "$PROVENANCE/resolved_config.yaml"
"$PY" - "$PROVENANCE/resolved_config.yaml" "$PARAMETER_COUNT" <<'PY'
import pathlib, sys
from hydra.utils import instantiate
from omegaconf import OmegaConf
c = OmegaConf.load(sys.argv[1])
assert set(c.data.train_datasets) == {"pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper"}
assert c.run_provenance.obstacle_data is False and c.norm_stats.sample_frac == 1.0
assert c.model.optimizer._target_ == "torch.optim.AdamW"
assert c.model.optimizer.lr == 1e-4 and c.model.optimizer.weight_decay == 1e-6
m = instantiate(c.model.pipeline)
assert sum(p.numel() for p in m.nets.parameters()) == int(sys.argv[2])
assert [s.__class__.__name__ for s in m.pipeline.stages] == [
    "FusedObsEncoder", "ActionTargetBuilder", "DiffusionNoisingStage",
    "DiffusionDenoiserStage", "DiffusionEpsilonLossStage",
]
print("[preflight] resolved config and parameters PASS")
PY
sha256sum "$PROVENANCE/resolved_config.yaml" "$PROVENANCE/split_audit.json" > "$PROVENANCE/sha256.txt"

# Materialize train-only normalization once, before DDP ranks start, then pin it.
"$PY" -m egomimic.trainHydra "${OVERRIDES[@]}" norm_stats_only=true \
  trainer.accelerator=cpu trainer.devices=1 logger=null \
  "norm_stats.save_cache_dir=$RUN_DIR" hydra.run.dir="$RUN_DIR/norm_materialize"
test -s "$RUN_DIR/norm_stats/norm_stats.json"

srun --kill-on-bad-exit=1 --unbuffered "$PY" -m egomimic.trainHydra "${OVERRIDES[@]}" \
  "norm_stats.precomputed_norm_path=$RUN_DIR/norm_stats" \
  2>&1 | tee "$RUN_DIR/training.log"
"$PY" scripts/train/verify_planar_training_smoke.py "$RUN_DIR" \
  --expected-head "$EXPECTED_HEAD" --world-size "$WORLD_SIZE" --parameter-count "$PARAMETER_COUNT"
sha256sum "$RUN_DIR/norm_stats/norm_stats.json" >> "$PROVENANCE/sha256.txt"
printf '[smoke] ALL_PASS job=%s wandb=%s\n' "$SLURM_JOB_ID" \
  "https://wandb.ai/rl2-group/pushshapes-planar-v2/runs/$WANDB_ID"
