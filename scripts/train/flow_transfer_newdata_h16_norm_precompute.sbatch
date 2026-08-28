#!/bin/bash
# Precompute the single immutable full-train min/max artifact shared by the
# new-data Flow Transfer DP H16 and decoder-only latent H16 cotrain arms.
# Submit this file as a one-node, one-task CPU job only after freezing all
# three dataset roots and recording the exact contract values required below.

set -Eeuo pipefail

EXPECTED_HEAD=${EXPECTED_HEAD:?set the exact clean 40-character commit SHA}
EXPECTED_SCRIPT_SHA=${EXPECTED_SCRIPT_SHA:?set the exact SHA256 of this script}

EXPECTED_U_COUNT=${EXPECTED_U_COUNT:?set the exact U-Socket episode count}
EXPECTED_U_INVENTORY_SHA=${EXPECTED_U_INVENTORY_SHA:?set the sorted U-Socket inventory SHA256}
EXPECTED_U_TRAIN_FRAMES=${EXPECTED_U_TRAIN_FRAMES:?set the exact U-Socket train-frame count}

EXPECTED_CHAIN_BASE_COUNT=${EXPECTED_CHAIN_BASE_COUNT:?set the exact base-Chain episode count}
EXPECTED_CHAIN_BASE_INVENTORY_SHA=${EXPECTED_CHAIN_BASE_INVENTORY_SHA:?set the sorted base-Chain inventory SHA256}
EXPECTED_CHAIN_BASE_TRAIN_FRAMES=${EXPECTED_CHAIN_BASE_TRAIN_FRAMES:?set the exact base-Chain train-frame count}

EXPECTED_CHAIN_GEN_COUNT=${EXPECTED_CHAIN_GEN_COUNT:?set the exact generated-Chain episode count}
EXPECTED_CHAIN_GEN_INVENTORY_SHA=${EXPECTED_CHAIN_GEN_INVENTORY_SHA:?set the sorted generated-Chain inventory SHA256}
EXPECTED_CHAIN_GEN_SOURCE_FRAMES=${EXPECTED_CHAIN_GEN_SOURCE_FRAMES:?set the exact physical generated-Chain frame count}
EXPECTED_CHAIN_GEN_TRAIN_FRAMES=${EXPECTED_CHAIN_GEN_TRAIN_FRAMES:?set the effective generated-Chain train-frame count after filtering}

REPO=/coc/flash7/paphiwetsa3/worktrees/flow-transfer-cotrain-chain-combined-config-20260828
PY_ENV=/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv
EXPECTED_ENV_SHA=8fd1504c955756adf8167f7bd34fc1a09cb844d268898b3ac30693e99ac60e87
EXP_ROOT=/coc/flash7/paphiwetsa3/experiments/flow_transfer_newdata_h16_world2_smokes_20260828
SCRIPT=$REPO/scripts/train/flow_transfer_newdata_h16_norm_precompute.sbatch
U_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/u_socket_3000_v2_clean
CHAIN_BASE_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_3000_v2
CHAIN_GEN_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_gen
DP_EXPERIMENT=pusht/pipeline_diffusion_usocket_chain_newdata_h16
LATENT_EXPERIMENT=pusht/pipeline_sampler_usocket_chain_newdata_dense_medium_h16
EXPECTED_ACTION_HORIZON=16
EXCLUDED_CHAIN_GEN_EPISODE=episode_T_chain_gripper_obs7_000050
EXCLUDED_CHAIN_GEN_FOLDER=${EXCLUDED_CHAIN_GEN_EPISODE}.zarr
EXCLUDED_CHAIN_GEN_FRAMES=3118
SLURM_BIN=/opt/slurm/Ubuntu-20.04/24.11.0/bin
export PATH=$SLURM_BIN:$PATH

is_sha256() {
  [[ "$1" =~ ^[0-9a-f]{64}$ ]]
}

is_git_sha() {
  [[ "$1" =~ ^[0-9a-f]{40}$ ]]
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

scan_root() {
  local label=$1
  local root=$2
  local inventory=$3
  local metadata=$4
  local expected_count=$5
  local expected_inventory_sha=$6
  local expected_frames=$7

  "$PY_ENV/bin/python" - \
    "$label" "$root" "$inventory" "$metadata" \
    "$expected_count" "$expected_inventory_sha" "$expected_frames" <<'PY'
import hashlib
import json
import pathlib
import sys

(
    label,
    root_arg,
    inventory_arg,
    metadata_arg,
    expected_count_arg,
    expected_inventory_sha,
    expected_frames_arg,
) = sys.argv[1:]
root = pathlib.Path(root_arg)
inventory_path = pathlib.Path(inventory_arg)
metadata_path = pathlib.Path(metadata_arg)
expected_count = int(expected_count_arg)
expected_frames = int(expected_frames_arg)

assert root.is_dir(), f"{label} root is missing: {root}"
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
inventory_sha = hashlib.sha256(inventory_bytes).hexdigest()
assert inventory_sha == expected_inventory_sha, (
    label,
    inventory_sha,
    expected_inventory_sha,
)
inventory_path.write_bytes(inventory_bytes)

total_frames = 0
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
    total_frames += frames
    metadata_rows.append(
        f"{episode.name}\t{frames}\t{hashlib.sha256(raw).hexdigest()}\n"
    )
assert total_frames == expected_frames, (label, total_frames, expected_frames)
metadata_path.write_text("".join(metadata_rows))
print(
    f"[inventory] {label} episodes={len(names)} frames={total_frames} "
    f"sha256={inventory_sha}"
)
PY
}

verify_effective_chain_gen() {
  local inventory=$1
  local metadata=$2
  local effective_inventory=$3

  "$PY_ENV/bin/python" - \
    "$inventory" "$metadata" "$effective_inventory" \
    "$EXCLUDED_CHAIN_GEN_FOLDER" "$EXCLUDED_CHAIN_GEN_FRAMES" \
    "$EXPECTED_CHAIN_GEN_COUNT" "$EXPECTED_CHAIN_GEN_SOURCE_FRAMES" \
    "$EXPECTED_CHAIN_GEN_TRAIN_FRAMES" <<'PY'
import pathlib
import sys

(
    inventory_arg,
    metadata_arg,
    effective_inventory_arg,
    excluded_folder,
    excluded_frames_arg,
    expected_source_count_arg,
    expected_source_frames_arg,
    expected_train_frames_arg,
) = sys.argv[1:]
inventory = pathlib.Path(inventory_arg).read_text().splitlines()
metadata_rows = pathlib.Path(metadata_arg).read_text().splitlines()
excluded_frames = int(excluded_frames_arg)
expected_source_count = int(expected_source_count_arg)
expected_source_frames = int(expected_source_frames_arg)
expected_train_frames = int(expected_train_frames_arg)

assert len(inventory) == expected_source_count
assert inventory.count(excluded_folder) == 1
frames_by_episode = {}
for row in metadata_rows:
    episode, frames, _metadata_sha = row.split("\t")
    frames_by_episode[episode] = int(frames)
assert set(frames_by_episode) == set(inventory)
assert sum(frames_by_episode.values()) == expected_source_frames
assert frames_by_episode[excluded_folder] == excluded_frames

effective = [name for name in inventory if name != excluded_folder]
assert len(effective) == expected_source_count - 1
assert sum(frames_by_episode[name] for name in effective) == expected_train_frames
assert expected_source_frames - expected_train_frames == excluded_frames
pathlib.Path(effective_inventory_arg).write_text(
    "".join(f"{name}\n" for name in effective)
)
print(
    f"[inventory] chain_gen_effective episodes={len(effective)} "
    f"frames={expected_train_frames} excluded={excluded_folder}"
)
PY
}

scan_all() {
  local phase=$1
  local destination=$OUT/provenance/inventories/$phase
  mkdir -p "$destination"
  scan_root \
    usocket "$U_DATA" \
    "$destination/usocket_inventory.txt" \
    "$destination/usocket_episode_metadata.tsv" \
    "$EXPECTED_U_COUNT" "$EXPECTED_U_INVENTORY_SHA" \
    "$EXPECTED_U_TRAIN_FRAMES"
  scan_root \
    chain_base "$CHAIN_BASE_DATA" \
    "$destination/chain_base_inventory.txt" \
    "$destination/chain_base_episode_metadata.tsv" \
    "$EXPECTED_CHAIN_BASE_COUNT" "$EXPECTED_CHAIN_BASE_INVENTORY_SHA" \
    "$EXPECTED_CHAIN_BASE_TRAIN_FRAMES"
  scan_root \
    chain_gen "$CHAIN_GEN_DATA" \
    "$destination/chain_gen_inventory.txt" \
    "$destination/chain_gen_episode_metadata.tsv" \
    "$EXPECTED_CHAIN_GEN_COUNT" "$EXPECTED_CHAIN_GEN_INVENTORY_SHA" \
    "$EXPECTED_CHAIN_GEN_SOURCE_FRAMES"
  verify_effective_chain_gen \
    "$destination/chain_gen_inventory.txt" \
    "$destination/chain_gen_episode_metadata.tsv" \
    "$destination/chain_gen_effective_inventory.txt"
}

test -x "$PY_ENV/bin/python"
test -x "$SLURM_BIN/scontrol"
is_git_sha "$EXPECTED_HEAD"
is_sha256 "$EXPECTED_SCRIPT_SHA"
is_sha256 "$EXPECTED_U_INVENTORY_SHA"
is_sha256 "$EXPECTED_CHAIN_BASE_INVENTORY_SHA"
is_sha256 "$EXPECTED_CHAIN_GEN_INVENTORY_SHA"
is_positive_integer "$EXPECTED_U_COUNT"
is_positive_integer "$EXPECTED_CHAIN_BASE_COUNT"
is_positive_integer "$EXPECTED_CHAIN_GEN_COUNT"
is_positive_integer "$EXPECTED_U_TRAIN_FRAMES"
is_positive_integer "$EXPECTED_CHAIN_BASE_TRAIN_FRAMES"
is_positive_integer "$EXPECTED_CHAIN_GEN_SOURCE_FRAMES"
is_positive_integer "$EXPECTED_CHAIN_GEN_TRAIN_FRAMES"
test "$EXPECTED_CHAIN_GEN_COUNT" = 720
test "$EXPECTED_CHAIN_GEN_SOURCE_FRAMES" = 309709
test "$EXPECTED_CHAIN_GEN_TRAIN_FRAMES" = 306591
test "$((EXPECTED_CHAIN_GEN_SOURCE_FRAMES - EXPECTED_CHAIN_GEN_TRAIN_FRAMES))" \
  = "$EXCLUDED_CHAIN_GEN_FRAMES"
test "${SLURM_RESTART_COUNT:-0}" = 0
test "${SLURM_JOB_NUM_NODES:?}" = 1
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" -ge 8
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$SCRIPT" | awk '{print $1}')" = "$EXPECTED_SCRIPT_SHA"
test "$(environment_manifest | sha256sum | awk '{print $1}')" = "$EXPECTED_ENV_SHA"

OUT=$EXP_ROOT/norm_artifacts/shared_h16/job_${SLURM_JOB_ID:?}
ART=$OUT/artifact/norm_stats/norm_stats.json
test ! -e "$OUT"
mkdir -p "$OUT/provenance"

environment_manifest > "$OUT/provenance/python_environment.txt"
sha256sum "$OUT/provenance/python_environment.txt" \
  > "$OUT/provenance/python_environment.sha256"
cp "$SCRIPT" "$OUT/provenance/launcher.sbatch"
git -C "$REPO" log -1 --format=fuller > "$OUT/provenance/git_commit.txt"
git -C "$REPO" status --porcelain=v1 --untracked-files=all \
  > "$OUT/provenance/git_status.txt"
"$SLURM_BIN/scontrol" show job -dd "$SLURM_JOB_ID" \
  > "$OUT/provenance/slurm_job.txt"

scan_all before

"$PY_ENV/bin/python" - \
  "$OUT/provenance/input_contract.json" \
  "$EXPECTED_HEAD" "$EXPECTED_SCRIPT_SHA" \
  "$EXPECTED_U_COUNT" "$EXPECTED_U_INVENTORY_SHA" \
  "$EXPECTED_U_TRAIN_FRAMES" \
  "$EXPECTED_CHAIN_BASE_COUNT" "$EXPECTED_CHAIN_BASE_INVENTORY_SHA" \
  "$EXPECTED_CHAIN_BASE_TRAIN_FRAMES" \
  "$EXPECTED_CHAIN_GEN_COUNT" "$EXPECTED_CHAIN_GEN_INVENTORY_SHA" \
  "$EXPECTED_CHAIN_GEN_SOURCE_FRAMES" \
  "$EXPECTED_CHAIN_GEN_TRAIN_FRAMES" \
  "$EXCLUDED_CHAIN_GEN_EPISODE" "$EXCLUDED_CHAIN_GEN_FRAMES" <<'PY'
import json
import pathlib
import sys

(
    output,
    head,
    script_sha,
    u_count,
    u_sha,
    u_frames,
    base_count,
    base_sha,
    base_frames,
    gen_count,
    gen_sha,
    gen_source_frames,
    gen_train_frames,
    excluded_gen_episode,
    excluded_gen_frames,
) = sys.argv[1:]
payload = {
    "expected_head": head,
    "script_sha256": script_sha,
    "action_horizon": 16,
    "norm_mode": "minmax",
    "reduce_all_but_last": True,
    "sample_frac": 1.0,
    "split": "train_only",
    "effective_chain_train_episode_count": int(base_count) + int(gen_count) - 1,
    "datasets": {
        "usocket": {
            "count": int(u_count),
            "inventory_sha256": u_sha,
            "frames": int(u_frames),
        },
        "chain_base": {
            "count": int(base_count),
            "inventory_sha256": base_sha,
            "frames": int(base_frames),
        },
        "chain_gen": {
            "source_count": int(gen_count),
            "effective_train_count": int(gen_count) - 1,
            "inventory_sha256": gen_sha,
            "source_frames": int(gen_source_frames),
            "effective_train_frames": int(gen_train_frames),
            "excluded_episode": excluded_gen_episode,
            "excluded_episode_frames": int(excluded_gen_frames),
        },
    },
}
pathlib.Path(output).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY

source "$PY_ENV/bin/activate"
cd "$REPO"
export PYTHONPATH=$REPO
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

NORM_OVERRIDES=(
  mode=norm_stats
  "hydra.run.dir=$OUT/hydra_run"
  "++paths.root_dir=$OUT"
  "paths.output_dir=$OUT/hydra_run"
  "paths.work_dir=$REPO"
  norm_stats.norm_mode=minmax
  norm_stats.reduce_all_but_last=true
  norm_stats.sample_frac=1.0
  norm_stats.num_workers=4
  "norm_stats.save_cache_dir=$OUT/artifact"
  norm_stats.precomputed_norm_path=null
)

DP_CONFIG=$OUT/provenance/resolved_dp_config.yaml
LATENT_CONFIG=$OUT/provenance/resolved_latent_h16_config.yaml
python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  "+experiment=$DP_EXPERIMENT" "${NORM_OVERRIDES[@]}" \
  --cfg job --resolve > "$DP_CONFIG"
python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  "+experiment=$LATENT_EXPERIMENT" "${NORM_OVERRIDES[@]}" \
  --cfg job --resolve > "$LATENT_CONFIG"

python - "$DP_CONFIG" "$LATENT_CONFIG" "$EXPECTED_ACTION_HORIZON" <<'PY'
import pathlib
import sys

from omegaconf import OmegaConf

dp_path = pathlib.Path(sys.argv[1])
latent_path = pathlib.Path(sys.argv[2])
horizon = int(sys.argv[3])
dp = OmegaConf.load(dp_path)
latent = OmegaConf.load(latent_path)

expected_domains = {
    "pushshapes_sim_u_socket",
    "pushshapes_sim_chain_gripper",
}
expected_roots = [
    "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_3000_v2",
    "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_gen",
]
expected_exclusion = (
    "lambda row: row.get('episode_hash') != "
    "'episode_T_chain_gripper_obs7_000050'"
)

for cfg in (dp, latent):
    assert cfg.seed == 42
    assert cfg.mode == "norm_stats"
    assert cfg.norm_stats.norm_mode == "minmax"
    assert cfg.norm_stats.reduce_all_but_last is True
    assert cfg.norm_stats.sample_frac == 1.0
    assert cfg.norm_stats.precomputed_norm_path is None
    assert cfg.model.robomimic_model.action_horizon == horizon
    assert set(cfg.model.robomimic_model.domains) == expected_domains
    assert set(cfg.data.train_datasets) == expected_domains
    assert set(cfg.data.valid_datasets) == expected_domains
    assert all(
        dataset.mode == "train" and float(dataset.valid_ratio) == 0.0
        for dataset in cfg.data.train_datasets.values()
    )
    for split_name in ("train_datasets", "valid_datasets"):
        chain_dataset = cfg.data[split_name].pushshapes_sim_chain_gripper
        filters = chain_dataset.filters
        assert filters._target_ == "egomimic.rldb.filters.DatasetFilter"
        assert list(filters.filter_lambdas) == [expected_exclusion]
    for split_name in ("train_datasets", "valid_datasets"):
        split = cfg.data[split_name]
        for dataset in split.values():
            assert dataset.resolver.key_map.action_horizon == horizon
        u = split.pushshapes_sim_u_socket.resolver
        chain = split.pushshapes_sim_chain_gripper.resolver
        assert str(u.folder_path).endswith("/u_socket_3000_v2_clean")
        assert u.transform_list._target_.endswith("get_rotvec_transform_list")
        assert list(chain.folder_paths) == expected_roots
        assert chain._target_.endswith(
            "LocalEpisodeResolverManyWithEmbodimentOverride"
        )
        assert chain.transform_list._target_.endswith(
            "get_chain_gripper_point_transform_list"
        )
        assert float(chain.transform_list.world_size) == 512.0

# The data and transform tree is byte-for-byte equivalent after resolution, so
# the two model families must consume the same shared normalization artifact.
assert OmegaConf.to_container(dp.data, resolve=True) == OmegaConf.to_container(
    latent.data, resolve=True
)

dp_stage = dp.model.robomimic_model.stages[1]
assert dp_stage.action_horizon == horizon
assert all(policy.action_horizon == horizon for policy in dp_stage.policies.values())
assert (
    dp_stage.policies.pushshapes_sim_u_socket.infer_ac_dims
    .pushshapes_sim_u_socket
    == 4
)
assert (
    dp_stage.policies.pushshapes_sim_chain_gripper.infer_ac_dims
    .pushshapes_sim_chain_gripper
    == 6
)

latent_model = latent.model.robomimic_model
assert latent_model.stages[1].action_horizon == horizon
latent_sampler = latent_model.stages[2]
assert latent_sampler.action_horizon == horizon
assert latent_sampler.denoising_module.act_seq == horizon
assert latent_sampler.action_dims.pushshapes_sim_u_socket == 4
assert latent_sampler.action_dims.pushshapes_sim_chain_gripper == 6
assert "action_encoder" not in OmegaConf.to_yaml(latent.model).lower()
assert "arc_length" not in dp_path.read_text().lower()
assert "arc_length" not in latent_path.read_text().lower()
PY

# Norm mode uses only datamodule.train_datasets and returns before constructing
# either policy, so the canonical DP composition computes this shared artifact.
python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  "+experiment=$DP_EXPERIMENT" "${NORM_OVERRIDES[@]}"
test -s "$ART"

python - \
  "$ART" "$OUT/validation.json" "$EXPECTED_ACTION_HORIZON" \
  "$EXPECTED_U_TRAIN_FRAMES" \
  "$EXPECTED_CHAIN_BASE_TRAIN_FRAMES" \
  "$EXPECTED_CHAIN_GEN_SOURCE_FRAMES" \
  "$EXPECTED_CHAIN_GEN_TRAIN_FRAMES" \
  "$EXPECTED_CHAIN_BASE_COUNT" "$EXPECTED_CHAIN_GEN_COUNT" \
  "$EXCLUDED_CHAIN_GEN_EPISODE" <<'PY'
import hashlib
import json
import math
import pathlib
import sys

artifact = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
action_horizon = int(sys.argv[3])
u_frames = int(sys.argv[4])
base_frames = int(sys.argv[5])
gen_source_frames = int(sys.argv[6])
gen_train_frames = int(sys.argv[7])
base_source_count = int(sys.argv[8])
gen_source_count = int(sys.argv[9])
excluded_gen_episode = sys.argv[10]
expected_frames = {"19": u_frames, "20": base_frames + gen_train_frames}

raw = artifact.read_bytes()
payload = json.loads(raw)
assert action_horizon == 16
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
stat_names = {
    "mean",
    "std",
    "min",
    "max",
    "median",
    "quantile_1",
    "quantile_99",
    "quantile_0_01",
    "quantile_99_99",
}
for embodiment, widths in {
    "19": {"state_agent_obj": 6, "actions": 4},
    "20": {"state_agent_obj": 6, "actions": 6},
}.items():
    assert set(payload["stats"][embodiment]) == set(widths)
    for key, width in widths.items():
        stats = payload["stats"][embodiment][key]
        assert set(stats) == stat_names
        for values in stats.values():
            assert isinstance(values, list) and len(values) == width
            assert all(math.isfinite(float(value)) for value in values)
        assert all(a <= b for a, b in zip(stats["min"], stats["max"]))
        assert all(value >= 0 for value in stats["std"])

active_stats = {
    embodiment: {
        key: {
            name: payload["stats"][embodiment][key][name]
            for name in ("min", "max")
        }
        for key in sorted(payload["stats"][embodiment])
    }
    for embodiment in sorted(payload["stats"])
}
active_bytes = json.dumps(
    active_stats, sort_keys=True, separators=(",", ":")
).encode()
artifact_sha = hashlib.sha256(raw).hexdigest()
output.write_text(
    json.dumps(
        {
            "status": "PASS",
            "shared_by": ["dp_h16", "latent_dense_h16"],
            "action_horizon": action_horizon,
            "artifact": str(artifact),
            "sha256": artifact_sha,
            "bytes": len(raw),
            "per_source_train_frames": {
                "usocket": u_frames,
                "chain_base": base_frames,
                "chain_gen": gen_train_frames,
            },
            "chain_gen_source_frames": gen_source_frames,
            "chain_gen_source_episodes": gen_source_count,
            "chain_gen_effective_train_episodes": gen_source_count - 1,
            "chain_effective_train_episodes": (
                base_source_count + gen_source_count - 1
            ),
            "excluded_chain_gen_episode": excluded_gen_episode,
            "per_embodiment_train_frames": expected_frames,
            "total_train_frames": payload["frames"],
            "sample_frac": 1.0,
            "seed": 42,
            "active_minmax_sha256": hashlib.sha256(active_bytes).hexdigest(),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
print(artifact_sha)
PY

# Reject any append, removal, rename, or episode-root metadata change that
# happened while the full-data statistics were being computed.
scan_all after
for label in usocket chain_base chain_gen; do
  cmp -s \
    "$OUT/provenance/inventories/before/${label}_inventory.txt" \
    "$OUT/provenance/inventories/after/${label}_inventory.txt"
  cmp -s \
    "$OUT/provenance/inventories/before/${label}_episode_metadata.tsv" \
    "$OUT/provenance/inventories/after/${label}_episode_metadata.tsv"
done
cmp -s \
  "$OUT/provenance/inventories/before/chain_gen_effective_inventory.txt" \
  "$OUT/provenance/inventories/after/chain_gen_effective_inventory.txt"

test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$SCRIPT" | awk '{print $1}')" = "$EXPECTED_SCRIPT_SHA"

sha256sum "$ART" > "$OUT/norm_stats.sha256"
sha256sum \
  "$OUT/provenance/input_contract.json" \
  "$OUT/provenance/resolved_dp_config.yaml" \
  "$OUT/provenance/resolved_latent_h16_config.yaml" \
  "$OUT/provenance/inventories/before/"* \
  "$OUT/provenance/inventories/after/"* \
  > "$OUT/provenance/artifacts.sha256"
chmod 0444 \
  "$ART" \
  "$OUT/norm_stats.sha256" \
  "$OUT/validation.json" \
  "$OUT/provenance/input_contract.json" \
  "$OUT/provenance/resolved_dp_config.yaml" \
  "$OUT/provenance/resolved_latent_h16_config.yaml" \
  "$OUT/provenance/artifacts.sha256" \
  "$OUT/provenance/inventories/before/usocket_inventory.txt" \
  "$OUT/provenance/inventories/before/chain_base_inventory.txt" \
  "$OUT/provenance/inventories/before/chain_gen_inventory.txt" \
  "$OUT/provenance/inventories/before/chain_gen_effective_inventory.txt" \
  "$OUT/provenance/inventories/before/"*_episode_metadata.tsv \
  "$OUT/provenance/inventories/after/"*

printf '[norm] ALL_PASS head=%s horizon=%s artifact=%s\n' \
  "$EXPECTED_HEAD" "$EXPECTED_ACTION_HORIZON" "$ART"
