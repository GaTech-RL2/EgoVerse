#!/bin/bash
# Precompute the immutable full-train min/max artifact shared by the corrected
# regular and temporal Flow Transfer cotrain arms with per-embodiment proprio.
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

REPO=/coc/flash7/paphiwetsa3/worktrees/flow-transfer-temporal-compression-20260829
PY_ENV=/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv
EXPECTED_ENV_SHA=8fd1504c955756adf8167f7bd34fc1a09cb844d268898b3ac30693e99ac60e87
EXP_ROOT=/coc/flash7/paphiwetsa3/experiments/flow_transfer_temporal_h8_l8_world2_smokes_20260829
SCRIPT=$REPO/scripts/train/flow_transfer_newdata_h16_norm_precompute.sbatch
U_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/u_socket_3000_v2_clean
CHAIN_BASE_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_3000_v2
CHAIN_GEN_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_gen_flow_transfer_frozen719_20260829
COTRAIN12_EXPERIMENT=pusht/pipeline_sampler_usocket_chain_newdata_cotrain12_per_emb_proprio_h16
TEMPORAL_EXPERIMENT=pusht/pipeline_sampler_usocket_chain_newdata_temporal_h8_l8_w256_d12_dec64_per_emb_proprio
EXPECTED_ACTION_HORIZON=16
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
test "$EXPECTED_CHAIN_GEN_COUNT" = 719
test "$EXPECTED_CHAIN_GEN_SOURCE_FRAMES" = 306591
test "${SLURM_RESTART_COUNT:-0}" = 0
test "${SLURM_JOB_NUM_NODES:?}" = 1
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" -ge 8
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$SCRIPT" | awk '{print $1}')" = "$EXPECTED_SCRIPT_SHA"
test "$(environment_manifest | sha256sum | awk '{print $1}')" = "$EXPECTED_ENV_SHA"

OUT=$EXP_ROOT/norm_artifacts/per_emb_proprio_h16/job_${SLURM_JOB_ID:?}
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
  "$EXPECTED_CHAIN_GEN_SOURCE_FRAMES" <<'PY'
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
) = sys.argv[1:]
payload = {
    "expected_head": head,
    "script_sha256": script_sha,
    "action_horizon": 16,
    "norm_mode": "minmax",
    "reduce_all_but_last": True,
    "sample_frac": 1.0,
    "split": "train_only",
    "experiments": [
        "pusht/pipeline_sampler_usocket_chain_newdata_"
        "cotrain12_per_emb_proprio_h16",
        "pusht/pipeline_sampler_usocket_chain_newdata_"
        "temporal_h8_l8_w256_d12_dec64_per_emb_proprio",
    ],
    "normalized_schema": {
        "19": {"state_agent_model": 4, "actions": 4},
        "20": {"state_agent_model": 6, "actions": 6},
    },
    "chain_train_episode_count": int(base_count) + int(gen_count),
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
            "count": int(gen_count),
            "inventory_sha256": gen_sha,
            "frames": int(gen_source_frames),
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

COTRAIN12_CONFIG=$OUT/provenance/resolved_cotrain12_per_emb_proprio_config.yaml
TEMPORAL_CONFIG=$OUT/provenance/resolved_temporal_h8_l8_per_emb_proprio_config.yaml
python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  "+experiment=$COTRAIN12_EXPERIMENT" "${NORM_OVERRIDES[@]}" \
  --cfg job --resolve > "$COTRAIN12_CONFIG"
python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  "+experiment=$TEMPORAL_EXPERIMENT" "${NORM_OVERRIDES[@]}" \
  --cfg job --resolve > "$TEMPORAL_CONFIG"

python - "$COTRAIN12_CONFIG" "$TEMPORAL_CONFIG" "$EXPECTED_ACTION_HORIZON" <<'PY'
import pathlib
import sys

from omegaconf import OmegaConf

cotrain12_path = pathlib.Path(sys.argv[1])
temporal_path = pathlib.Path(sys.argv[2])
horizon = int(sys.argv[3])
cotrain12 = OmegaConf.load(cotrain12_path)
temporal = OmegaConf.load(temporal_path)
assert OmegaConf.to_container(cotrain12.data, resolve=True) == OmegaConf.to_container(
    temporal.data, resolve=True
)
cfg = temporal

expected_domains = {
    "pushshapes_sim_u_socket",
    "pushshapes_sim_chain_gripper",
}
expected_roots = [
    "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_3000_v2",
    "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_gen_flow_transfer_frozen719_20260829",
]

assert cfg.seed == 42
assert cfg.mode == "norm_stats"
assert cfg.norm_stats.norm_mode == "minmax"
assert cfg.norm_stats.reduce_all_but_last is True
assert cfg.norm_stats.sample_frac == 1.0
assert cfg.norm_stats.precomputed_norm_path is None
model = cfg.model.robomimic_model
assert model.action_horizon == horizon
assert set(model.domains) == expected_domains
assert set(cfg.data.train_datasets) == expected_domains
assert set(cfg.data.valid_datasets) == expected_domains
assert all(
    dataset.mode == "train" and float(dataset.valid_ratio) == 0.0
    for dataset in cfg.data.train_datasets.values()
)
for split_name in ("train_datasets", "valid_datasets"):
    split = cfg.data[split_name]
    chain_dataset = split.pushshapes_sim_chain_gripper
    assert "filters" not in chain_dataset

    for dataset in split.values():
        assert dataset.resolver.key_map.action_horizon == horizon
        assert dataset.resolver.key_map._target_.endswith(
            "get_keymap_hpt_per_emb_proprio"
        )
    u = split.pushshapes_sim_u_socket.resolver
    chain = split.pushshapes_sim_chain_gripper.resolver
    assert str(u.folder_path).endswith("/u_socket_3000_v2_clean")
    assert u.transform_list._target_.endswith(
        "get_usocket_rotvec_action_state_transform_list"
    )
    assert list(chain.folder_paths) == expected_roots
    assert chain._target_.endswith(
        "LocalEpisodeResolverManyWithEmbodimentOverride"
    )
    assert chain.transform_list._target_.endswith(
        "get_chain_gripper_point_transform_list"
    )
    assert float(chain.transform_list.world_size) == 512.0

projection = model.stages[0]
assert projection._target_.endswith("EmbodimentProprioProjection")
assert projection.projections.pushshapes_sim_u_socket.source_dim == 4
assert projection.projections.pushshapes_sim_chain_gripper.source_dim == 6
noise = model.stages[2]
assert noise.num_tokens == 8
assert noise.latent_dim == 8
sampler = model.stages[3]
assert sampler._target_.endswith("LatentFlowSampler")
assert sampler.denoising_module.act_seq == 8
decoder = model.stages[4]
assert decoder._target_.endswith("PerEmbodimentActionDecoder")
assert decoder.decoders.pushshapes_sim_u_socket.action_dim == 4
assert decoder.decoders.pushshapes_sim_chain_gripper.action_dim == 6
assert "action_encoder" not in OmegaConf.to_yaml(cfg.model).lower()
assert "arc_length" not in temporal_path.read_text().lower()

cotrain12_model = cotrain12.model.robomimic_model
assert cotrain12_model.stages[0]._target_.endswith("EmbodimentProprioProjection")
assert cotrain12_model.stages[2].num_tokens == 16
assert cotrain12_model.stages[3]._target_.endswith("LatentFlowSampler")
assert cotrain12_model.stages[4]._target_.endswith(
    "PerEmbodimentActionDecoder"
)
assert "arc_length" not in cotrain12_path.read_text().lower()
PY

# Norm mode uses only datamodule.train_datasets and returns before constructing
# the policy. Compose the exact temporal recipe so transformed U proprio is 4-D.
python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  "+experiment=$TEMPORAL_EXPERIMENT" "${NORM_OVERRIDES[@]}"
test -s "$ART"

python - \
  "$ART" "$OUT/validation.json" "$EXPECTED_ACTION_HORIZON" \
  "$EXPECTED_U_TRAIN_FRAMES" \
  "$EXPECTED_CHAIN_BASE_TRAIN_FRAMES" \
  "$EXPECTED_CHAIN_GEN_SOURCE_FRAMES" \
  "$EXPECTED_CHAIN_BASE_COUNT" "$EXPECTED_CHAIN_GEN_COUNT" <<'PY'
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
base_source_count = int(sys.argv[7])
gen_source_count = int(sys.argv[8])
expected_frames = {"19": u_frames, "20": base_frames + gen_source_frames}

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
    "19": {"state_agent_model": 4, "actions": 4},
    "20": {"state_agent_model": 6, "actions": 6},
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
            "shared_by": [
                "cotrain12_per_emb_proprio_h16",
                "temporal_h8_l8_dec64_per_emb_proprio",
            ],
            "action_horizon": action_horizon,
            "artifact": str(artifact),
            "sha256": artifact_sha,
            "bytes": len(raw),
            "per_source_train_frames": {
                "usocket": u_frames,
                "chain_base": base_frames,
                "chain_gen": gen_source_frames,
            },
            "chain_gen_source_frames": gen_source_frames,
            "chain_gen_source_episodes": gen_source_count,
            "chain_gen_train_episodes": gen_source_count,
            "chain_train_episodes": base_source_count + gen_source_count,
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

test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$SCRIPT" | awk '{print $1}')" = "$EXPECTED_SCRIPT_SHA"

sha256sum "$ART" > "$OUT/norm_stats.sha256"
sha256sum \
  "$OUT/provenance/input_contract.json" \
  "$OUT/provenance/resolved_cotrain12_per_emb_proprio_config.yaml" \
  "$OUT/provenance/resolved_temporal_h8_l8_per_emb_proprio_config.yaml" \
  "$OUT/provenance/inventories/before/"* \
  "$OUT/provenance/inventories/after/"* \
  > "$OUT/provenance/artifacts.sha256"
chmod 0444 \
  "$ART" \
  "$OUT/norm_stats.sha256" \
  "$OUT/validation.json" \
  "$OUT/provenance/input_contract.json" \
  "$OUT/provenance/resolved_cotrain12_per_emb_proprio_config.yaml" \
  "$OUT/provenance/resolved_temporal_h8_l8_per_emb_proprio_config.yaml" \
  "$OUT/provenance/artifacts.sha256" \
  "$OUT/provenance/inventories/before/usocket_inventory.txt" \
  "$OUT/provenance/inventories/before/chain_base_inventory.txt" \
  "$OUT/provenance/inventories/before/chain_gen_inventory.txt" \
  "$OUT/provenance/inventories/before/"*_episode_metadata.tsv \
  "$OUT/provenance/inventories/after/"*

printf '[norm] ALL_PASS head=%s horizon=%s artifact=%s\n' \
  "$EXPECTED_HEAD" "$EXPECTED_ACTION_HORIZON" "$ART"
