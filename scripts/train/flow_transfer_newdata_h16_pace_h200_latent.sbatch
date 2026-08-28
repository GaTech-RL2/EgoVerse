#!/bin/bash
# PACE-only smoke/full launcher for the new-data Flow Transfer H16 latent
# cotrain arm. Resource directives stay in the sbatch submission so this file
# can fail closed against the live H200 allocation and immutable smoke gate.

set -Eeuo pipefail

MODE=${MODE:?set MODE=smoke or MODE=full}
EXPECTED_HEAD=${EXPECTED_HEAD:?set the exact clean 40-character commit SHA}
EXPECTED_LAUNCHER_SHA=${EXPECTED_LAUNCHER_SHA:?set the exact launcher SHA256}
SMOKE_GATE=${SMOKE_GATE:-}
EXPECTED_SMOKE_GATE_SHA=${EXPECTED_SMOKE_GATE_SHA:-}

REPO=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-flow-transfer-newdata-h16-pace-20260828
PY_ENV=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse/.venv
EXPECTED_ENV_SHA=512f34f64413cfc326cf080113bd5c8ca2aae9b17813122eff0d1d2f3362bbed
EXP_ROOT=/storage/project/r-dxu345-0/paphiwetsa3/experiments/flow_transfer_newdata_h16_pace_h200_20260828
ARTIFACT_ROOT=$EXP_ROOT/source_artifacts/shared_h16/skynet_job_3724719
NORM_ARTIFACT=$ARTIFACT_ROOT/artifact/norm_stats/norm_stats.json
NORM_VALIDATION=$ARTIFACT_ROOT/validation.json
EXPECTED_NORM_SHA=9af0eb68f42d70d17af4fac36164ba0207457190beaba40d2e9bf7c23d08823e
EXPECTED_NORM_VALIDATION_SHA=5a250b0427f7ad7784d28cb6f42ef3fa3ca4654a65213781f32ccc95fc951980
TRANSPORT_VALIDATION=$EXP_ROOT/source_artifacts/transport/skynet_job_3724901/validation.json
EXPECTED_TRANSPORT_VALIDATION_SHA=c0a73f5c3f35b6975dc93210541185980179d777d6a7f1c8539320505dafda46
INVENTORY_ROOT=$ARTIFACT_ROOT/provenance/inventories/before
U_INVENTORY=$INVENTORY_ROOT/usocket_inventory.txt
EXPECTED_U_INVENTORY_SHA=fb6e107b585d6a5fae9e9ea4186f441b625a680a36d5691469d9c4125cd9a0fa
U_EPISODE_METADATA=$INVENTORY_ROOT/usocket_episode_metadata.tsv
EXPECTED_U_EPISODE_METADATA_SHA=63ae9fc7c8f1d1e092360f1b3c2784881641716c4035a3f07a2a7482b768bebe
EXPECTED_U_COUNT=2999
EXPECTED_U_TRAIN_FRAMES=521676
CHAIN_BASE_INVENTORY=$INVENTORY_ROOT/chain_base_inventory.txt
EXPECTED_CHAIN_BASE_INVENTORY_SHA=3ca50bf2558a3afbe11784a0b3d6424d3d3d9b1d9a564470618d1b64e2e99ede
CHAIN_BASE_EPISODE_METADATA=$INVENTORY_ROOT/chain_base_episode_metadata.tsv
EXPECTED_CHAIN_BASE_EPISODE_METADATA_SHA=45ffa17064761e6f1c11998fdfe8cf985289ce004fcb4c030f3dc28fc408cfb0
EXPECTED_CHAIN_BASE_COUNT=3000
EXPECTED_CHAIN_BASE_FRAMES=931061
CHAIN_GEN_INVENTORY=$INVENTORY_ROOT/chain_gen_inventory.txt
EXPECTED_CHAIN_GEN_INVENTORY_SHA=f5d1b67b474b748522bbda122af2e6f359a1bf69db207a2531bee6b548d89956
CHAIN_GEN_EPISODE_METADATA=$INVENTORY_ROOT/chain_gen_episode_metadata.tsv
EXPECTED_CHAIN_GEN_EPISODE_METADATA_SHA=53f0b81aa4c00b8c6afc07b386a76ea03f5e3c3863403340bfc3f95c67aa4515
CHAIN_GEN_EFFECTIVE_INVENTORY=$INVENTORY_ROOT/chain_gen_effective_inventory.txt
EXPECTED_CHAIN_GEN_EFFECTIVE_INVENTORY_SHA=f741815ad52b94a1e9a6ae2d15c73c3d9415bb302e102c29769590dfa0339f87
EXPECTED_CHAIN_GEN_COUNT=720
EXPECTED_CHAIN_GEN_SOURCE_FRAMES=309709
EXPECTED_CHAIN_GEN_EFFECTIVE_COUNT=719
EXPECTED_CHAIN_GEN_EFFECTIVE_FRAMES=306591
EXPECTED_CHAIN_TRAIN_FRAMES=1237652

LAUNCHER=$REPO/scripts/train/flow_transfer_newdata_h16_pace_h200_latent.sbatch
VERIFIER=$REPO/egomimic/scripts/verify_training_smoke.py
U_DATA=/storage/project/r-dxu345-0/paphiwetsa3/datasets/flow_transfer_20260826/u_socket_3000_v2_clean
CHAIN_BASE_DATA=/storage/project/r-dxu345-0/paphiwetsa3/datasets/flow_transfer_newdata_h16_20260828/chain_gripper_3000_v2
CHAIN_GEN_DATA=/storage/project/r-dxu345-0/paphiwetsa3/datasets/flow_transfer_newdata_h16_20260828/chain_gripper_gen
EXCLUDED_CHAIN_EPISODE=episode_T_chain_gripper_obs7_000050
EXCLUDED_CHAIN_FOLDER=${EXCLUDED_CHAIN_EPISODE}.zarr
EXCLUDED_CHAIN_FRAMES=3118
EXPECTED_WORLD_SIZE=1
EXPECTED_GPU_MODEL=H200
EXPECTED_PARTITION=gpu-h200
EXPECTED_ACCOUNT=gts-dxu345-rl2
EXPECTED_QOS=inferno
SLURM_BIN=/opt/slurm/current/bin
export PATH=$SLURM_BIN:$PATH
SRUN=$SLURM_BIN/srun
EXPERIMENT=pusht/pipeline_sampler_usocket_chain_newdata_dense_medium_h16
ARM=latent_h16
BASE_WANDB_ID=ft_cotrain_newdata3719_latent_dense_m96_h16_s42_world1_pace_h200_20260828
WANDB_GROUP=flow_transfer_newdata3719_cotrain_h16_20260828

case "$MODE" in
  smoke)
    test -z "$SMOKE_GATE"
    test -z "$EXPECTED_SMOKE_GATE_SHA"
    RUN_DIR=$EXP_ROOT/smokes/$ARM/job_${SLURM_JOB_ID:?}
    PROVENANCE_DIR=$EXP_ROOT/provenance/smoke/$ARM/job_${SLURM_JOB_ID}
    WANDB_ID=${BASE_WANDB_ID}_smoke_job_${SLURM_JOB_ID}
    ;;
  full)
    SMOKE_GATE=${SMOKE_GATE:?full mode requires the immutable smoke gate JSON}
    EXPECTED_SMOKE_GATE_SHA=${EXPECTED_SMOKE_GATE_SHA:?full mode requires its SHA256}
    RUN_DIR=$EXP_ROOT/runs/cotrain_newdata_latent_h16_seed42
    PROVENANCE_DIR=$EXP_ROOT/provenance/full/$ARM/job_${SLURM_JOB_ID:?}
    WANDB_ID=$BASE_WANDB_ID
    ;;
  *)
    printf 'Unknown MODE=%s; expected smoke or full\n' "$MODE" >&2
    exit 64
    ;;
esac

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
  test "$(stat -c '%a' "$CHAIN_GEN_EFFECTIVE_INVENTORY")" = 444
  test "$(sha256sum "$CHAIN_GEN_EFFECTIVE_INVENTORY" | awk '{print $1}')" = \
    "$EXPECTED_CHAIN_GEN_EFFECTIVE_INVENTORY_SHA"
  "$PY_ENV/bin/python" - \
    "$U_EPISODE_METADATA" "$CHAIN_BASE_EPISODE_METADATA" \
    "$CHAIN_GEN_EPISODE_METADATA" "$CHAIN_GEN_EFFECTIVE_INVENTORY" \
    "$EXCLUDED_CHAIN_FOLDER" "$EXCLUDED_CHAIN_FRAMES" \
    "$EXPECTED_U_TRAIN_FRAMES" "$EXPECTED_CHAIN_BASE_FRAMES" \
    "$EXPECTED_CHAIN_GEN_SOURCE_FRAMES" \
    "$EXPECTED_CHAIN_GEN_EFFECTIVE_COUNT" \
    "$EXPECTED_CHAIN_GEN_EFFECTIVE_FRAMES" \
    "$EXPECTED_CHAIN_TRAIN_FRAMES" <<'PY'
import pathlib
import re
import sys

(
    u_metadata_arg,
    chain_base_metadata_arg,
    chain_gen_metadata_arg,
    chain_gen_effective_inventory_arg,
    excluded_folder,
    excluded_frames_arg,
    expected_u_frames_arg,
    expected_base_frames_arg,
    expected_gen_source_frames_arg,
    expected_gen_effective_count_arg,
    expected_gen_effective_frames_arg,
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
assert sum(base_frames.values()) == int(expected_base_frames_arg)
assert sum(gen_frames.values()) == int(expected_gen_source_frames_arg)
assert excluded_folder not in base_frames
assert gen_frames.get(excluded_folder) == excluded_frames
effective_gen_names = sorted(set(gen_frames) - {excluded_folder})
expected_effective_names = pathlib.Path(
    chain_gen_effective_inventory_arg
).read_text().splitlines()
assert effective_gen_names == expected_effective_names
assert len(effective_gen_names) == int(expected_gen_effective_count_arg)
effective_gen_frames = sum(gen_frames[name] for name in effective_gen_names)
assert effective_gen_frames == int(expected_gen_effective_frames_arg)
level_counts = {level: 0 for level in range(1, 31)}
for name in gen_frames:
    match = re.search(r"_obs(\d+)_", name)
    assert match is not None, name
    level = int(match.group(1))
    assert level in level_counts, (name, level)
    level_counts[level] += 1
assert set(level_counts.values()) == {24}, level_counts
effective_chain_frames = sum(base_frames.values()) + effective_gen_frames
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
is_sha256 "$EXPECTED_NORM_VALIDATION_SHA"
is_sha256 "$EXPECTED_TRANSPORT_VALIDATION_SHA"
is_sha256 "$EXPECTED_U_INVENTORY_SHA"
is_sha256 "$EXPECTED_U_EPISODE_METADATA_SHA"
is_sha256 "$EXPECTED_CHAIN_BASE_INVENTORY_SHA"
is_sha256 "$EXPECTED_CHAIN_BASE_EPISODE_METADATA_SHA"
is_sha256 "$EXPECTED_CHAIN_GEN_INVENTORY_SHA"
is_sha256 "$EXPECTED_CHAIN_GEN_EPISODE_METADATA_SHA"
is_sha256 "$EXPECTED_CHAIN_GEN_EFFECTIVE_INVENTORY_SHA"
is_positive_integer "$EXPECTED_U_COUNT"
is_positive_integer "$EXPECTED_CHAIN_BASE_COUNT"
is_positive_integer "$EXPECTED_CHAIN_GEN_COUNT"
is_positive_integer "$EXPECTED_CHAIN_GEN_EFFECTIVE_COUNT"
is_positive_integer "$EXPECTED_U_TRAIN_FRAMES"
is_positive_integer "$EXPECTED_CHAIN_TRAIN_FRAMES"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$EXPECTED_LAUNCHER_SHA"
test "$(environment_manifest | sha256sum | awk '{print $1}')" = "$EXPECTED_ENV_SHA"
test -s "$NORM_ARTIFACT"
test -s "$NORM_VALIDATION"
test "$(stat -c '%a' "$NORM_ARTIFACT")" = 444
test "$(stat -c '%a' "$NORM_VALIDATION")" = 444
test "$(sha256sum "$NORM_ARTIFACT" | awk '{print $1}')" = "$EXPECTED_NORM_SHA"
test "$(sha256sum "$NORM_VALIDATION" | awk '{print $1}')" = \
  "$EXPECTED_NORM_VALIDATION_SHA"
test -s "$TRANSPORT_VALIDATION"
test "$(stat -c '%a' "$TRANSPORT_VALIDATION")" = 444
test "$(sha256sum "$TRANSPORT_VALIDATION" | awk '{print $1}')" = \
  "$EXPECTED_TRANSPORT_VALIDATION_SHA"
test $((8#$(stat -c '%a' "$CHAIN_BASE_DATA") & 8#222)) = 0
test $((8#$(stat -c '%a' "$CHAIN_GEN_DATA") & 8#222)) = 0
validate_all_inventories

"$PY_ENV/bin/python" - \
  "$TRANSPORT_VALIDATION" "$U_DATA" "$CHAIN_BASE_DATA" \
  "$CHAIN_GEN_DATA" <<'PY'
import json
import pathlib
import sys

validation_arg, u_data, chain_base_data, chain_gen_data = sys.argv[1:]
payload = json.loads(pathlib.Path(validation_arg).read_text())
assert payload["status"] == "PASS"
assert payload["phase"] == "published"
assert payload["transfer_job"] == {
    "cluster": "skynet",
    "deleted_files": 0,
    "exit_code": "0:0",
    "regular_files_transferred": 52090,
    "slurm_job_id": 3724901,
    "state": "COMPLETED",
    "stderr_sha256": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "stdout_sha256": (
        "e4fd3ad40056367131d7f30e1f5452f5bd116f1faafae95d8326a4d05b1c5bcc"
    ),
    "terminal_marker": "TRANSFER_PASS",
}
assert payload["chain_transport_tree"] == {
    "directories_including_root": 70683,
    "directories_transferred_excluding_root": 70682,
    "other_nodes": 0,
    "regular_bytes": 2169451517,
    "regular_files": 52090,
    "symlinks": 0,
}
assert payload["datasets"]["u_socket"]["root"] == u_data
assert payload["datasets"]["chain_base"]["root"] == chain_base_data
assert payload["datasets"]["chain_gen"]["root"] == chain_gen_data
assert payload["data_contract"]["u_socket_effective_episodes"] == 2999
assert payload["data_contract"]["u_socket_effective_frames"] == 521676
assert payload["data_contract"]["chain_effective_total_episodes"] == 3719
assert payload["data_contract"]["chain_effective_total_frames"] == 1237652
assert payload["data_contract"]["chain_gen_level_counts"] == {
    str(level): 24 for level in range(1, 31)
}
PY

"$PY_ENV/bin/python" - \
  "$NORM_ARTIFACT" "$NORM_VALIDATION" \
  "$EXPECTED_NORM_SHA" "$EXPECTED_NORM_VALIDATION_SHA" \
  "$EXPECTED_U_TRAIN_FRAMES" "$EXPECTED_CHAIN_TRAIN_FRAMES" <<'PY'
import hashlib
import json
import math
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
validation_path = pathlib.Path(sys.argv[2])
expected_sha = sys.argv[3]
expected_validation_sha = sys.argv[4]
expected_frames = {"19": int(sys.argv[5]), "20": int(sys.argv[6])}
raw = path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == expected_sha
validation_raw = validation_path.read_bytes()
assert hashlib.sha256(validation_raw).hexdigest() == expected_validation_sha
payload = json.loads(raw)
validation = json.loads(validation_raw)
assert validation["status"] == "PASS"
assert validation["action_horizon"] == 16
assert validation["sha256"] == expected_sha
assert validation["sample_frac"] == 1.0
assert validation["seed"] == 42
assert validation["chain_effective_train_episodes"] == 3719
assert validation["chain_gen_source_episodes"] == 720
assert validation["chain_gen_effective_train_episodes"] == 719
assert validation["excluded_chain_gen_episode"] == "episode_T_chain_gripper_obs7_000050"
assert validation["per_embodiment_train_frames"] == {
    "19": expected_frames["19"],
    "20": expected_frames["20"],
}
assert validation["shared_by"] == ["dp_h16", "latent_dense_h16"]
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

RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
if test "$MODE" = smoke; then
  test "$RESTART_COUNT" = 0
fi
test "${SLURM_JOB_NUM_NODES:?}" = 1
test "${SLURM_NTASKS:?}" = "$EXPECTED_WORLD_SIZE"
test "${SLURM_CPUS_PER_TASK:?}" -ge 8
test "${SLURM_JOB_PARTITION:?}" = "$EXPECTED_PARTITION"
test "${SLURM_JOB_ACCOUNT:?}" = "$EXPECTED_ACCOUNT"
JOB_RECORD=$("$SLURM_BIN/scontrol" show job -o "$SLURM_JOB_ID")
grep -q " QOS=$EXPECTED_QOS " <<< "$JOB_RECORD"
"$PY_ENV/bin/python" - "$JOB_RECORD" "$MODE" <<'PY'
import re
import shlex
import sys

job_record, mode = sys.argv[1:]
fields = dict(re.findall(r"([A-Za-z][A-Za-z0-9/]*)=([^\s]+)", job_record))
submit_match = re.search(r"(?:^|\s)SubmitLine=(.*?)(?=\sWorkDir=)", job_record)
assert submit_match is not None, job_record
submit_tokens = shlex.split(submit_match.group(1))
has_usr1_300 = "--signal=USR1@300" in submit_tokens or any(
    submit_tokens[index : index + 2] == ["--signal", "USR1@300"]
    for index in range(len(submit_tokens) - 1)
)
if mode == "full":
    assert fields["TimeLimit"] == "2-00:00:00"
    assert fields["Requeue"] == "1"
    assert "--requeue" in submit_tokens
    assert has_usr1_300, submit_tokens
else:
    assert mode == "smoke"
    assert fields["Requeue"] == "0"
    assert "--requeue" not in submit_tokens
    assert not any(token.startswith("--signal") for token in submit_tokens)
PY
test -n "${CUDA_VISIBLE_DEVICES:-}"
IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
test "${#visible_gpus[@]}" = "$EXPECTED_WORLD_SIZE"

if test "$MODE" = full; then
  is_sha256 "$EXPECTED_SMOKE_GATE_SHA"
  test -s "$SMOKE_GATE"
  test "$(stat -c '%a' "$SMOKE_GATE")" = 444
  test "$(sha256sum "$SMOKE_GATE" | awk '{print $1}')" = \
    "$EXPECTED_SMOKE_GATE_SHA"
  "$PY_ENV/bin/python" - \
    "$SMOKE_GATE" "$EXPECTED_HEAD" "$EXPECTED_LAUNCHER_SHA" \
    "$EXPECTED_ENV_SHA" "$EXPECTED_NORM_SHA" \
    "$EXPECTED_TRANSPORT_VALIDATION_SHA" \
    "$EXPECTED_U_INVENTORY_SHA" "$EXPECTED_U_EPISODE_METADATA_SHA" \
    "$EXPECTED_CHAIN_BASE_INVENTORY_SHA" \
    "$EXPECTED_CHAIN_BASE_EPISODE_METADATA_SHA" \
    "$EXPECTED_CHAIN_GEN_INVENTORY_SHA" \
    "$EXPECTED_CHAIN_GEN_EPISODE_METADATA_SHA" \
    "$EXPECTED_CHAIN_GEN_EFFECTIVE_INVENTORY_SHA" <<'PY'
import hashlib
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
(
    expected_head,
    expected_launcher_sha,
    expected_env_sha,
    expected_norm_sha,
    expected_transport_validation_sha,
    expected_u_inventory_sha,
    expected_u_metadata_sha,
    expected_chain_base_inventory_sha,
    expected_chain_base_metadata_sha,
    expected_chain_gen_inventory_sha,
    expected_chain_gen_metadata_sha,
    expected_chain_gen_effective_inventory_sha,
) = sys.argv[2:]
payload = json.loads(path.read_text())
assert payload["status"] == "PASS"
assert payload["mode"] == "smoke"
assert payload["arm"] == "latent_h16"
assert payload["repo_head"] == expected_head
assert payload["launcher_sha256"] == expected_launcher_sha
assert payload["environment_sha256"] == expected_env_sha
assert payload["norm_sha256"] == expected_norm_sha
assert payload["transport_validation_sha256"] == expected_transport_validation_sha
assert payload["inventory_sha256"] == {
    "chain_base": expected_chain_base_inventory_sha,
    "chain_gen": expected_chain_gen_inventory_sha,
    "chain_gen_effective": expected_chain_gen_effective_inventory_sha,
    "usocket": expected_u_inventory_sha,
}
assert payload["episode_metadata_sha256"] == {
    "chain_base": expected_chain_base_metadata_sha,
    "chain_gen": expected_chain_gen_metadata_sha,
    "usocket": expected_u_metadata_sha,
}
assert payload["world_size"] == 1
assert payload["global_batch_per_domain"] == {
    "pushshapes_sim_chain_gripper": 64,
    "pushshapes_sim_u_socket": 64,
}
assert payload["action_horizon"] == 16
assert payload["gpu_model"] == "H200"
assert payload["partition"] == "gpu-h200"
assert payload["account"] == "gts-dxu345-rl2"
assert payload["qos"] == "inferno"
assert payload["smoke_result"]["status"] == "passed"
assert payload["smoke_result"]["global_step"] == 2
assert payload["smoke_result"]["scheduler_last_epoch"] == 2
assert payload["smoke_result"]["dense_training_steps"] == [0, 1]
assert payload["smoke_result"]["validation_trainer_global_step"] >= 1
assert payload["smoke_result"]["required_embodiments"] == [19, 20]
assert payload["smoke_result"]["world_size"] == 1
smoke_result_path = pathlib.Path(payload["smoke_result_path"])
assert smoke_result_path.is_file()
assert hashlib.sha256(smoke_result_path.read_bytes()).hexdigest() == payload[
    "smoke_result_sha256"
]
PY
fi

mkdir -p "$(dirname "$RUN_DIR")" "$(dirname "$PROVENANCE_DIR")"
if test "$RESTART_COUNT" = 0; then
  test ! -e "$RUN_DIR"
  test ! -e "$PROVENANCE_DIR"
  mkdir "$RUN_DIR" "$PROVENANCE_DIR"
  printf '%s\n' "$SLURM_JOB_ID" > "$RUN_DIR/.owner_job_id"
else
  test "$MODE" = full
  test -d "$RUN_DIR"
  test -d "$PROVENANCE_DIR"
  test "$(sed -n '1p' "$RUN_DIR/.owner_job_id")" = "$SLURM_JOB_ID"
fi

GPU_RECORD=$PROVENANCE_DIR/gpu_restart_${RESTART_COUNT}.txt
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
  --query-gpu=index,name,uuid,memory.total,driver_version \
  --format=csv,noheader > "$GPU_RECORD"
test "$(wc -l < "$GPU_RECORD")" = "$EXPECTED_WORLD_SIZE"
test "$(grep -ci "$EXPECTED_GPU_MODEL" "$GPU_RECORD")" = \
  "$EXPECTED_WORLD_SIZE"

copy_once_and_verify() {
  local source=$1 destination=$2
  if test -e "$destination"; then
    cmp -s "$source" "$destination"
  else
    cp "$source" "$destination"
  fi
}

copy_once_and_verify "$LAUNCHER" "$PROVENANCE_DIR/launcher.sbatch"
copy_once_and_verify "$VERIFIER" "$PROVENANCE_DIR/verify_training_smoke.py"
copy_once_and_verify "$NORM_ARTIFACT" "$PROVENANCE_DIR/norm_stats.json"
copy_once_and_verify "$NORM_VALIDATION" "$PROVENANCE_DIR/norm_validation.json"
copy_once_and_verify \
  "$TRANSPORT_VALIDATION" "$PROVENANCE_DIR/transport_validation.json"
copy_once_and_verify "$U_INVENTORY" "$PROVENANCE_DIR/usocket_inventory.txt"
copy_once_and_verify \
  "$U_EPISODE_METADATA" "$PROVENANCE_DIR/usocket_episode_metadata.tsv"
copy_once_and_verify \
  "$CHAIN_BASE_INVENTORY" "$PROVENANCE_DIR/chain_base_inventory.txt"
copy_once_and_verify "$CHAIN_BASE_EPISODE_METADATA" \
  "$PROVENANCE_DIR/chain_base_episode_metadata.tsv"
copy_once_and_verify \
  "$CHAIN_GEN_INVENTORY" "$PROVENANCE_DIR/chain_gen_inventory.txt"
copy_once_and_verify "$CHAIN_GEN_EPISODE_METADATA" \
  "$PROVENANCE_DIR/chain_gen_episode_metadata.tsv"
copy_once_and_verify "$CHAIN_GEN_EFFECTIVE_INVENTORY" \
  "$PROVENANCE_DIR/chain_gen_effective_inventory.txt"
if test "$MODE" = full; then
  copy_once_and_verify "$SMOKE_GATE" "$PROVENANCE_DIR/smoke_gate.json"
fi
environment_manifest > "$PROVENANCE_DIR/python_environment.txt"
test "$(sha256sum "$PROVENANCE_DIR/python_environment.txt" | awk '{print $1}')" = \
  "$EXPECTED_ENV_SHA"
git -C "$REPO" log -1 --format=fuller > "$PROVENANCE_DIR/git_commit.txt"
git -C "$REPO" status --porcelain=v1 --untracked-files=all \
  > "$PROVENANCE_DIR/git_status.txt"
"$SLURM_BIN/scontrol" show job -dd "$SLURM_JOB_ID" \
  > "$PROVENANCE_DIR/slurm_job.txt"
sha256sum \
  "$PROVENANCE_DIR/launcher.sbatch" \
  "$PROVENANCE_DIR/verify_training_smoke.py" \
  "$PROVENANCE_DIR/norm_stats.json" \
  "$PROVENANCE_DIR/norm_validation.json" \
  "$PROVENANCE_DIR/transport_validation.json" \
  "$PROVENANCE_DIR/usocket_inventory.txt" \
  "$PROVENANCE_DIR/usocket_episode_metadata.tsv" \
  "$PROVENANCE_DIR/chain_base_inventory.txt" \
  "$PROVENANCE_DIR/chain_base_episode_metadata.tsv" \
  "$PROVENANCE_DIR/chain_gen_inventory.txt" \
  "$PROVENANCE_DIR/chain_gen_episode_metadata.tsv" \
  "$PROVENANCE_DIR/chain_gen_effective_inventory.txt" \
  > "$PROVENANCE_DIR/artifacts.sha256"

RUNTIME_ENV=$PROVENANCE_DIR/runtime_env
mkdir -p "$RUNTIME_ENV"
if test -e "$RUNTIME_ENV/emimic"; then
  test "$(readlink -f "$RUNTIME_ENV/emimic")" = "$(readlink -f "$PY_ENV")"
else
  ln -s "$PY_ENV" "$RUNTIME_ENV/emimic"
fi
cd "$RUNTIME_ENV"
# shellcheck source=/dev/null
source emimic/bin/activate
cd "$REPO"
export PYTHONPATH=$REPO
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true

if test "$MODE" = smoke; then
  export WANDB_MODE=offline
  WANDB_OFFLINE=true
  WANDB_RESUME=never
else
  export WANDB_MODE=online
  WANDB_OFFLINE=false
  if test "$RESTART_COUNT" = 0; then
    WANDB_RESUME=never
  else
    WANDB_RESUME=allow
  fi
fi

COMMON_OVERRIDES=(
  --config-name=train_zarr_cartesian
  "+experiment=$EXPERIMENT"
  mode=train
  "hydra.run.dir=$RUN_DIR"
  "++paths.root_dir=$RUN_DIR"
  "paths.output_dir=$RUN_DIR"
  "paths.work_dir=$REPO"
  launch_params.gpus_per_node=1
  launch_params.nodes=1
  data.train_dataloader_params.pushshapes_sim_u_socket.batch_size=64
  data.train_dataloader_params.pushshapes_sim_chain_gripper.batch_size=64
  "data.train_datasets.pushshapes_sim_u_socket.resolver.folder_path=$U_DATA"
  "data.valid_datasets.pushshapes_sim_u_socket.resolver.folder_path=$U_DATA"
  "data.train_datasets.pushshapes_sim_chain_gripper.resolver.folder_paths=[$CHAIN_BASE_DATA,$CHAIN_GEN_DATA]"
  "data.valid_datasets.pushshapes_sim_chain_gripper.resolver.folder_paths=[$CHAIN_BASE_DATA,$CHAIN_GEN_DATA]"
  trainer.accumulate_grad_batches=1
  trainer.log_every_n_steps=1
  ++model.train_metrics_on_step=true
  ++model.train_metrics_on_epoch=true
  "logger.wandb.offline=$WANDB_OFFLINE"
  logger.wandb.project=pushshapes-flow-transfer
  logger.wandb.entity=rl2-group
  "logger.wandb.group=$WANDB_GROUP"
  "logger.wandb.id=$WANDB_ID"
  "++logger.wandb.name=$WANDB_ID"
  "++logger.wandb.resume=$WANDB_RESUME"
  norm_stats.norm_mode=minmax
  norm_stats.reduce_all_but_last=true
  norm_stats.sample_frac=1.0
  norm_stats.save_cache_dir=null
  "norm_stats.precomputed_norm_path=$NORM_ARTIFACT"
)

if test "$MODE" = smoke; then
  COMMON_OVERRIDES+=(
    trainer.max_steps=2
    trainer.limit_train_batches=2
    trainer.val_check_interval=1
    trainer.limit_val_batches=1
    trainer.num_sanity_val_steps=0
    callbacks.model_checkpoint.every_n_epochs=null
    callbacks.model_checkpoint.every_n_train_steps=2
    callbacks.model_checkpoint.train_time_interval=null
    callbacks.model_checkpoint.save_on_train_epoch_end=false
    callbacks.terminal_checkpoint.every_n_train_steps=1
  )
else
  COMMON_OVERRIDES+=(
    trainer.max_steps=240000
    trainer.limit_train_batches=1.0
    trainer.val_check_interval=10000
    trainer.limit_val_batches=0
    trainer.num_sanity_val_steps=0
    callbacks.model_checkpoint.every_n_epochs=null
    callbacks.model_checkpoint.every_n_train_steps=null
    callbacks.model_checkpoint.save_on_train_epoch_end=true
    callbacks.terminal_checkpoint.every_n_train_steps=240000
  )
fi

if test "$MODE" = full && test "$RESTART_COUNT" != 0; then
  shopt -s nullglob
  RESUME_CANDIDATES=("$RUN_DIR"/hpc_ckpt_*.ckpt)
  shopt -u nullglob
  if test -s "$RUN_DIR/checkpoints/last.ckpt"; then
    RESUME_CANDIDATES+=("$RUN_DIR/checkpoints/last.ckpt")
  fi
  test "${#RESUME_CANDIDATES[@]}" -gt 0
  RESUME_CKPT=$(python - "${RESUME_CANDIDATES[@]}" <<'PY'
import pathlib
import sys

paths = [pathlib.Path(raw) for raw in sys.argv[1:]]
for path in paths:
    assert path.is_file() and path.stat().st_size > 0, path
print(max(paths, key=lambda path: (path.stat().st_mtime_ns, str(path))))
PY
)
  test -s "$RESUME_CKPT"
  COMMON_OVERRIDES+=("ckpt_path=$RESUME_CKPT")
fi

RESOLVED_CONFIG=$PROVENANCE_DIR/resolved_config.yaml
python -m egomimic.trainHydra "${COMMON_OVERRIDES[@]}" \
  --cfg job --resolve > "$RESOLVED_CONFIG"
python - \
  "$RESOLVED_CONFIG" "$MODE" "$NORM_ARTIFACT" "$RUN_DIR" \
  "$REPO" "$U_DATA" "$CHAIN_BASE_DATA" "$CHAIN_GEN_DATA" \
  "$EXCLUDED_CHAIN_EPISODE" "$WANDB_ID" "$WANDB_GROUP" \
  "$WANDB_OFFLINE" "$WANDB_RESUME" <<'PY'
import pathlib
import sys

from hydra.utils import instantiate
from omegaconf import OmegaConf

(
    path,
    mode,
    norm_artifact,
    run_dir,
    repo,
    u_data,
    chain_base_data,
    chain_gen_data,
    excluded_chain_episode,
    wandb_id,
    wandb_group,
    wandb_offline,
    wandb_resume,
) = sys.argv[1:]
cfg = OmegaConf.load(path)
model = cfg.model.robomimic_model
domains = {"pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper"}
expected_chain_filter = (
    "lambda row: row.get('episode_hash') != "
    f"'{excluded_chain_episode}'"
)

assert cfg.seed == 42
assert cfg.mode == "train"
assert str(cfg.paths.root_dir) == run_dir
assert str(cfg.paths.output_dir) == run_dir
assert str(cfg.paths.work_dir) == repo
assert cfg.trainer.precision == "bf16"
assert cfg.trainer.strategy == "ddp"
assert cfg.trainer.devices == 1
assert cfg.trainer.num_nodes == 1
assert cfg.trainer.sync_batchnorm is True
assert cfg.trainer.accumulate_grad_batches == 1
assert cfg.trainer.log_every_n_steps == 1
assert cfg.trainer.get("gradient_clip_val") is None
assert cfg.launch_params.gpus_per_node == 1
assert cfg.launch_params.nodes == 1
assert cfg.logger.wandb.offline is (wandb_offline == "true")
assert cfg.logger.wandb.project == "pushshapes-flow-transfer"
assert cfg.logger.wandb.entity == "rl2-group"
assert cfg.logger.wandb.group == wandb_group
assert cfg.logger.wandb.id == wandb_id
assert cfg.logger.wandb.resume == wandb_resume
assert cfg.model.enable_grad_norm is False
assert cfg.model.train_metrics_on_step is True
assert cfg.model.train_metrics_on_epoch is True
assert cfg.model.optimizer.lr == 3.0e-5
assert cfg.model.optimizer.weight_decay == 1.0e-4
assert cfg.model.scheduler.max_steps == 240_000
assert cfg.model.scheduler.warmup_steps == 3_000
assert cfg.model.scheduler.warmup_start_factor == 0.1
assert cfg.model.scheduler.eta_min == 3.0e-6
assert cfg.norm_stats.norm_mode == "minmax"
assert cfg.norm_stats.reduce_all_but_last is True
assert cfg.norm_stats.sample_frac == 1.0
assert cfg.norm_stats.save_cache_dir is None
assert str(cfg.norm_stats.precomputed_norm_path) == norm_artifact
assert model._target_ == "egomimic.pipeline.algo.PipelineAlgo"
assert model.action_horizon == 16
assert set(model.domains) == domains
assert set(cfg.data.train_datasets) == domains
assert set(cfg.data.valid_datasets) == domains
assert all(
    dataset.mode == "train" and float(dataset.valid_ratio) == 0.0
    for dataset in cfg.data.train_datasets.values()
)
assert all(
    dataset.mode == "valid" and float(dataset.valid_ratio) == 0.02
    for dataset in cfg.data.valid_datasets.values()
)
for domain in domains:
    assert cfg.data.train_dataloader_params[domain].batch_size == 64
    assert cfg.data.valid_dataloader_params[domain].batch_size == 16
    assert cfg.data.train_datasets[domain].resolver.key_map.action_horizon == 16
    assert cfg.data.valid_datasets[domain].resolver.key_map.action_horizon == 16
assert str(
    cfg.data.train_datasets.pushshapes_sim_u_socket.resolver.folder_path
) == u_data
assert str(
    cfg.data.valid_datasets.pushshapes_sim_u_socket.resolver.folder_path
) == u_data
for split in ("train_datasets", "valid_datasets"):
    chain_dataset = cfg.data[split].pushshapes_sim_chain_gripper
    chain = chain_dataset.resolver
    assert chain._target_.endswith(
        "LocalEpisodeResolverManyWithEmbodimentOverride"
    )
    assert list(chain.folder_paths) == [chain_base_data, chain_gen_data]
    assert chain.key_map.action_zarr_key == "actions"
    assert chain_dataset.filters._target_ == "egomimic.rldb.filters.DatasetFilter"
    assert list(chain_dataset.filters.filter_lambdas) == [expected_chain_filter]

assert model.rollout_adapters.pushshapes_sim_u_socket._target_.endswith(
    "USocketRotVecRolloutAdapter"
)
assert model.rollout_adapters.pushshapes_sim_chain_gripper._target_.endswith(
    "ChainGripperPointRolloutAdapter"
)
assert model.stages[1]._target_.endswith("GaussianLatentNoise")
assert model.stages[1].action_horizon == 16
assert model.stages[1].latent_dim == 96
sampler = model.stages[2]
assert sampler._target_.endswith("MultiJActionSampler")
assert sampler.action_horizon == 16
assert sampler.action_dims.pushshapes_sim_u_socket == 4
assert sampler.action_dims.pushshapes_sim_chain_gripper == 6
assert sampler.latent_dim == 96
assert sampler.decoder_hidden_dim == 512
assert sampler.denoiser_hidden_dim == 384
assert sampler.gradient_accumulation_steps == 1
assert sampler.denoising_module.act_seq == 16
assert sampler.denoising_module.hidden_dim == 384
assert sampler.denoising_module.nblocks == 16
assert "action_encoder" not in OmegaConf.to_yaml(model).lower()
assert cfg.evaluator._target_.endswith("HumanRobotOverlayEval")

terminal = cfg.callbacks.terminal_checkpoint
assert terminal._target_.endswith("ModelCheckpoint")
assert terminal.filename == "step-{step}"
assert terminal.monitor is None
assert terminal.save_top_k == 1
assert terminal.save_last is False
assert terminal.save_on_train_epoch_end is False
assert terminal.save_weights_only is False
assert terminal.enable_version_counter is False
regular = cfg.callbacks.model_checkpoint
assert regular._target_.endswith("ModelCheckpoint")
assert regular.save_last is True

if mode == "smoke":
    assert cfg.trainer.max_steps == 2
    assert cfg.trainer.limit_train_batches == 2
    assert cfg.trainer.val_check_interval == 1
    assert cfg.trainer.limit_val_batches == 1
    assert cfg.trainer.num_sanity_val_steps == 0
    assert regular.every_n_epochs is None
    assert regular.every_n_train_steps == 2
    assert regular.train_time_interval is None
    assert regular.save_on_train_epoch_end is False
    assert terminal.every_n_train_steps == 1
else:
    assert mode == "full"
    assert cfg.trainer.max_steps == 240_000
    assert cfg.trainer.limit_train_batches == 1.0
    assert cfg.trainer.val_check_interval == 10_000
    assert cfg.trainer.limit_val_batches == 0
    assert cfg.trainer.num_sanity_val_steps == 0
    assert regular.every_n_epochs is None
    assert regular.every_n_train_steps is None
    assert regular.train_time_interval._target_ == "datetime.timedelta"
    assert regular.train_time_interval.hours == 1
    assert regular.save_on_train_epoch_end is True
    assert terminal.every_n_train_steps == 240_000

callback_names = []
callback_state_keys = []
for name, callback_cfg in cfg.callbacks.items():
    if str(callback_cfg.get("_target_", "")).endswith("ModelCheckpoint"):
        callback_names.append(name)
        callback_state_keys.append(instantiate(callback_cfg).state_key)
assert set(callback_names) == {"model_checkpoint", "terminal_checkpoint"}
assert len(callback_state_keys) == len(set(callback_state_keys)), callback_state_keys

yaml = pathlib.Path(path).read_text().lower()
assert "arc_length" not in yaml
assert "actions.points" not in yaml
PY

TRAIN_LOG=$RUN_DIR/training.log
"$SRUN" --kill-on-bad-exit=1 --unbuffered \
  python -m egomimic.trainHydra "${COMMON_OVERRIDES[@]}" \
  2>&1 | tee -a "$TRAIN_LOG"

# A live append or artifact mutation invalidates either result.
validate_all_inventories
test "$(sha256sum "$NORM_ARTIFACT" | awk '{print $1}')" = "$EXPECTED_NORM_SHA"
test "$(sha256sum "$NORM_VALIDATION" | awk '{print $1}')" = \
  "$EXPECTED_NORM_VALIDATION_SHA"
test "$(sha256sum "$TRANSPORT_VALIDATION" | awk '{print $1}')" = \
  "$EXPECTED_TRANSPORT_VALIDATION_SHA"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$EXPECTED_LAUNCHER_SHA"

if test "$MODE" = smoke; then
  python "$VERIFIER" "$RUN_DIR" \
    --required-embodiments 19,20 \
    --expected-head "$EXPECTED_HEAD" \
    --expected-world-size "$EXPECTED_WORLD_SIZE" \
    --dry-run
  python "$VERIFIER" "$RUN_DIR" \
    --required-embodiments 19,20 \
    --expected-head "$EXPECTED_HEAD" \
    --expected-world-size "$EXPECTED_WORLD_SIZE"
  "$PY_ENV/bin/python" - \
    "$RUN_DIR/SMOKE_RESULT.json" "$PROVENANCE_DIR/run_validation.json" \
    "$EXPECTED_HEAD" "$EXPECTED_LAUNCHER_SHA" "$EXPECTED_ENV_SHA" \
    "$EXPECTED_NORM_SHA" "$EXPECTED_TRANSPORT_VALIDATION_SHA" \
    "$EXPECTED_U_INVENTORY_SHA" \
    "$EXPECTED_U_EPISODE_METADATA_SHA" \
    "$EXPECTED_CHAIN_BASE_INVENTORY_SHA" \
    "$EXPECTED_CHAIN_BASE_EPISODE_METADATA_SHA" \
    "$EXPECTED_CHAIN_GEN_INVENTORY_SHA" \
    "$EXPECTED_CHAIN_GEN_EPISODE_METADATA_SHA" \
    "$EXPECTED_CHAIN_GEN_EFFECTIVE_INVENTORY_SHA" \
    "$WANDB_ID" "$RESOLVED_CONFIG" <<'PY'
import hashlib
import json
import pathlib
import sys

(
    smoke_result_arg,
    output_arg,
    expected_head,
    launcher_sha,
    environment_sha,
    norm_sha,
    transport_validation_sha,
    u_inventory_sha,
    u_metadata_sha,
    chain_base_inventory_sha,
    chain_base_metadata_sha,
    chain_gen_inventory_sha,
    chain_gen_metadata_sha,
    chain_gen_effective_inventory_sha,
    wandb_id,
    resolved_config_arg,
) = sys.argv[1:]
smoke_result_path = pathlib.Path(smoke_result_arg)
resolved_config_path = pathlib.Path(resolved_config_arg)
smoke_result = json.loads(smoke_result_path.read_text())
assert smoke_result["status"] == "passed"
assert smoke_result["global_step"] == 2
assert smoke_result["scheduler_last_epoch"] == 2
assert smoke_result["world_size"] == 1
assert smoke_result["required_embodiments"] == [19, 20]
assert smoke_result["dense_training_steps"] == [0, 1]
assert smoke_result["validation_trainer_global_step"] >= 1
payload = {
    "status": "PASS",
    "mode": "smoke",
    "arm": "latent_h16",
    "repo_head": expected_head,
    "launcher_sha256": launcher_sha,
    "environment_sha256": environment_sha,
    "norm_sha256": norm_sha,
    "transport_validation_sha256": transport_validation_sha,
    "inventory_sha256": {
        "usocket": u_inventory_sha,
        "chain_base": chain_base_inventory_sha,
        "chain_gen": chain_gen_inventory_sha,
        "chain_gen_effective": chain_gen_effective_inventory_sha,
    },
    "episode_metadata_sha256": {
        "usocket": u_metadata_sha,
        "chain_base": chain_base_metadata_sha,
        "chain_gen": chain_gen_metadata_sha,
    },
    "world_size": 1,
    "global_batch_per_domain": {
        "pushshapes_sim_u_socket": 64,
        "pushshapes_sim_chain_gripper": 64,
    },
    "action_horizon": 16,
    "gpu_model": "H200",
    "partition": "gpu-h200",
    "account": "gts-dxu345-rl2",
    "qos": "inferno",
    "wandb_id": wandb_id,
    "resolved_config": str(resolved_config_path),
    "resolved_config_sha256": hashlib.sha256(
        resolved_config_path.read_bytes()
    ).hexdigest(),
    "smoke_result_path": str(smoke_result_path),
    "smoke_result_sha256": hashlib.sha256(smoke_result_path.read_bytes()).hexdigest(),
    "smoke_result": smoke_result,
}
output = pathlib.Path(output_arg)
temporary = output.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY
  chmod 444 "$PROVENANCE_DIR/run_validation.json"
  printf '[smoke] ALL_PASS arm=%s head=%s world_size=%s gate=%s\n' \
    "$ARM" "$EXPECTED_HEAD" "$EXPECTED_WORLD_SIZE" \
    "$PROVENANCE_DIR/run_validation.json"
else
  "$PY_ENV/bin/python" - \
    "$RUN_DIR" "$PROVENANCE_DIR/run_validation.json" \
    "$EXPECTED_HEAD" "$EXPECTED_LAUNCHER_SHA" "$WANDB_ID" <<'PY'
import hashlib
import json
import math
import pathlib
import sys

import torch

from egomimic.scripts.verify_training_smoke import read_successful_wandb_exit_code

run_dir = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
expected_head = sys.argv[3]
launcher_sha = sys.argv[4]
wandb_id = sys.argv[5]
assert (
    wandb_id
    == "ft_cotrain_newdata3719_latent_dense_m96_h16_s42_"
    "world1_pace_h200_20260828"
)
terminal = run_dir / "checkpoints" / "final" / "step-240000.ckpt"
assert terminal.is_file(), terminal
checkpoint = torch.load(terminal, map_location="cpu", weights_only=False)
assert int(checkpoint["global_step"]) == 240_000
assert checkpoint.get("state_dict")
assert checkpoint.get("optimizer_states")
assert len(checkpoint.get("lr_schedulers", [])) == 1
assert int(checkpoint["lr_schedulers"][0]["last_epoch"]) == 240_000
optimizer_lrs = [
    float(group["lr"])
    for state in checkpoint["optimizer_states"]
    for group in state["param_groups"]
]
assert optimizer_lrs and all(math.isfinite(value) for value in optimizer_lrs)
offline_runs = sorted(run_dir.glob("wandb/offline-run-*"))
assert not offline_runs, offline_runs
wandb_streams = sorted(run_dir.glob(f"wandb/run-*/run-{wandb_id}.wandb"))
assert wandb_streams and all(path.stat().st_size > 0 for path in wandb_streams)
final_wandb_stream = max(wandb_streams, key=lambda path: path.stat().st_mtime_ns)
wandb_exit_code = read_successful_wandb_exit_code(final_wandb_stream)
payload = {
    "status": "PASS",
    "mode": "full",
    "arm": "latent_h16",
    "repo_head": expected_head,
    "launcher_sha256": launcher_sha,
    "wandb_id": wandb_id,
    "wandb_exit_code": wandb_exit_code,
    "final_wandb_stream": str(final_wandb_stream),
    "final_wandb_stream_sha256": hashlib.sha256(
        final_wandb_stream.read_bytes()
    ).hexdigest(),
    "wandb_streams": [str(path) for path in wandb_streams],
    "wandb_stream_sha256": [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in wandb_streams
    ],
    "global_step": 240_000,
    "scheduler_last_epoch": 240_000,
    "optimizer_lrs": optimizer_lrs,
    "terminal_checkpoint": str(terminal),
    "terminal_checkpoint_sha256": hashlib.sha256(terminal.read_bytes()).hexdigest(),
}
temporary = output.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY
  chmod 444 "$PROVENANCE_DIR/run_validation.json"
  printf '[full] ALL_PASS arm=%s head=%s run=%s wandb_id=%s\n' \
    "$ARM" "$EXPECTED_HEAD" "$RUN_DIR" "$WANDB_ID"
fi
