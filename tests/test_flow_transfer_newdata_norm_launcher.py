from pathlib import Path
from re import DOTALL, findall
from subprocess import run

LAUNCHER = (
    Path(__file__).parents[1]
    / "scripts"
    / "train"
    / "flow_transfer_newdata_h16_norm_precompute.sbatch"
)


def test_newdata_h16_norm_launcher_has_valid_bash_syntax() -> None:
    run(["bash", "-n", str(LAUNCHER)], check=True)


def test_newdata_h16_norm_launcher_embedded_python_compiles() -> None:
    blocks = findall(r"<<'PY'\n(.*?)\nPY", LAUNCHER.read_text(), flags=DOTALL)
    assert len(blocks) >= 5
    for index, block in enumerate(blocks):
        compile(block, f"{LAUNCHER.name}:heredoc-{index}", "exec")


def test_temporal_per_emb_proprio_norm_launcher_is_exact_and_fail_closed() -> None:
    launcher = LAUNCHER.read_text()

    for contract in (
        "EXPECTED_HEAD=${EXPECTED_HEAD:?",
        "EXPECTED_SCRIPT_SHA=${EXPECTED_SCRIPT_SHA:?",
        "EXPECTED_U_COUNT=${EXPECTED_U_COUNT:?",
        "EXPECTED_U_INVENTORY_SHA=${EXPECTED_U_INVENTORY_SHA:?",
        "EXPECTED_U_TRAIN_FRAMES=${EXPECTED_U_TRAIN_FRAMES:?",
        "EXPECTED_CHAIN_BASE_COUNT=${EXPECTED_CHAIN_BASE_COUNT:?",
        "EXPECTED_CHAIN_BASE_INVENTORY_SHA=${EXPECTED_CHAIN_BASE_INVENTORY_SHA:?",
        "EXPECTED_CHAIN_BASE_TRAIN_FRAMES=${EXPECTED_CHAIN_BASE_TRAIN_FRAMES:?",
        "EXPECTED_CHAIN_GEN_COUNT=${EXPECTED_CHAIN_GEN_COUNT:?",
        "EXPECTED_CHAIN_GEN_INVENTORY_SHA=${EXPECTED_CHAIN_GEN_INVENTORY_SHA:?",
        "EXPECTED_CHAIN_GEN_SOURCE_FRAMES=${EXPECTED_CHAIN_GEN_SOURCE_FRAMES:?",
    ):
        assert contract in launcher

    assert "EXPECTED_ACTION_HORIZON=16" in launcher
    assert 'test "$EXPECTED_CHAIN_GEN_COUNT" = 719' in launcher
    assert 'test "$EXPECTED_CHAIN_GEN_SOURCE_FRAMES" = 306591' in launcher
    assert (
        "CHAIN_GEN_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/"
        "chain_gripper_gen_flow_transfer_frozen719_20260829"
    ) in launcher
    assert "EXPECTED_CHAIN_GEN_TRAIN_FRAMES" not in launcher
    assert "EXCLUDED_CHAIN_GEN" not in launcher
    assert "verify_effective_chain_gen" not in launcher
    assert "chain_gen_effective_inventory.txt" not in launcher
    assert '"chain_train_episode_count"' in launcher
    assert '"chain_train_episodes"' in launcher
    assert (
        "COTRAIN12_EXPERIMENT=pusht/pipeline_sampler_usocket_chain_newdata_"
        "cotrain12_per_emb_proprio_h16"
    ) in launcher
    assert (
        "TEMPORAL_EXPERIMENT=pusht/pipeline_sampler_usocket_chain_newdata_"
        "temporal_h8_l8_w256_d12_dec64_per_emb_proprio"
    ) in launcher
    assert (
        "REPO=/coc/flash7/paphiwetsa3/worktrees/"
        "flow-transfer-temporal-compression-20260829"
    ) in launcher
    assert (
        "EXP_ROOT=/coc/flash7/paphiwetsa3/experiments/"
        "flow_transfer_temporal_h8_l8_world2_smokes_20260829"
    ) in launcher
    assert "norm_artifacts/per_emb_proprio_h16" in launcher
    assert "pipeline_diffusion_usocket_chain_newdata_h16" not in launcher
    assert "pipeline_sampler_usocket_chain_newdata_dense_medium_h16" not in launcher
    assert "DP_EXPERIMENT" not in launcher
    assert "LATENT_EXPERIMENT" not in launcher
    assert "ARM=${ARM:?" not in launcher
    assert 'test -z "$(git -C "$REPO" status --porcelain=v1' in launcher
    assert (
        'test "$(sha256sum "$SCRIPT" | awk \'{print $1}\')" '
        '= "$EXPECTED_SCRIPT_SHA"' in launcher
    )
    assert "scan_all before" in launcher
    assert "scan_all after" in launcher
    assert launcher.count("cmp -s") >= 2
    assert "mode=norm_stats" in launcher
    assert '"++paths.root_dir=$OUT"' in launcher
    assert '"paths.output_dir=$OUT/hydra_run"' in launcher
    assert '"paths.work_dir=$REPO"' in launcher
    assert "norm_stats.norm_mode=minmax" in launcher
    assert "norm_stats.reduce_all_but_last=true" in launcher
    assert "norm_stats.sample_frac=1.0" in launcher
    assert "norm_stats.precomputed_norm_path=null" in launcher
    assert "resolved_cotrain12_per_emb_proprio_config.yaml" in launcher
    assert "resolved_temporal_h8_l8_per_emb_proprio_config.yaml" in launcher
    assert "resolved_dp_config.yaml" not in launcher
    assert "resolved_latent_h16_config.yaml" not in launcher
    assert "get_keymap_hpt_per_emb_proprio" in launcher
    assert "get_usocket_rotvec_action_state_transform_list" in launcher
    assert 'assert "filters" not in chain_dataset' in launcher
    assert "DatasetFilter" not in launcher
    assert "episode_T_chain_gripper_obs7_000050" not in launcher
    assert (
        'expected_frames = {"19": u_frames, "20": base_frames + gen_source_frames}'
        in launcher
    )
    assert '"19": {"state_agent_model": 4, "actions": 4}' in launcher
    assert '"20": {"state_agent_model": 6, "actions": 6}' in launcher
    assert "state_agent_obj" not in launcher
    assert "chmod 0444 \\" in launcher
    assert '  "$ART" \\' in launcher
    for inventory in (
        "usocket_inventory.txt",
        "chain_base_inventory.txt",
        "chain_gen_inventory.txt",
    ):
        assert f'"$OUT/provenance/inventories/before/{inventory}"' in launcher
    assert "\nsbatch " not in launcher
