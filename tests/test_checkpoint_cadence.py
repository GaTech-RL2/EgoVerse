from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "egomimic" / "hydra_configs"
PRIORITY_FLOW_TRANSFER_CONFIGS = (
    "pipeline_sampler_usocket_chain_newdata_cotrain12_per_emb_proprio_h16.yaml",
    (
        "pipeline_sampler_usocket_chain_newdata_temporal_h8_l8_w256_d12_"
        "dec64_per_emb_proprio.yaml"
    ),
)


def _load_yaml(path: Path) -> dict:
    with path.open() as stream:
        return yaml.safe_load(stream)


def test_default_checkpoint_is_unranked_20k_step_cadence() -> None:
    config = _load_yaml(CONFIG_ROOT / "callbacks" / "checkpoints.yaml")
    checkpoint = config["model_checkpoint"]

    assert checkpoint["filename"] == "step-{step}"
    assert checkpoint["monitor"] is None
    assert checkpoint["save_top_k"] == -1
    assert checkpoint["save_last"] is True
    assert checkpoint["every_n_epochs"] is None
    assert checkpoint["every_n_train_steps"] == 20_000
    assert checkpoint["train_time_interval"] is None
    assert checkpoint["save_on_train_epoch_end"] is False
    assert checkpoint["auto_insert_metric_name"] is False


def test_priority_flow_transfer_configs_inherit_default_checkpoint() -> None:
    experiment_root = CONFIG_ROOT / "experiment" / "pusht"
    for filename in PRIORITY_FLOW_TRANSFER_CONFIGS:
        config = _load_yaml(experiment_root / filename)
        assert "model_checkpoint" not in config.get("callbacks", {})
