"""W&B tags for configs under hydra_configs/experiments/<folder>/."""

from omegaconf import OmegaConf

from egomimic.utils.hydra_resolvers import (
    apply_experiment_wandb_tag,
    experiment_wandb_tag,
)


def test_experiment_folder_is_the_tag():
    assert (
        experiment_wandb_tag("experiments/fixed_budget_opscale/rdt_1b_opscale1")
        == "fixed_budget_opscale"
    )
    assert (
        experiment_wandb_tag("experiments/fixed_budget_opscale/nested/run")
        == "fixed_budget_opscale"
    )


def test_configs_outside_experiments_have_no_tag():
    assert experiment_wandb_tag("train_zarr_cartesian") is None
    assert experiment_wandb_tag("experiments/only_a_file") is None
    assert experiment_wandb_tag(None) is None


def test_apply_appends_folder_without_dropping_existing_tags():
    cfg = OmegaConf.create(
        {"logger": {"wandb": {"tags": ["keep-me"], "_target_": "x"}}}
    )
    apply_experiment_wandb_tag(cfg, "experiments/fixed_budget_opscale/rdt_1b_opscale1")
    assert list(cfg.logger.wandb.tags) == ["keep-me", "fixed_budget_opscale"]

    apply_experiment_wandb_tag(cfg, "experiments/fixed_budget_opscale/rdt_1b_opscale1")
    assert list(cfg.logger.wandb.tags) == ["keep-me", "fixed_budget_opscale"]


def test_apply_leaves_other_configs_untagged():
    cfg = OmegaConf.create({"logger": {"wandb": {"tags": []}}})
    apply_experiment_wandb_tag(cfg, "train_zarr_cartesian")
    assert list(cfg.logger.wandb.tags) == []
