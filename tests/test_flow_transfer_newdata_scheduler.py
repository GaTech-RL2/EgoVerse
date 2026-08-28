import math
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

CONFIG_DIR = Path(__file__).parents[1] / "egomimic" / "hydra_configs"

NEWDATA_H16_EXPERIMENTS = [
    "pusht/pipeline_sampler_usocket_chain_newdata_dense_medium_h16",
    "pusht/pipeline_diffusion_usocket_chain_newdata_h16",
    "pusht/pipeline_sampler_chain_gripper_newdata_points_dense_medium_h16",
    "pusht/pipeline_diffusion_chain_gripper_newdata_points_h16",
]

OLD_OBSTACLE_EXPERIMENTS = [
    "pusht/pipeline_sampler_usocket_chain_obstacle_dense_medium",
    "pusht/pipeline_diffusion_usocket_chain_obstacle_h16",
]


def _compose(experiment: str):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment={experiment}"],
        )


def _traverse_scheduler(cfg, requested_steps: set[int]) -> dict[int, float]:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = instantiate(cfg.model.optimizer)([parameter])
    scheduler = instantiate(cfg.model.scheduler)(optimizer)

    lrs = {0: optimizer.param_groups[0]["lr"]}
    # The scheduler only depends on completed-step count. One real optimizer
    # step establishes PyTorch's required call order; repeating AdamW updates
    # on a dummy scalar 240k times would test the optimizer, not this curve.
    optimizer.step()
    for step in range(1, max(requested_steps) + 1):
        scheduler.step()
        if step in requested_steps:
            lrs[step] = optimizer.param_groups[0]["lr"]
    return lrs


@pytest.mark.parametrize("experiment", NEWDATA_H16_EXPERIMENTS)
def test_newdata_h16_scheduler_follows_scaled_warmup_cosine_curve(
    experiment: str,
) -> None:
    cfg = _compose(experiment)

    assert cfg.model.optimizer.lr == pytest.approx(3.0e-5)
    assert cfg.model.scheduler.warmup_steps == 3_000
    assert cfg.model.scheduler.max_steps == 240_000
    assert cfg.model.scheduler.warmup_start_factor == pytest.approx(0.1)
    assert cfg.model.scheduler.eta_min == pytest.approx(3.0e-6)

    requested_steps = {0, 3_000, 120_000, 240_000}
    lrs = _traverse_scheduler(cfg, requested_steps)
    expected_midpoint = (
        3.0e-6
        + (3.0e-5 - 3.0e-6)
        * (1.0 + math.cos(math.pi * (120_000 - 3_000) / (240_000 - 3_000)))
        / 2.0
    )

    assert lrs[0] == pytest.approx(3.0e-6, rel=1.0e-9, abs=1.0e-12)
    assert lrs[3_000] == pytest.approx(3.0e-5, rel=1.0e-9, abs=1.0e-12)
    assert lrs[120_000] == pytest.approx(
        expected_midpoint,
        rel=1.0e-9,
        abs=1.0e-12,
    )
    assert lrs[240_000] == pytest.approx(3.0e-6, rel=1.0e-9, abs=1.0e-12)


@pytest.mark.parametrize("experiment", OLD_OBSTACLE_EXPERIMENTS)
def test_old_obstacle_scheduler_defaults_are_unchanged(experiment: str) -> None:
    cfg = _compose(experiment)

    assert cfg.model.optimizer.lr == pytest.approx(1.0e-4)
    assert cfg.model.scheduler.eta_min == pytest.approx(1.0e-5)
