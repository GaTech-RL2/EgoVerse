"""WSD recipes replace the model's LinearLR through _scheduler_reset, not a merge."""

import hydra
import pytest
import torch
from omegaconf import OmegaConf

WSD = "egomimic.utils.scheduler_utils.warmup_stable_decay"


def _wsd(warmup, stable, decay):
    return {
        "_target_": WSD,
        "_partial_": True,
        "warmup": warmup,
        "stable": stable,
        "decay": decay,
    }


@pytest.mark.parametrize(
    "config_name, overrides, expected",
    [
        ("train_zarr_mecka_freeform_fold_kp_rdt_op2", [], _wsd(1000, 47000, 12000)),
        ("train_zarr_mecka_freeform_fold_kp_rdt_all", [], _wsd(1000, 119000, 30000)),
        # A model swap after the reset must not bring its LinearLR keys back.
        (
            "train_zarr_mecka_freeform_fold_kp_rdt_op2",
            ["model=rdt_bc_mecka_kp_dinov3_vitb"],
            _wsd(1000, 47000, 12000),
        ),
        ("train_zarr_mecka_private_tshirt_opscale_10", [], _wsd(1000, 131000, 33000)),
        (
            "train_zarr_mecka_private_tshirt30m_opscale_base_1",
            [],
            _wsd(500, 35500, 9000),
        ),
        ("train_zarr_mecka_private_tshirt_overfit1", [], _wsd(500, 4500, 1000)),
    ],
)
def test_recipe_scheduler_is_exactly_wsd(
    config_name, overrides, expected, compose_resolve
):
    cfg = compose_resolve(config_name, overrides, keep_hydra=True)
    sched = OmegaConf.to_container(cfg.model.scheduler, resolve=True)
    assert sched == expected

    opt = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1.0)
    lr_sched = hydra.utils.instantiate(cfg.model.scheduler)(optimizer=opt)
    assert isinstance(lr_sched, torch.optim.lr_scheduler.LambdaLR)


def test_launcher_swap_still_wins(compose_resolve):
    """logs/ launch scripts keep passing '~model.scheduler' '+model.scheduler=...'."""
    cfg = compose_resolve(
        "train_zarr_mecka_freeform_fold_kp_rdt_op2",
        [
            "~model.scheduler",
            f"+model.scheduler={{_target_:{WSD},_partial_:true,"
            "warmup:10,stable:20,decay:30}",
        ],
        keep_hydra=True,
    )
    assert OmegaConf.to_container(cfg.model.scheduler) == _wsd(10, 20, 30)
