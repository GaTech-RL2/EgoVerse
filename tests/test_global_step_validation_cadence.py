import warnings
from pathlib import Path

import lightning
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.pytorch import LightningModule, Trainer
from packaging.version import Version
from torch.utils.data import DataLoader, TensorDataset

CONFIG_DIR = Path(__file__).parents[1] / "egomimic" / "hydra_configs"
EXPERIMENT = (
    "pusht/pipeline_sampler_usocket_chain_newdata_r01_l4_energy_score_val01_h16"
)
GROUPED_MSE_OVERRIDE = (
    "model.robomimic_model.stages.8._target_="
    "egomimic.pipeline.stages_sampler.GroupedActionMSELoss"
)


def _compose(extra_overrides=()):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment={EXPERIMENT}", *extra_overrides],
        )


class _CadenceModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.validation_global_steps = []

    def training_step(self, batch, batch_idx):
        del batch, batch_idx
        return (self.weight - 1.0).square()

    def validation_step(self, batch, batch_idx):
        del batch, batch_idx

    def on_validation_start(self):
        self.validation_global_steps.append(int(self.global_step))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _loader(length):
    return DataLoader(TensorDataset(torch.zeros(length, 1)), batch_size=1)


def _trainer(root, *, max_steps):
    return Trainer(
        default_root_dir=root,
        max_steps=max_steps,
        max_epochs=-1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_val_batches=1,
        val_check_interval=3,
        check_val_every_n_epoch=None,
        accumulate_grad_batches=1,
    )


@pytest.mark.parametrize("extra_overrides", [(), (GROUPED_MSE_OVERRIDE,)])
def test_energy_score_pair_resolves_global_step_validation_contract(extra_overrides):
    assert Version(lightning.__version__).release[:3] == (2, 6, 1)
    cfg = _compose(extra_overrides)

    assert cfg.trainer.val_check_interval == 10_000
    assert cfg.trainer.check_val_every_n_epoch is None
    assert cfg.trainer.accumulate_grad_batches == 1


def test_validation_interval_continues_across_epoch_boundaries(tmp_path):
    model = _CadenceModule()
    trainer = _trainer(tmp_path / "cross_epoch", max_steps=10)

    trainer.fit(
        model,
        train_dataloaders=_loader(5),
        val_dataloaders=_loader(1),
    )

    assert model.validation_global_steps == [3, 6, 9]
    assert trainer.global_step == 10


def test_full_state_resume_preserves_next_validation_step(tmp_path):
    model = _CadenceModule()
    first_trainer = _trainer(tmp_path / "before_resume", max_steps=4)
    first_trainer.fit(
        model,
        train_dataloaders=_loader(5),
        val_dataloaders=_loader(1),
    )
    assert model.validation_global_steps == [3]

    checkpoint = tmp_path / "full_state_step4.ckpt"
    first_trainer.save_checkpoint(checkpoint, weights_only=False)

    resumed_model = _CadenceModule()
    resumed_trainer = _trainer(tmp_path / "after_resume", max_steps=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PossibleUserWarning)
        resumed_trainer.fit(
            resumed_model,
            train_dataloaders=_loader(5),
            val_dataloaders=_loader(1),
            ckpt_path=checkpoint,
        )

    assert resumed_model.validation_global_steps == [6]
    assert resumed_trainer.global_step == 8
