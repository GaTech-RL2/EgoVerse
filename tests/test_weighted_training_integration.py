"""Small CPU training runs covering Lightning's distributed mixture handling."""

import json
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from lightning import Callback, Trainer
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import Dataset

from egomimic.pl_utils.pl_data_utils import MultiDataModuleWrapper
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.weighted_dataset import WeightedDatasetSampler
from egomimic.utils.batch_utils import sample_mean


class _TrainingSamples(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {"actions": torch.tensor([[float(index) / 8.0]])}


class _TrainingAlgo:
    def __init__(self):
        self.nets = nn.ModuleDict(
            {"eva_bimanual": nn.Linear(1, 1), "human_bimanual": nn.Linear(1, 1)}
        )
        self.seen = []

    def process_batch_for_training(self, batch):
        self.seen.append(tuple(batch))
        return batch

    def forward_training(self, batch):
        return {
            name: self.nets[name](part["actions"]).square().mean()
            for name, part in batch.items()
        }

    def compute_losses(self, predictions, batch):
        return {
            "action_loss": sample_mean(
                list(predictions.values()),
                [part["actions"].shape[0] for part in batch.values()],
            ),
            **{name + "_loss": loss for name, loss in predictions.items()},
        }

    def log_info(self, info):
        return {"Loss": info["losses"]["action_loss"], **info["losses"]}


class _SamplerTrace(Callback):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_train_epoch_end(self, trainer, module):
        sampler = trainer.train_dataloader.sampler
        assert isinstance(sampler, WeightedDatasetSampler)
        assert sampler.num_replicas == 2
        assert sampler.rank == trainer.global_rank
        path = (
            Path(self.output_dir)
            / f"rank{trainer.global_rank}-epoch{trainer.current_epoch}.json"
        )
        path.write_text(
            json.dumps(
                {
                    "epoch": sampler.epoch,
                    "indices": list(sampler),
                    "seen": module.model.seen,
                }
            )
        )


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="CPU DDP requires Gloo"
)
def test_weighted_training_handles_different_missing_domains_on_two_ranks(tmp_path):
    children = {
        "eva_bimanual": _TrainingSamples(),
        "human_bimanual": _TrainingSamples(),
    }
    data = MultiDataModuleWrapper(
        children,
        {},
        {},
        {},
        dataset_weights={"eva_bimanual": 1, "human_bimanual": 3},
        weighted_dataloader_params={"batch_size": 1, "num_workers": 0},
        samples_per_epoch=12,
        sampling_seed=0,
    )
    model = ModelWrapper(
        robomimic_model=_TrainingAlgo(),
        optimizer=partial(torch.optim.SGD, lr=0.01),
        enable_grad_norm=False,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=2,
        strategy=DDPStrategy(start_method="spawn", find_unused_parameters=True),
        max_epochs=2,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[_SamplerTrace(str(tmp_path))],
        default_root_dir=str(tmp_path),
    )
    trainer.fit(model, datamodule=data)
    traces = {
        (rank, epoch): json.loads(
            (tmp_path / f"rank{rank}-epoch{epoch}.json").read_text()
        )
        for rank in range(2)
        for epoch in range(2)
    }
    assert traces[0, 0]["seen"] != traces[1, 0]["seen"]
    for rank in range(2):
        assert traces[rank, 0]["epoch"] == 0
        assert traces[rank, 1]["epoch"] == 1
        assert traces[rank, 0]["indices"] != traces[rank, 1]["indices"]
    assert torch.isfinite(trainer.callback_metrics["Train/Loss"])


def test_weighted_cotrain_overlay_composes_with_hydra():
    config_dir = str(Path(__file__).resolve().parents[1] / "egomimic" / "hydra_configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="train_zarr_cartesian_pi",
            overrides=[
                "hydra/launcher=basic",
                "data=cotrain_pi_lang",
                "model=pi0.5_cotrain_eva_aria",
                "+experiment=weighted_cotrain",
                "data.dataset_weights.human_bimanual=3.0",
            ],
        )
    assert cfg.data.dataset_weights.human_bimanual == 3.0
    assert cfg.data.weighted_dataloader_params.batch_size == 64
    assert cfg.data.sampling_seed == cfg.seed
    assert cfg.model.robomimic_model.homogeneous_training is True
    assert cfg.trainer.strategy == "ddp_find_unused_parameters_true"
    assert cfg.trainer.sync_batchnorm is False
    for name in ("eva_bimanual", "human_bimanual"):
        assert (
            cfg.data.valid_datasets[name]._target_
            == cfg.data.train_datasets[name]._target_
        )
        assert (
            cfg.data.valid_datasets[name].resolver._target_
            == cfg.data.train_datasets[name].resolver._target_
        )
        assert (
            cfg.data.valid_datasets[name].valid_ratio
            == cfg.data.train_datasets[name].valid_ratio
        )
