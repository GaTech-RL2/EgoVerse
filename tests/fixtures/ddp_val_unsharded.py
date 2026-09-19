"""Run a 2-rank DDP validation over a single val head and record its frames.

Launched as a subprocess by tests/unit/test_val_loader_sharding.py. Each rank
writes the frames it validated to ``<out>/rank<N>.json``. Module scope (not a
closure) so the ddp launcher can re-import it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from lightning import LightningModule, Trainer
from lightning.fabric.plugins.environments import LightningEnvironment

from egomimic.pl_utils.pl_data_utils import MultiDataModuleWrapper
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

HUMAN = "human_bimanual"


class _Episode:
    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"frame": np.float32(i)}


class _Module(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.seen: list[float] = []

    def training_step(self, batch, batch_idx):
        return self.layer(batch[HUMAN]["frame"].reshape(-1, 1)).sum()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, tuple):  # a list of CombinedLoaders yields triples
            batch = batch[0]
        self.seen.extend(float(f) for f in batch[HUMAN]["frame"])

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)


def main(out_dir: str) -> None:
    train_params = {HUMAN: {"batch_size": 2, "num_workers": 0}}
    valid_params = {HUMAN: {"batch_size": 2, "num_workers": 0, "shuffle": False}}
    dm = MultiDataModuleWrapper(
        train_datasets={
            HUMAN: MultiDataset(datasets={"ep0": _Episode(4)}, mode="total")
        },
        valid_datasets={
            HUMAN: MultiDataset(datasets={"ep0": _Episode(8)}, mode="total")
        },
        train_dataloader_params=train_params,
        valid_dataloader_params=valid_params,
    )
    model = _Module()
    trainer = Trainer(
        strategy="ddp",
        # force the subprocess launcher: under `srun pytest` lightning would
        # otherwise detect SLURM and wait for ranks srun was never asked for
        plugins=[LightningEnvironment()],
        accelerator="cpu",
        devices=2,
        max_epochs=1,
        limit_train_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=out_dir,
    )
    trainer.fit(model, datamodule=dm)
    Path(out_dir, f"rank{trainer.global_rank}.json").write_text(
        json.dumps({"rank": trainer.global_rank, "seen": model.seen})
    )


if __name__ == "__main__":
    main(sys.argv[1])
