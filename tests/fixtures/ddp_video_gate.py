"""Run a 2-rank DDP fit where only rank 0 iterates the video loader.

Launched as a subprocess by tests/unit/test_eval_loaders.py. Each rank writes
what it saw to ``<out>/rank<N>.json``; the test reads them back. Module scope
(not a closure) so the ddp launcher can re-import it.
"""

from __future__ import annotations

import json
import multiprocessing as mp
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
    """Counts reads in shared memory: the loader workers are other processes."""

    def __init__(self, n: int):
        self.n = n
        self.reads = mp.Value("i", 0)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        with self.reads.get_lock():
            self.reads.value += 1
        return {"frame": np.float32(i)}


class _Module(LightningModule):
    """Mimics ModelWrapper's collective pattern: sync_dist metrics off the
    metric loader, nothing at all off the video loader."""

    def __init__(self, pinned):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.pinned = pinned
        self.video_steps: list[int] = []

    def _val_heads(self):
        return {"valid": _Viz(self)}

    def training_step(self, batch, batch_idx):
        return self.layer(batch[HUMAN]["frame"].reshape(-1, 1)).sum()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 1:
            self.video_steps.append(self.current_epoch)
            return  # the video loader logs nothing, as EvalVideo does
        self.log("val/mse", torch.tensor(1.0), sync_dist=True, add_dataloader_idx=False)

    def on_validation_end(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)


class _Viz:
    def __init__(self, module):
        self.module = module

    def _should_viz(self) -> bool:
        return self.module.current_epoch == 1


def main(out_dir: str) -> None:
    pinned, metric = _Episode(8), _Episode(8)
    dm = MultiDataModuleWrapper(
        train_datasets={
            HUMAN: MultiDataset(datasets={"ep0": _Episode(8)}, mode="total")
        },
        valid_datasets={HUMAN: MultiDataset(datasets={"ep0": metric}, mode="total")},
        train_dataloader_params={HUMAN: {"batch_size": 2, "num_workers": 0}},
        # num_workers > 0 on purpose: with 0 a DataLoader prefetches nothing, so
        # a gate that still builds an iterator would read no frame and look clean.
        valid_dataloader_params={
            HUMAN: {"batch_size": 2, "num_workers": 2, "shuffle": False}
        },
        video_datasets={
            "valid": {HUMAN: MultiDataset(datasets={"pin": pinned}, mode="total")}
        },
    )
    model = _Module(pinned)
    trainer = Trainer(
        strategy="ddp",
        # force the subprocess launcher: run under `srun pytest` lightning would
        # otherwise detect SLURM and wait for ranks srun was never asked for
        plugins=[LightningEnvironment()],
        accelerator="cpu",
        devices=2,
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=2,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=out_dir,
    )
    trainer.fit(model, datamodule=dm)
    Path(out_dir, f"rank{trainer.global_rank}.json").write_text(
        json.dumps(
            {
                "rank": trainer.global_rank,
                "video_steps": model.video_steps,
                "pinned_reads": pinned.reads.value,
                "finished": True,
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
