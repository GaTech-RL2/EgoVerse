"""ModelWrapper hooks must not require a torch.distributed process group.

The CPU smoke test (``trainer.accelerator=cpu devices=1``) never initialises a
process group, and every hook that unconditionally calls
``torch.distributed.barrier()`` crashes there with
``RuntimeError: Default process group has not been initialized``.
"""

import torch
import torch.nn as nn

from egomimic.pl_utils.pl_model import ModelWrapper


class _StubAlgo:
    """Minimal stand-in for an ``Algo``: just enough for ``ModelWrapper.__init__``."""

    def __init__(self):
        self.nets = nn.ModuleDict({"policy": nn.Linear(1, 1)})
        self.device = None


def _wrapper() -> ModelWrapper:
    assert not torch.distributed.is_initialized()
    return ModelWrapper(robomimic_model=_StubAlgo())


def test_on_fit_start_runs_without_process_group():
    _wrapper().on_fit_start()


def test_on_validation_end_runs_without_process_group():
    _wrapper().on_validation_end()
