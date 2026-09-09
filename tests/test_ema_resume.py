from types import SimpleNamespace

import pytest
import torch
from lightning import LightningModule

from egomimic.utils.ema_callback import EMACallback


class _TinyModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1, bias=False)

    def log(self, *_args, **_kwargs):
        pass


def _resume_checkpoint(module, *, global_step=20_000, ema_num_updates=20_000):
    return {
        "global_step": global_step,
        "ema_config": {
            "decay": 0.9978,
            "use_warmup": False,
            "update_after_step": 0,
            "inv_gamma": 1.0,
            "power": 0.75,
            "min_decay": 0.0,
        },
        "ema_num_updates": ema_num_updates,
        "ema_state_dict": {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
        },
    }


def test_ema_resume_baselines_before_lightning_restores_loop_progress():
    module = _TinyModule()
    callback = EMACallback(decay=0.9978)
    callback.on_load_checkpoint(
        SimpleNamespace(global_step=0),
        module,
        _resume_checkpoint(module),
    )

    # Lightning can invoke fit-start hooks while trainer.global_step is still
    # zero, then restore loop progress before the next train-batch-end hook.
    trainer = SimpleNamespace(global_step=0)
    callback.on_fit_start(trainer, module)
    trainer.global_step = 20_001
    callback.on_train_batch_end(trainer, module, None, None, 0)

    assert callback.num_updates == 20_001


def test_ema_resume_rejects_update_count_drift():
    module = _TinyModule()
    callback = EMACallback(decay=0.9978)

    with pytest.raises(RuntimeError, match="must equal checkpoint global_step"):
        callback.on_load_checkpoint(
            SimpleNamespace(global_step=0),
            module,
            _resume_checkpoint(module, ema_num_updates=19_999),
        )
