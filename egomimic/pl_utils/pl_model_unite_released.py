"""Lightning wrapper adapter for the released UNITE composite optimizer."""

from typing import Any, Dict

import hydra

from egomimic.pl_utils.pl_model import ModelWrapper


class ReleasedUniteModelWrapper(ModelWrapper):
    """Use the current wrapper lifecycle with named-parameter optimizer input.

    The released Muon/AdamW facade partitions parameters by name, whereas the
    base wrapper supplies only a parameter iterator. All training, validation,
    checkpoint, and metric behavior remains inherited from ``ModelWrapper``.
    """

    def configure_optimizers(self) -> Dict[str, Any]:
        # ``self.model`` is the single owner of trainable policy parameters.
        # Using the Trainer's wrapper-level iterator gives aliases such as
        # ``model.*`` / ``nets.*`` and is unavailable in isolated construction
        # tests.  Model-local names are stable and preserve the concrete
        # ``...encoder.img_encoders...`` path needed by the released grouping.
        named_parameters = tuple(self.model.named_parameters())
        config_tree = getattr(self.hparams, "config_tree", None)
        if config_tree is not None:
            cfg = self._as_config(config_tree)
            optimizer = hydra.utils.instantiate(
                cfg.model.optimizer,
                named_params=named_parameters,
            )
            if callable(optimizer):
                optimizer = optimizer()
            scheduler_cfg = cfg.model.get("scheduler")
            if scheduler_cfg is not None:
                scheduler = hydra.utils.instantiate(
                    scheduler_cfg,
                    optimizer=optimizer,
                )
                if callable(scheduler):
                    scheduler = scheduler()
            else:
                scheduler = None
        else:
            optimizer = self.hparams.optimizer(named_params=named_parameters)
            scheduler = (
                self.hparams.scheduler(optimizer=optimizer)
                if self.hparams.scheduler is not None
                else None
            )

        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.hparams.scheduler_interval,
                    "frequency": self.hparams.scheduler_frequency,
                },
            }
        return {"optimizer": optimizer}
