"""Train-set visualization evaluator.

Wraps a concrete EvalVideo (PIEvalVideo, HPTEvalVideo, ...) so the same
forward/metric/viz logic can run a second time against a separate
``train_viz`` dataloader. Videos go to ``<root>/videos_train_viz/`` and
metric keys are prefixed with ``train_viz/`` so they don't collide with the
canonical ``Valid/...`` keys.

Instantiated via Hydra from a config like
``hydra_configs/evaluator/train_viz_pi_wristframe_6d.yaml`` and enabled with
``+evaluator@train_viz_evaluator=<that config>`` on a data config that
defines ``train_viz_datasets``.
"""

from __future__ import annotations

import os

from egomimic.eval.eval_video import EvalVideo


class TrainVizEvalVideo(EvalVideo):
    def __init__(
        self, base: EvalVideo, limit_val_batches: int = 50, prefix: str = "train_viz"
    ):
        """``prefix`` names the metric-key prefix and the ``videos_<prefix>/``
        subdirectory; trainHydra reuses this wrapper for the opsplit heads
        (``unseen_op_valid``, and ``seen_op_valid`` via ``data.valid_prefix``)."""
        # `base` must be set before super().__init__: the trainer/model
        # property setters fire on the base attribute during construction.
        self.base = base
        self.prefix = prefix
        super().__init__(
            limit_val_batches=limit_val_batches,
            viz_func=base.viz_func,
            transform_lists=base.transform_lists,
            viz_every_n_epochs=base.viz_every_n_epochs,
            viz_max_batches=base.viz_max_batches,
        )

    @property
    def trainer(self):
        return self._trainer

    @trainer.setter
    def trainer(self, value):
        self._trainer = value
        self.base.trainer = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        self.base.model = value

    def video_dir(self):
        return os.path.join(self.root_dir(), f"videos_{self.prefix}")

    def compute_metrics_and_viz(self, batch, do_viz=True):
        metrics, images_dict = self.base.compute_metrics_and_viz(batch, do_viz=do_viz)
        metrics = {f"{self.prefix}/{k}": v for k, v in metrics.items()}
        return metrics, images_dict
