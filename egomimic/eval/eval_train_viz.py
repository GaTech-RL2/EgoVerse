"""Train-set visualization evaluator.

Wraps a concrete EvalVideo (PIEvalVideo, ...) so the same
forward/metric/viz logic can run a second time against a separate
``train_viz`` dataloader. Videos go to ``<root>/videos_train_viz/`` and
metric keys are prefixed with ``train_viz/`` so they don't collide with the
canonical ``Valid/...`` keys.

Instantiated via Hydra from a config like
``hydra_configs/evaluator/train_viz_pi_wristframe_6d.yaml``.
"""

from __future__ import annotations

import os

from egomimic.eval.eval_video import EvalVideo


class TrainVizEvalVideo(EvalVideo):
    def __init__(self, base: EvalVideo, limit_val_batches: int = 50):
        # `base` must be set before super().__init__: the trainer/model
        # property setters fire on the base attribute during construction.
        self.base = base
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
        return os.path.join(self.root_dir(), "videos_train_viz")

    def compute_metrics_and_viz(self, batch, do_viz=True):
        # The M-sample metrics (reverse KL / best-of-M) multiply eval sampling
        # cost; keep them on the canonical valid loader only — the train pass
        # is a spot check, not the metric of record.
        algo = self.base.model
        saved_rkl = getattr(algo, "rkl_samples", 1)
        algo.rkl_samples = 1
        try:
            metrics, images_dict = self.base.compute_metrics_and_viz(
                batch, do_viz=do_viz
            )
        finally:
            algo.rkl_samples = saved_rkl
        metrics = {f"train_viz/{k}": v for k, v in metrics.items()}
        return metrics, images_dict
