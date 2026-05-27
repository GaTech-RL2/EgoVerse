"""Train-set visualization evaluator.

Wraps a concrete EvalVideo (HPTEvalVideo, PIEvalVideo, ...) so the same
forward/metric/viz logic can run a second time against a separate
``train_viz`` dataloader. Videos go to ``<root>/videos_train_viz/`` and
metric keys are prefixed with ``train_viz/`` so they don't collide with the
canonical ``Valid/...`` keys.

Instantiated via Hydra from a config like
``hydra_configs/evaluator/train_viz_pi.yaml``.
"""

from __future__ import annotations

import os

from egomimic.eval.eval_video import EvalVideo


class TrainVizEvalVideo(EvalVideo):
    def __init__(self, base: EvalVideo, limit_val_batches: int = 50):
        self.base = base
        # Forward eval-affecting knobs from the wrapped evaluator so the
        # wrapper's own buffering/flushing logic (inherited from EvalVideo)
        # matches the base's intent. compute_metrics_and_viz is still
        # delegated to base.
        super().__init__(
            limit_val_batches=limit_val_batches,
            viz_func=base.viz_func,
            transform_lists=base.transform_lists,
            one_video_per_task=base.one_video_per_task,
            max_frames_per_task=base.max_frames_per_task,
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

    def compute_metrics_and_viz(self, batch):
        metrics, images_dict = self.base.compute_metrics_and_viz(batch)
        metrics = {f"train_viz/{k}": v for k, v in metrics.items()}
        return metrics, images_dict
