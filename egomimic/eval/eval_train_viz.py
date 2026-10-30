"""Prefixed evaluator heads.

Wraps a concrete EvalVideo (PIEvalVideo, HPTEvalVideo, ...) so the same
forward/metric/viz logic runs against a second val loader: the train split
(``train_viz``), the unseen-operator split (``unseen_op_valid``) or a renamed
canonical head (``data.valid_prefix``). Metric keys are prefixed with
``<prefix>/`` and videos go to ``<root>/videos_<prefix>/``.

trainHydra builds the train_viz head around the canonical ``evaluator`` by
default; the ``evaluator/train_viz_*.yaml`` configs pin a different ``base``
via ``+evaluator@train_viz_evaluator=<config>``. That ``base`` is composed
separately from the canonical evaluator, so CLI overrides of
``evaluator.viz_every_n_epochs`` / ``viz_max_batches`` do not reach it;
override ``train_viz_evaluator.base.<key>`` as well.
"""

from __future__ import annotations

import os

from egomimic.eval.eval_video import EvalVideo


class PrefixedEvalVideo(EvalVideo):
    """Runs ``base`` on another val loader under a ``prefix`` for metric keys
    and the ``videos_<prefix>/`` directory."""

    def __init__(self, base: EvalVideo, prefix: str = "train_viz"):
        # `base` must be set before super().__init__: the trainer/model
        # property setters fire on the base attribute during construction.
        self.base = base
        self.prefix = prefix
        super().__init__(
            viz_func=base.viz_func,
            transform_lists=base.transform_lists,
            viz_every_n_epochs=base.viz_every_n_epochs,
            viz_max_batches=base.viz_max_batches,
            viz_mode=base.viz_mode,
            replay_chunks=base.replay_chunks,
            replay_trail=base.replay_trail,
            action_stride=base.action_stride,
        )

    def _set_replay_now(self, value: bool) -> None:
        # The base renders the batch, so it must know which mode this step is.
        super()._set_replay_now(value)
        self.base._set_replay_now(value)

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


# Configs and launch scripts written against the original name.
TrainVizEvalVideo = PrefixedEvalVideo
