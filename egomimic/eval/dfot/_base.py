"""Shared knob-storage + ``video_dir`` boilerplate for the DFoT evaluators.

The DFoT rollout/policy/bundle evaluators all subclass :class:`EvalVideo` and
each re-stored the SAME handful of scalar knobs (``embodiment_name``,
``image_key``, ``recon_loss_n_frames``, ``upscale_to``, ``n_chunk_steps``,
``_video_subdir``) and re-defined the SAME 2-line ``video_dir`` override
(``root_dir()/_video_subdir``). COMBINE B hoists that boilerplate here.

This is a pure storage/path mixin — it adds NO new behaviour and changes NO
resolved attribute value. Each subclass ``__init__`` still declares its own
per-class defaults in its signature and passes them through ``store_dfot_knobs``
explicitly, so every evaluator lands on byte-identical attribute values (the
configs also pass every knob explicitly). ``limit_val_batches`` / ``max_videos``
already live on :class:`EvalVideo` and are forwarded via ``super().__init__``.
"""

from __future__ import annotations

import os


class DFoTVideoEvalMixin:
    """Mixin holding the knobs + ``video_dir`` shared by every DFoT evaluator.

    Subclasses call :meth:`store_dfot_knobs` from their ``__init__`` (after
    ``super().__init__`` has set up the :class:`EvalVideo` base) with their own
    default-bearing arguments. ``upscale_to`` / ``n_chunk_steps`` /
    ``recon_loss_n_frames`` accept ``None`` for evaluators that genuinely don't
    use a given knob (none currently do, but keeps the mixin honest).
    """

    def store_dfot_knobs(
        self,
        *,
        embodiment_name: str,
        image_key: str,
        video_subdir: str,
        recon_loss_n_frames: int | None = None,
        upscale_to: int | None = None,
        n_chunk_steps: int | None = None,
    ) -> None:
        self.embodiment_name = embodiment_name
        self.image_key = str(image_key)
        self._video_subdir = str(video_subdir)
        if recon_loss_n_frames is not None:
            self.recon_loss_n_frames = int(recon_loss_n_frames)
        if upscale_to is not None:
            self.upscale_to = int(upscale_to)
        if n_chunk_steps is not None:
            self.n_chunk_steps = int(n_chunk_steps)

    def video_dir(self):
        return os.path.join(self.root_dir(), self._video_subdir)
