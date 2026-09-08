"""Closed-loop sim eval for ARC-TOKENIZED HPT policies.

Why a separate class from ``HPTSimEval``: the two variants execute different
things. The time-chunked baseline predicts ``action_horizon`` per-frame targets
and the env consumes them one per step — chunk slot k IS frame k. The arc-tok
policy predicts ``(M + 1, A)`` tokens that carry NO time base: M waypoints
spaced uniformly in ARC LENGTH plus one velocity token. Those cannot be fed to
the env as-is; they must be DETOKENIZED first — read the velocity token, turn
the chunk's chord into a duration, and resample the M waypoints onto the 30 Hz
control grid (``expand_arc_chunk_to_time`` in
``egomimic/rldb/zarr/pushshapes_arc_tokenizer.py``, the exact inverse of the
dataloader's ``TokenizePushShapesArcLength``).

The detokenizer itself lives in the algo's ``inference_step`` (one
implementation, shared with any other rollout driver, and the algo is the only
place that knows its own action normalization). What this class adds is the
arc-specific contract and telemetry around it:

  * **Fails loudly on a config mismatch.** An arc checkpoint whose ``arc_tok``
    block is missing, or whose ``min_distance_unit`` / ``dt`` disagree with the
    values this evaluator was configured with, would still roll out — silently,
    at the wrong speed, producing a coverage number that looks comparable to
    the baseline and is not. Both are hard errors here.
  * **Reports the detok geometry.** ``Valid/emb*_arc_frames_per_chunk``,
    ``*_arc_chunk_duration_s``, ``*_arc_replans`` and ``*_arc_maxframes_frac``
    expose what the policy actually executed. These are the diagnostics for the
    failure mode that a coverage number alone hides: a near-zero predicted
    velocity inflates the schedule until it clips at ``max_frames`` and the
    pusher crawls, so the rollout stalls without ever producing a bad action.

Use with ``--eval-class arc`` (``egomimic/eval/core/ckpt_loading.py``) or as
``evaluator=hpt/sim_pusht_arc`` inside a Hydra run.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from egomimic.eval.core.eval_sim import HPTSimEval


class ArcSimEval(HPTSimEval):
    """Closed-loop rollout of an arc-tok HPT policy, with detok diagnostics.

    Args:
        arc_min_distance_unit: D (pixels) the run tokenized with. Must equal the
            algo's ``arc_tok.min_distance_unit``. Declared here so a launcher
            can assert the comparison grid instead of trusting the checkpoint.
        arc_dt: control period the detokenizer resamples onto (1/30 s).
        arc_points: M, waypoints per chunk (chunk is (M+1, A) with the velocity
            token). Checked against the policy's ``action_horizon``.
        require_arc_tok: keep True. False is an escape hatch for rolling out a
            NON-arc checkpoint through this class (diagnostics degrade to the
            time-chunked buffer length) — never for an arc-vs-nonarc report.
    """

    def __init__(
        self,
        *args,
        arc_min_distance_unit: float = 100.0,
        arc_dt: float = 1.0 / 30.0,
        arc_points: int = 40,
        require_arc_tok: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.arc_min_distance_unit = float(arc_min_distance_unit)
        self.arc_dt = float(arc_dt)
        self.arc_points = int(arc_points)
        self.require_arc_tok = bool(require_arc_tok)
        self._arc_checked = False
        self._arc_stats: List[Dict[str, float]] = []

    # ----- contract check ----- #

    def _check_arc_contract(self) -> None:
        """Verify the loaded algo really detokenizes, on this evaluator's grid."""
        if self._arc_checked:
            return
        self._arc_checked = True
        algo = self.model
        arc_tok = getattr(algo, "arc_tok", None)
        if arc_tok is None:
            if self.require_arc_tok:
                raise RuntimeError(
                    f"{type(self).__name__} requires an arc-tok algo, but "
                    f"{type(algo).__name__}.arc_tok is None. The checkpoint's "
                    f"model config has no ``arc_tok`` block (see "
                    f"hydra_configs/model/pusht_arc/base.yaml), so "
                    f"inference_step would execute the (M+1, A) token chunk "
                    f"directly as per-frame targets — including the velocity "
                    f"token as a position. Use --eval-class hpt for a "
                    f"time-chunked checkpoint."
                )
            return

        d_algo = float(arc_tok["min_distance_unit"])
        if abs(d_algo - self.arc_min_distance_unit) > 1e-6:
            raise ValueError(
                f"arc D mismatch: algo arc_tok.min_distance_unit={d_algo} but "
                f"evaluator arc_min_distance_unit={self.arc_min_distance_unit}. "
                f"D sets the arc span of one chunk; rolling out on a different D "
                f"than the run trained with makes the rollout — and any "
                f"comparison against the baseline — meaningless."
            )
        dt_algo = float(arc_tok.get("dt", 1.0 / 30.0))
        # 1e-4 tolerance, not exact equality: the configs write 30 Hz as the
        # rounded 0.033333 while the code default is 1/30, and those are the
        # same control rate. A real mismatch (60 Hz vs 30 Hz) is ~0.017 apart.
        if abs(dt_algo - self.arc_dt) > 1e-4:
            raise ValueError(
                f"arc dt mismatch: algo arc_tok.dt={dt_algo} but evaluator "
                f"arc_dt={self.arc_dt}. dt is the control period the "
                f"detokenizer resamples onto; a mismatch replays every chunk at "
                f"the wrong speed."
            )

        # (M+1, A) chunk -> action_horizon must be M+1.
        horizon = getattr(self.model.nets["policy"], "action_horizon", None)
        if horizon is not None and int(horizon) != self.arc_points + 1:
            raise ValueError(
                f"arc M mismatch: policy.action_horizon={int(horizon)} but "
                f"evaluator arc_points={self.arc_points} implies "
                f"{self.arc_points + 1} (M waypoints + 1 velocity token)."
            )
        print(
            f"[arc-sim] detok on: D={d_algo} px  dt={dt_algo:.6f}s  M={self.arc_points}"
            f"  max_frames={int(arc_tok.get('max_frames', 200))}"
        )

    # ----- rollout with detok telemetry ----- #

    def _rollout_one_impl(
        self,
        env_init_dict: dict | None,
        emb_id: int,
        ep_idx: int,
    ) -> Tuple[float, List[np.ndarray]]:
        """Same env loop as the base class, wrapped to record each replan.

        The algo buffers one detokenized chunk at a time in ``_sim_state``. We
        watch that buffer across ``inference_step`` calls: every time a NEW
        buffer object appears, a replan happened and its length is the frame
        count the detokenizer produced for that chunk.
        """
        self._check_arc_contract()
        algo = self.model
        arc_tok = getattr(algo, "arc_tok", None) or {}
        max_frames = int(arc_tok.get("max_frames", 200))

        buffer_lens: List[int] = []
        seen_buffer_id = None
        inner = algo.inference_step

        def recording_inference_step(obs_zarr, t, emb, T_max=None):
            nonlocal seen_buffer_id
            action = inner(obs_zarr, t, emb, T_max=T_max)
            state = getattr(algo, "_sim_state", None) or {}
            buf = state.get("action_chunk_world")
            if buf is not None and id(buf) != seen_buffer_id:
                seen_buffer_id = id(buf)
                buffer_lens.append(int(buf.shape[0]))
            return action

        algo.inference_step = recording_inference_step
        try:
            cov, frames = super()._rollout_one_impl(env_init_dict, emb_id, ep_idx)
        finally:
            algo.inference_step = inner

        if buffer_lens:
            lens = np.asarray(buffer_lens, dtype=np.float64)
            self._arc_stats.append(
                {
                    "frames_per_chunk": float(lens.mean()),
                    "chunk_duration_s": float(lens.mean() * self.arc_dt),
                    "replans": float(lens.size),
                    # Fraction of chunks that hit the max_frames clamp — the
                    # signature of a near-zero predicted velocity.
                    "maxframes_frac": float((lens >= max_frames).mean()),
                }
            )
        return cov, frames

    def compute_metrics_and_viz(
        self, batch: Dict[int, Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, np.ndarray]]:
        device = self.trainer.lightning_module.device
        metrics: Dict[str, torch.Tensor] = {}
        images_dict: Dict[int, np.ndarray] = {}
        merged_images: Dict[int, np.ndarray] = {}

        # Per-embodiment so the arc stats are attributed to the right emb id
        # (the base class loops internally and would mix them).
        merged_ep_frames: Dict[int, Any] = {}
        merged_ep_cov: Dict[int, Any] = {}
        for emb_id, _batch in batch.items():
            self._arc_stats = []
            sub_metrics, sub_images = super().compute_metrics_and_viz({emb_id: _batch})
            metrics.update(sub_metrics)
            merged_images.update(sub_images)
            # The base class resets _last_per_ep_* on every call, so accumulate
            # them here — the per-episode video/coverage dump must cover every
            # embodiment, not just the last one looped over.
            merged_ep_frames.update(getattr(self, "_last_per_ep_frames", {}))
            merged_ep_cov.update(getattr(self, "_last_per_ep_coverages", {}))
            if self._arc_stats:
                for field, suffix in (
                    ("frames_per_chunk", "arc_frames_per_chunk"),
                    ("chunk_duration_s", "arc_chunk_duration_s"),
                    ("replans", "arc_replans"),
                    ("maxframes_frac", "arc_maxframes_frac"),
                ):
                    val = float(np.mean([s[field] for s in self._arc_stats]))
                    metrics[f"Valid/emb{emb_id}_{suffix}"] = torch.tensor(
                        val, device=device
                    )
                print(
                    f"[arc-sim] emb{emb_id} detok: "
                    f"frames/chunk={metrics[f'Valid/emb{emb_id}_arc_frames_per_chunk']:.1f}  "
                    f"duration={metrics[f'Valid/emb{emb_id}_arc_chunk_duration_s']:.2f}s  "
                    f"replans/ep={metrics[f'Valid/emb{emb_id}_arc_replans']:.1f}  "
                    f"maxframes_frac={metrics[f'Valid/emb{emb_id}_arc_maxframes_frac']:.2f}"
                )
        self._last_per_ep_frames = merged_ep_frames
        self._last_per_ep_coverages = merged_ep_cov
        images_dict.update(merged_images)
        return metrics, images_dict
