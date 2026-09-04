"""E1 tempo metrics (E_time, E_arc, d_clock) for the fold speed-spread rows.

Scores every variant against the same un-tokenized 30 Hz ground truth
(``actions_time``, carried by the E1 transform list). Per sample and arm:

  E_time  : RMS xyz error at equal time over the first ``h_match_frames``
            frames — the protocol's primary read (H_match from Step 0).
  E_arc   : RMS xyz error at equal progress, M points over min(D, reach).
  d_clock : RMS error, in seconds, of the time-of-progress at those points.

Decoding: time predictions are used as-is; arcmean tokens are walked at their
(path-normed) mean speed by the tokenizer's ``detokenize``; arcvel tokens use
the integral clock. Results accumulate over the validation pass and are written
as JSON to ``results_path``; per-arm and pooled values are logged as metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from egomimic.eval.hpt.eval_hpt import HPTEvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment
from egomimic.rldb.zarr.arc_length_tokenizer import cumulative_arc_length
from egomimic.rldb.zarr.e1_arc_tokenizer import ARM_LAYOUT, TokenizeBimanualArcLengthE1


def _at_progress(p, cum, s):
    return np.stack([np.interp(s, cum, p[:, k]) for k in range(p.shape[1])], axis=1)


class E1FoldTempoEval(HPTEvalVideo):
    def __init__(
        self,
        *,
        variant: str,
        time_key: str = "actions_time",
        D: float = 0.40,
        M: int = 100,
        dt: float = 1.0 / 30.0,
        h_match_frames: int = 40,
        results_path: str | None = None,
        **kwargs,
    ):
        kwargs.setdefault("viz_func", None)
        super().__init__(**kwargs)
        if variant not in ("time", "arcmean", "arcvel"):
            raise ValueError("variant must be time | arcmean | arcvel")
        self.variant = variant
        self.time_key = time_key
        self.D, self.M, self.dt = float(D), int(M), float(dt)
        self.h_match = int(h_match_frames)
        self.results_path = Path(results_path) if results_path else None
        self._detok = None
        if variant != "time":
            self._detok = TokenizeBimanualArcLengthE1(
                min_distance_unit=self.D,
                resampled_vector_length=self.M,
                dt=self.dt,
                velocity_norm="path",
                velocity_mode="mean" if variant == "arcmean" else "profile",
            )
        self._reset()

    def _reset(self):
        self._sums = {k: 0.0 for k in ("e_time_sq", "e_arc_sq", "d_clock_sq")}
        self._arm_sums = {a: {k: 0.0 for k in ("e_time_sq", "e_arc_sq", "d_clock_sq")} for a in ("L", "R")}
        self._n = 0
        self._n_partial = 0
        self._per_chunk = []

    # -- decoding ----------------------------------------------------------
    def _predicted(self, pred: np.ndarray):
        """Returns per arm: (xyz over H frames, waypoint path, cum progress, clock at path points)."""
        out = []
        if self.variant == "time":
            for xyz_off, _, _, _ in ARM_LAYOUT:
                path = pred[:, xyz_off : xyz_off + 3]
                out.append((path[: self.h_match], path, cumulative_arc_length(path), self.dt * np.arange(len(path))))
            return out
        series = self._detok.detokenize(pred, action_horizon=self.h_match)  # (H, 14)
        clocks = self._detok.clock_at_waypoints(pred)
        for k, (xyz_off, _, _, _) in enumerate(ARM_LAYOUT):
            wp = pred[: self.M, xyz_off : xyz_off + 3]
            out.append((series[:, xyz_off : xyz_off + 3], wp, cumulative_arc_length(wp), clocks[k]))
        return out

    # -- lightning hooks ---------------------------------------------------
    def on_validation_start(self):
        super().on_validation_start()
        self._reset()

    @torch.inference_mode()
    def compute_metrics_and_viz(self, batch):
        algo = self.model
        preds = algo.forward_eval(batch)
        metrics = {}
        for embodiment_id, _batch in batch.items():
            _batch = algo.norm_stats.unnormalize(_batch, embodiment_id)
            name = get_embodiment(embodiment_id).lower()
            ac_key = algo.ac_keys[embodiment_id]
            pk = f"{name}_{ac_key}"
            if pk not in preds or preds[pk] is None or self.time_key not in _batch:
                continue
            loss_key = f"{name}_loss"
            if loss_key in preds:
                metrics[f"Valid/{loss_key}"] = preds[loss_key]
            pred = preds[pk].detach().float().cpu().numpy().astype(np.float64)
            gt = _batch[self.time_key].detach().float().cpu().numpy().astype(np.float64)
            for b in range(pred.shape[0]):
                chunk_vals = []
                for a, (arm, (xyz_off, _, _, _)) in enumerate(zip(("L", "R"), ARM_LAYOUT)):
                    gt_xyz = gt[b, :, xyz_off : xyz_off + 3]
                    gt_cum = cumulative_arc_length(gt_xyz)
                    xyz_t, path, cum_pred, clock = self._predicted(pred[b])[a]
                    H = min(self.h_match, len(xyz_t), len(gt_xyz))
                    e_time_sq = float(np.mean(np.sum((xyz_t[:H] - gt_xyz[:H]) ** 2, axis=1)))
                    reach = min(self.D, float(gt_cum[-1]), float(cum_pred[-1]))
                    if reach <= 1e-6:
                        e_arc_sq = float(np.mean(np.sum((path[:1] - gt_xyz[:1]) ** 2, axis=1)))
                        d_clock_sq = 0.0
                    else:
                        u = np.linspace(0.0, reach, self.M)
                        e_arc_sq = float(np.mean(np.sum((_at_progress(path, cum_pred, u) - _at_progress(gt_xyz, gt_cum, u)) ** 2, axis=1)))
                        gt_t_u = np.interp(u, gt_cum, np.arange(len(gt_cum))) * self.dt
                        pred_t_u = np.interp(u, cum_pred, clock)
                        d_clock_sq = float(np.mean((pred_t_u - gt_t_u) ** 2))
                    self._n_partial += int(reach < self.D)
                    for k, v in (("e_time_sq", e_time_sq), ("e_arc_sq", e_arc_sq), ("d_clock_sq", d_clock_sq)):
                        self._arm_sums[arm][k] += v
                        self._sums[k] += 0.5 * v
                    chunk_vals.append((e_time_sq, e_arc_sq, d_clock_sq))
                self._n += 1
                self._per_chunk.append(tuple(np.mean(chunk_vals, axis=0)))
            n = max(self._n, 1)
            for k, label in (("e_time_sq", "E_time"), ("e_arc_sq", "E_arc"), ("d_clock_sq", "d_clock")):
                metrics[f"Valid/E1/{label}/{name}"] = torch.tensor(np.sqrt(self._sums[k] / n))
        return metrics, {}

    def on_validation_end(self):
        super().on_validation_end()
        n = max(self._n, 1)
        per = np.array(self._per_chunk) if self._per_chunk else np.zeros((0, 3))
        out = {
            "variant": self.variant,
            "n_chunks": int(self._n),
            "n_partial_arm_chunks": int(self._n_partial),
            "h_match_frames": self.h_match,
            "D": self.D,
            "M": self.M,
            "e_time": float(np.sqrt(self._sums["e_time_sq"] / n)),
            "e_arc": float(np.sqrt(self._sums["e_arc_sq"] / n)),
            "d_clock": float(np.sqrt(self._sums["d_clock_sq"] / n)),
            "per_arm": {a: {k.replace("_sq", ""): float(np.sqrt(v / n)) for k, v in s.items()} for a, s in self._arm_sums.items()},
            "e_time_p50": float(np.sqrt(np.quantile(per[:, 0], 0.5))) if len(per) else None,
            "e_time_p90": float(np.sqrt(np.quantile(per[:, 0], 0.9))) if len(per) else None,
            "e_time_p99": float(np.sqrt(np.quantile(per[:, 0], 0.99))) if len(per) else None,
        }
        print("E1_TEMPO_RESULT " + json.dumps(out), flush=True)
        if self.results_path is not None:
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            self.results_path.write_text(json.dumps(out, indent=2))
