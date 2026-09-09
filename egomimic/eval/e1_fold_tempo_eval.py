"""E1 tempo metrics (E_time, E_arc, d_clock) for the fold speed-spread rows.

Scores every variant against the same un-tokenized 30 Hz ground truth
(``actions_time``, carried by the E1 transform list). Per sample and arm:

  E_time  : RMS xyz error at equal time over the first ``h_match_frames``
            frames — the protocol's primary read (H_match from Step 0).
  E_arc   : RMS xyz error at equal progress, M points over min(D, reach).
  d_clock : RMS error, in seconds, of the time-of-progress at those points.
  paired_mse : the repo's metric, computed the way eval_eef_arcmatch does --
            _mse(pred[:B], gt[:B], PAIRED_COLS) with B = min(len(pred),
            len(gt)), i.e. over ALL actions of the chunk, xyz + gripper of both
            arms, rotation excluded so radians are never summed with metres.
            Physical units against the same raw ground truth for every row, so
            unlike the model's own validation loss (a flow-matching surrogate
            living in each run's own normalized space) it is directly
            comparable across variants and spreads. paired_mse_hmatch is the
            same quantity restricted to the protocol's matched horizon, for
            continuity with E_time; xyz-only splits alongside both.
  Progress parameterization: E_arc and d_clock compare at equal progress, and
  progress is "arc length at the tokenizer's bandwidth" — when the row was
  tokenized with ``progress_smooth_hz`` the ground-truth progress is measured
  on the same low-passed positions (else the smoothed token would be scored
  against a raw polyline that is ≈ 18 % longer per metre and every waypoint
  would look late). E_time and E_time_prog are time-domain and unaffected;
  E_time_prog's horizon always uses the RAW path so every row is scored over
  the same frames.
  E_time_prog : E_time over a fixed PROGRESS horizon instead of a fixed time
            horizon — the frames until the ground truth has covered
            ``prog_horizon_m`` of path (capped at the carried chunk). Added for
            the tempo ablation: under a fixed-time horizon a fast episode is
            scored over more path, so E_time rises with tempo for every row.

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
from egomimic.rldb.zarr.arc_length_tokenizer import (
    _dist_interval_indices,
    cumulative_arc_length,
    resample_by_distance,
)
from egomimic.rldb.zarr.e1_arc_tokenizer import ARM_LAYOUT, TokenizeBimanualArcLengthE1, lowpass_positions

# Same columns the lab's eval_eef_arcmatch uses: xyz + gripper per arm.
PAIRED_COLS = [0, 1, 2, 6, 7, 8, 9, 13]
XYZ_COLS = [0, 1, 2, 7, 8, 9]


def _at_progress(p, cum, s):
    return np.stack([np.interp(s, cum, p[:, k]) for k in range(p.shape[1])], axis=1)


# ---------------------------------------------------------------------------
# Arc-matched scoring -- a line-for-line port of the lab's eval_eef_arcmatch
# (aniketh/arc, 57763c49): per sample and arm the matched span is the SHORTER
# of the two travelled distances, both sides are re-tokenized over it with the
# tokenizer's own interpolators (linear xyz / gripper, SLERP rotation) to M
# waypoints uniform in arc length, and the waypoints are scored on the paired
# columns. Travel-normalized: it asks "is the path shape right" with the amount
# of travel divided out, which is what makes a time-indexed and an arc run
# comparable on one chart. The velocity row is recomputed by the same formula
# on both sides (MEAN_PER_DIM over the covered index range) for the with-vel
# variant. Spans below span_floor_m collapse every waypoint onto the origin
# and score a structural zero, so they are counted rather than hidden.
# ---------------------------------------------------------------------------
ARM_BLOCKS = ((0, 3, 6), (7, 10, 13))  # (xyz offset, ypr offset, gripper index)


def _arm_views(traj):
    for xyz_off, ypr_off, grip_i in ARM_BLOCKS:
        yield traj[:, xyz_off : xyz_off + 3], traj[:, ypr_off : ypr_off + 3], traj[:, grip_i : grip_i + 1]


def arm_travel(traj):
    """(T, 14) -> (2,) total translational arc length per arm, in metres."""
    return np.array([float(cumulative_arc_length(pos)[-1]) for pos, _, _ in _arm_views(traj)])


def match_spans(pred_ti, gt_ti):
    return np.minimum(arm_travel(pred_ti), arm_travel(gt_ti))


def tokenize_span(traj, spans, num_points, dt):
    """Re-tokenize a time-indexed (T, 14) chunk over a per-arm span -> (waypoints (M, 14), velocity (14,))."""
    waypoints = np.zeros((num_points, 14), dtype=np.float64)
    velocity = np.zeros(14, dtype=np.float64)
    for arm, (pos, ypr, grip) in enumerate(_arm_views(traj)):
        cum = cumulative_arc_length(pos)
        end_s = float(min(spans[arm], cum[-1]))
        p, y, g = resample_by_distance(pos, ypr, grip, cum, 0.0, end_s, num_points, start_idx=0)
        o = arm * 7
        waypoints[:, o : o + 3] = p
        waypoints[:, o + 3 : o + 6] = y
        waypoints[:, o + 6] = g[:, 0]
        start_i, end_i = _dist_interval_indices(cum, 0.0, end_s)
        dur = max(end_i - start_i, 1) * dt
        velocity[o : o + 3] = (p[-1] - p[0]) / dur
        velocity[o + 3 : o + 6] = (y[-1] - y[0]) / dur
        velocity[o + 6] = (g[-1, 0] - g[0, 0]) / dur
    return waypoints, velocity


def _mse_cols(a, b, cols):
    return float(np.mean((a[..., cols] - b[..., cols]) ** 2))


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
        prog_horizon_m: float = 0.19,
        progress_smooth_hz: float | None = None,
        velocity_norm: str = "path",
        arc_match_points: int = 100,
        span_floor_m: float = 0.01,
        **kwargs,
    ):
        self.arc_match_points = int(arc_match_points)
        self.span_floor_m = float(span_floor_m)
        kwargs.setdefault("viz_func", None)
        super().__init__(**kwargs)
        if variant not in ("time", "arcmean", "arcvel", "arclogdur", "arcdur"):
            raise ValueError("variant must be time | arcmean | arcvel | arclogdur | arcdur")
        self.prog_horizon_m = float(prog_horizon_m)
        self.progress_smooth_hz = None if progress_smooth_hz in (None, 0, 0.0) else float(progress_smooth_hz)
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
                velocity_norm=velocity_norm,
                velocity_mode={"arcmean": "mean", "arcvel": "profile", "arclogdur": "logdur", "arcdur": "dur"}[variant],
                progress_smooth_hz=progress_smooth_hz,
            )
        self._reset()

    def _reset(self):
        self._sums = {k: 0.0 for k in ("e_time_sq", "e_arc_sq", "d_clock_sq", "e_time_prog_sq")}
        self._arm_sums = {a: {k: 0.0 for k in ("e_time_sq", "e_arc_sq", "d_clock_sq", "e_time_prog_sq")} for a in ("L", "R")}
        self._prog_frames = []
        # Action-space MSEs, accumulated per sample (not per arm).
        self._paired = {k: 0.0 for k in ("paired_mse", "paired_mse_hmatch", "xyz_mse", "xyz_mse_hmatch")}
        self._n_paired = 0
        self._am = {k: 0.0 for k in ("arcmatch_paired_mse", "arcmatch_withvel_paired_mse", "arcmatch_xyz_mse", "arcmatch_span_m", "arcmatch_travel_ratio")}
        self._am_n = 0
        self._am_arms = 0
        self._am_degenerate = 0
        self._n = 0
        self._n_partial = 0
        self._per_chunk = []

    # -- decoding ----------------------------------------------------------
    def _predicted(self, pred: np.ndarray, n_frames: int):
        """Returns per arm: (decoded xyz over n_frames, waypoint path, cum progress, clock at path points)."""
        out = []
        if self.variant == "time":
            for xyz_off, _, _, _ in ARM_LAYOUT:
                path = pred[:, xyz_off : xyz_off + 3]
                out.append((path[:n_frames], path, cumulative_arc_length(path), self.dt * np.arange(len(path))))
            return out
        series = self._detok.detokenize(pred, action_horizon=n_frames)  # (n_frames, 14)
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
                # Lab-style paired MSE: decoded prediction vs raw ground truth,
                # xyz + gripper, rotation excluded. Both arms at once.
                gt_full = gt[b]
                dec = (
                    pred[b][: len(gt_full)]
                    if self.variant == "time"
                    else self._detok.detokenize(pred[b], action_horizon=len(gt_full))
                )
                nb = min(len(dec), len(gt_full))
                nh = min(self.h_match, nb)
                for key, cols, upto in (
                    ("paired_mse", PAIRED_COLS, nb),          # repo definition: all actions
                    ("paired_mse_hmatch", PAIRED_COLS, nh),
                    ("xyz_mse", XYZ_COLS, nb),
                    ("xyz_mse_hmatch", XYZ_COLS, nh),
                ):
                    d = dec[:upto][:, cols] - gt_full[:upto][:, cols]
                    self._paired[key] += float(np.mean(d ** 2))
                self._n_paired += 1
                # Arc-matched (lab metric): both sides over the shorter per-arm travel.
                p_, g_ = dec[:nb], gt_full[:nb]
                spans = match_spans(p_, g_)
                if np.all(np.isfinite(spans)):
                    Mm = self.arc_match_points
                    pw, pv = tokenize_span(p_, spans, Mm, self.dt)
                    gw, gv = tokenize_span(g_, spans, Mm, self.dt)
                    pf = np.concatenate([pw, pv[None, :]], axis=0)
                    gf = np.concatenate([gw, gv[None, :]], axis=0)
                    self._am["arcmatch_paired_mse"] += _mse_cols(pw, gw, PAIRED_COLS)
                    self._am["arcmatch_withvel_paired_mse"] += _mse_cols(pf, gf, PAIRED_COLS)
                    self._am["arcmatch_xyz_mse"] += _mse_cols(pw, gw, XYZ_COLS)
                    self._am["arcmatch_span_m"] += float(np.mean(spans))
                    self._am["arcmatch_travel_ratio"] += float(np.mean(arm_travel(p_) / np.maximum(arm_travel(g_), 1e-6)))
                    self._am_n += 1
                    self._am_arms += spans.size
                    self._am_degenerate += int(np.count_nonzero(spans < self.span_floor_m))
                for a, (arm, (xyz_off, _, _, _)) in enumerate(zip(("L", "R"), ARM_LAYOUT)):
                    gt_xyz = gt[b, :, xyz_off : xyz_off + 3]
                    gt_cum_raw = cumulative_arc_length(gt_xyz)
                    gt_cum = gt_cum_raw if self.progress_smooth_hz is None else cumulative_arc_length(
                        lowpass_positions(gt_xyz, self.progress_smooth_hz, 1.0 / self.dt))
                    xyz_t, path, cum_pred, clock = self._predicted(pred[b], len(gt_xyz))[a]
                    H = min(self.h_match, len(xyz_t), len(gt_xyz))
                    e_time_sq = float(np.mean(np.sum((xyz_t[:H] - gt_xyz[:H]) ** 2, axis=1)))
                    # fixed-progress horizon: frames until the truth has covered prog_horizon_m
                    Hp = int(np.searchsorted(gt_cum_raw, self.prog_horizon_m, side="left")) + 1
                    Hp = max(2, min(Hp, len(xyz_t), len(gt_xyz)))
                    e_time_prog_sq = float(np.mean(np.sum((xyz_t[:Hp] - gt_xyz[:Hp]) ** 2, axis=1)))
                    self._prog_frames.append(Hp)
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
                    for k, v in (("e_time_sq", e_time_sq), ("e_arc_sq", e_arc_sq), ("d_clock_sq", d_clock_sq), ("e_time_prog_sq", e_time_prog_sq)):
                        self._arm_sums[arm][k] += v
                        self._sums[k] += 0.5 * v
                    chunk_vals.append((e_time_sq, e_arc_sq, d_clock_sq, e_time_prog_sq))
                self._n += 1
                self._per_chunk.append(tuple(np.mean(chunk_vals, axis=0)))
            n = max(self._n, 1)
            for k, label in (("e_time_sq", "E_time"), ("e_arc_sq", "E_arc"), ("d_clock_sq", "d_clock"), ("e_time_prog_sq", "E_time_prog")):
                metrics[f"Valid/E1/{label}/{name}"] = torch.tensor(np.sqrt(self._sums[k] / n))
            npd = max(self._n_paired, 1)
            for k in ("paired_mse", "paired_mse_hmatch", "xyz_mse", "xyz_mse_hmatch"):
                metrics[f"Valid/E1/{k}/{name}"] = torch.tensor(self._paired[k] / npd)
            nam = max(self._am_n, 1)
            for k, v in self._am.items():
                metrics[f"Valid/E1/{k}/{name}"] = torch.tensor(v / nam)
            metrics[f"Valid/E1/arcmatch_degenerate_frac/{name}"] = torch.tensor(self._am_degenerate / max(self._am_arms, 1))
        return metrics, {}

    def on_validation_end(self):
        super().on_validation_end()
        n = max(self._n, 1)
        per = np.array(self._per_chunk) if self._per_chunk else np.zeros((0, 4))
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
            "e_time_prog": float(np.sqrt(self._sums["e_time_prog_sq"] / n)),
            **{k: float(v / max(self._n_paired, 1)) for k, v in self._paired.items()},
            "n_paired": int(self._n_paired),
            **{k: float(v / max(self._am_n, 1)) for k, v in self._am.items()},
            "arcmatch_degenerate_frac": float(self._am_degenerate / max(self._am_arms, 1)),
            "arcmatch_points": self.arc_match_points,
            "prog_horizon_m": self.prog_horizon_m,
            "progress_smooth_hz": self.progress_smooth_hz,
            "prog_frames_p50": float(np.median(self._prog_frames)) if self._prog_frames else None,
            "per_arm": {a: {k.replace("_sq", ""): float(np.sqrt(v / n)) for k, v in s.items()} for a, s in self._arm_sums.items()},
            "e_time_p50": float(np.sqrt(np.quantile(per[:, 0], 0.5))) if len(per) else None,
            "e_time_p90": float(np.sqrt(np.quantile(per[:, 0], 0.9))) if len(per) else None,
            "e_time_p99": float(np.sqrt(np.quantile(per[:, 0], 0.99))) if len(per) else None,
        }
        print("E1_TEMPO_RESULT " + json.dumps(out), flush=True)
        if self.results_path is not None:
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            self.results_path.write_text(json.dumps(out, indent=2))
