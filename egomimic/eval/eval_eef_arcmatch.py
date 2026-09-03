"""Arc-matched validation in the EEF frame, for arc and non-arc runs alike.

An arc-tokenized run predicts (M+1, 14) -- M waypoints spaced by arc length plus
a velocity token -- while a time-indexed run predicts (T, 14) at a fixed control
rate. Their native per-timestep MSEs are not comparable: over the same number of
rows the two cover different amounts of travel. Every metric here therefore goes
through ``arc_matched_resample``: clip to the first D metres of travel, resample
to N points spaced uniformly in arc length, then score. Both variants land on the
same grid, and the metric NAMES carry no hint of which produced them, so an arc
run and a baseline run plot on one axis.

Arc detection is automatic and by SEQUENCE LENGTH. ``ArcTokEvalVideo`` keys
detokenization off the last dim being 14, but a time-indexed chunk is 14-wide
too, so it would try to detokenize one and die on the ``shape[1] == M+1`` check.
Keying off the length lets one evaluator serve both run types.

Metrics are reported in the EEF frame only, and every one of them is invariant
to which frame that is. Position error survives the revert to camera frame
because that revert is a rigid transform applied identically to prediction and
ground truth, so |R(p) - R(g)| == |p - g| -- measured, eef and cam agree to
~2e-11. Rotation error is reported as a GEODESIC angle, |log(R_pred . R_gt^T)|,
which is invariant for the same reason (conjugation preserves the trace) and is
free of the wraparound and gimbal artifacts that make a raw ypr difference
frame-dependent and therefore uncomparable across runs.

``transform_lists`` is still used, for the val VIDEO: the overlay projects
through K and needs camera-frame poses to sit on the hands/grippers.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from egomimic.eval.eval_arctok import ArcTokEvalVideo
from egomimic.eval.hpt.eval_hpt import arc_matched_resample
from egomimic.rldb.embodiment.embodiment import get_embodiment

# arc_matched_resample(include_rotation=True) returns (N, 14) per-arm blocks of
# [xyz(3), ypr(3), grip(1)] -- the same layout as a 14-dim cartesian chunk.
RESAMPLED_ARM_XYZ = ((0, 1, 2), (7, 8, 9))
RESAMPLED_ARM_YPR = ((3, 4, 5), (10, 11, 12))
RESAMPLED_XYZ = [c for arm in RESAMPLED_ARM_XYZ for c in arm]
# paired_mse scores position + gripper only, matching the 8-dim form this used
# to resample, so its value is unchanged by rotation being carried through.
RESAMPLED_PAIRED = [0, 1, 2, 6, 7, 8, 9, 13]


def _geodesic_deg(P: torch.Tensor, G: torch.Tensor) -> float:
    """Mean geodesic angle in degrees between two (B, N, 14) resampled chunks.

    The geodesic distance between two orientations is the angle of the single
    rotation carrying one to the other -- the shortest path on SO(3), the
    minimum the gripper would have to turn to be correct. Unlike a per-axis ypr
    difference it is bounded in [0, 180], never wraps, is unaffected by gimbal
    lock, and is unchanged by expressing both rotations in a different frame.
    """
    per_arm = []
    for cols in RESAMPLED_ARM_YPR:
        rp = R.from_euler("ZYX", _ypr_rows(P, cols))
        rg = R.from_euler("ZYX", _ypr_rows(G, cols))
        per_arm.append(float(np.degrees((rp * rg.inv()).magnitude()).mean()))
    return float(np.mean(per_arm))


def _ypr_rows(T: torch.Tensor, cols) -> np.ndarray:
    return T[..., list(cols)].reshape(-1, 3).cpu().numpy().astype(np.float64)


class EefArcMatchEval(ArcTokEvalVideo):
    """Arc-matched EEF-frame metrics; accepts arc-tokenized and time-indexed runs."""

    def _is_arc(self, t) -> bool:
        """True when a chunk is arc tokens rather than time steps.

        Both are 14 wide, so width cannot tell them apart -- the ROW COUNT can.
        An arc chunk is M waypoints plus one velocity token, so M+1 rows; a
        time-indexed chunk is ``rollout_horizon`` rows. At D40/M100 that is 101
        vs 100. A baseline whose horizon happened to equal M+1 would be
        misread as arc, so keep them distinct when adding configs.
        """
        return t is not None and t.ndim == 3 and t.shape[1] == self._M + 1

    def _detokenize_batch(self, tensor):
        """Detokenize arc tokens; pass a time-indexed chunk through untouched.

        Overridden rather than guarded at each call site because the parent
        detokenizes in three places -- the metric path, the ``detok_*`` metrics,
        and ``_visualize_preds`` -- and the viz path is not otherwise reachable
        to override. Without this a baseline run dies in validation with
        "configured for M=100 (M+1=101 tokens), got 100 tokens".
        """
        t = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
        return super()._detokenize_batch(t) if self._is_arc(t) else t

    def _resample_pair(self, pred, gt):
        """Both chunks onto the shared arc grid -> (B, N, 14) each, or (None, None)."""
        if pred is None or gt is None:
            return None, None
        p = pred.detach().cpu().numpy() if torch.is_tensor(pred) else np.asarray(pred)
        g = gt.detach().cpu().numpy() if torch.is_tensor(gt) else np.asarray(gt)
        if p.ndim != 3 or g.ndim != 3:
            return None, None
        ps, gs = [], []
        for b in range(min(p.shape[0], g.shape[0])):
            rp = arc_matched_resample(
                p[b],
                self.arc_match_distance,
                self.arc_match_points,
                include_rotation=True,
            )
            rg = arc_matched_resample(
                g[b],
                self.arc_match_distance,
                self.arc_match_points,
                include_rotation=True,
            )
            if rp is None or rg is None:
                continue
            ps.append(rp)
            gs.append(rg)
        if not ps:
            return None, None
        return (
            torch.from_numpy(np.stack(ps)).float(),
            torch.from_numpy(np.stack(gs)).float(),
        )

    def compute_metrics_and_viz(self, batch):
        metrics, images_dict = super().compute_metrics_and_viz(batch)

        algo = self.model
        preds = algo.forward_eval(batch)
        # The parent emits `_detok_*` metrics. On a time-indexed run nothing was
        # detokenized, and its 100 time steps cover a different amount of travel
        # than the arc run's reconstruction, so letting the two share a name
        # would overlay non-comparable numbers on one chart -- the exact thing
        # the arc-matched metrics exist to avoid. Rename them for non-arc runs.
        if not any(self._is_arc(v) for v in preds.values() if torch.is_tensor(v)):
            metrics = {
                k.replace("_detok_", "_timeidx_"): v for k, v in metrics.items()
            }
        if not self.arc_match_distance:
            return metrics, images_dict

        F = torch.nn.functional
        for embodiment_id, _batch in batch.items():
            _batch = algo.norm_stats.unnormalize(_batch, embodiment_id)
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = algo.ac_keys[embodiment_id]
            pk = f"{embodiment_name}_{ac_key}"
            if pk not in preds or ac_key not in _batch:
                continue
            # Detokenize when the run is arc; a time-indexed chunk is untouched.
            # From here on both variants are the same shape and the same units.
            P, G = self._resample_pair(
                self._arc_match_source(preds[pk]),
                self._arc_match_source(_batch[ac_key]),
            )
            if P is None:
                continue

            metrics[f"Valid/{pk}_arcmatch_paired_mse_avg"] = F.mse_loss(
                P[..., RESAMPLED_PAIRED], G[..., RESAMPLED_PAIRED]
            )
            metrics[f"Valid/{pk}_arcmatch_xyz_mse_avg"] = F.mse_loss(
                P[..., RESAMPLED_XYZ], G[..., RESAMPLED_XYZ]
            )
            # Endpoint error in metres, averaged over arms: does an open-loop
            # rollout finish where it should, after D metres of travel.
            per_arm = [
                torch.linalg.norm(P[:, -1, list(s)] - G[:, -1, list(s)], dim=-1)
                for s in RESAMPLED_ARM_XYZ
            ]
            metrics[f"Valid/{pk}_arcmatch_final_xyz_l2_m_avg"] = torch.stack(
                per_arm
            ).mean()
            # Orientation error on the same arc grid, in degrees. Geodesic so
            # that arc and non-arc runs -- and eef vs camera frame -- compare.
            metrics[f"Valid/{pk}_arcmatch_rot_geodesic_deg_avg"] = torch.tensor(
                _geodesic_deg(P, G)
            )

        return metrics, images_dict
