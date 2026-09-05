"""EEF-frame validation for arc-tokenized and time-indexed runs on one axis.

Four metric families, every one of them anchored on the SAME object: the
ground-truth action chunk under normal (time-indexed) sampling, in metres.

  arcmatch  Re-tokenize prediction and ground truth onto a MATCHED arc span --
            per arm, the shorter of the two travelled distances -- then score
            the waypoints. Travel-normalized: it asks "is the path shape right",
            with the amount of travel divided out. Reported with and without the
            velocity row.
  dtw       Dynamic time warping of the prediction's time-indexed trajectory
            against the ground truth clipped to the first D metres. Elastic in
            time, so it scores path shape independent of speed, and it is the
            one family that DOES see a travel mismatch.
  detok     Arc runs only. The reconstruction the controller would actually
            receive, scored against the raw ground-truth chunk, step for step.
  baseline  Time-indexed runs only. The same MSE against the same raw chunk.

``detok`` and ``baseline`` are the same computation under two names; the name
records which action space the model natively predicts in. With ``chunk_length``
set so both runs cover the same wall-clock span they are numerically comparable:
same formula, same reference, same number of steps.

WHY THE RAW CHUNK HAS TO BE PRESERVED
The arc tokenizer overwrites ``actions_cartesian`` in place, so an arc run's
batch carries only the (M+1, 14) token. Recovering the time-indexed ground truth
by detokenizing it would be circular: the reconstruction spans exactly D metres
by construction, so its travelled distance -- the quantity the matched span is
built from -- would carry no information and every span would collapse to D.
``TokenizeBimanualArcLengthCartesian(preserve_action_key=...)`` therefore stashes
an untouched copy under ``actions_cartesian_untokenized``, which arrives here in
metres -- it is normalized on the way in (``_infer_key_type`` classifies any
post-transform key starting with "actions" as an action key) and unnormalized
again by the ``norm_stats.unnormalize`` call below, so the round trip cancels.

COMPARABILITY -- READ THIS BEFORE RANKING AN ARC RUN AGAINST A BASELINE
Two runs share a chart only if they share (a) ``arc_match_points`` M,
(b) ``arc_match_distance`` D for the dtw target, and (c) a ``chunk_length``
covering the same wall-clock span. (c) cannot be checked from in here -- an arc
keymap may subsample at a different stride than its baseline twin, so equal
sample counts need not mean equal time. ``_gt_chunk_travel_m_avg`` is logged for
exactly that: when it agrees across two runs they are looking at the same
stretch of ground truth.

Even then, no single family ranks the two hypothesis classes fairly, because
each removes a different invariance and the two classes put their error in
different channels. Measured on synthetic trajectories (D=0.40, M=100, T=45):

  * arcmatch is exactly travel-invariant -- 1e-9 to 3e-8 for predictions
    covering 0.5x to 2x the ground truth's distance. It scores SHAPE only.
  * The tokenize/detokenize round trip costs ~7e-9 in arcmatch space, so an arc
    run carries no handicap there. In the TIME domain the same round trip costs
    ~7e-4, which is a floor an arc run can never beat and a baseline never pays.
    A baseline at 1e-4 therefore outscores a PERFECT arc model on detok /
    baseline. Never rank the two classes on those two families alone.
  * Conversely, at matched time-domain error arcmatch flattered the arc model by
    ~1000x, because an arc head's error lands mostly in the velocity token --
    timing -- which arcmatch divides out. That is the metric behaving correctly,
    not a bug, but it is not a proxy for overall quality.
  * Arc length is very sensitive to high-frequency jitter, which a time-indexed
    prediction has and an arc reconstruction structurally cannot: 10 mm of iid
    noise inflated measured travel 3x (0.362 -> 1.087 m). Low-pass filtering
    before measuring shrinks the inflation but does not remove the bias above.
    ``_arcmatch_travel_ratio_avg`` is the tell -- it reads ~1.0 when a
    prediction's extent is honest.

So read them together, not one at a time: arcmatch for shape, the with-vel
variant minus the without-vel one for how much of the error is timing, dtw for
whether the extent is right, detok/baseline within a single action space, and
travel_ratio to know whether the prediction moved at all.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from egomimic.eval.eval_arctok import ArcTokEvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment
from egomimic.rldb.embodiment.eva import UNTOKENIZED_ACTION_KEY
from egomimic.rldb.zarr.arc_length_tokenizer import (
    _dist_interval_indices,
    cumulative_arc_length,
    resample_by_distance,
)

# Canonical bimanual cartesian layout, per arm: [xyz(3), ypr(3), grip(1)].
ARM_BLOCKS = ((0, 3, 6), (7, 10, 13))  # (xyz offset, ypr offset, gripper index)
ARM_XYZ = ((0, 1, 2), (7, 8, 9))
ARM_YPR = ((3, 4, 5), (10, 11, 12))
XYZ_COLS = [c for arm in ARM_XYZ for c in arm]
# Position + gripper. Rotation is excluded from every `paired` MSE so radians
# are never summed with metres; it gets its own geodesic chart instead.
PAIRED_COLS = [0, 1, 2, 6, 7, 8, 9, 13]
YPR_COLS = [c for arm in ARM_YPR for c in arm]
GRIP_COLS = [6, 13]


# ---------------------------------------------------------------------------
# arc-span tokenization
# ---------------------------------------------------------------------------


def _arm_views(traj: np.ndarray):
    """(T, 14) -> per-arm (pos (T,3), ypr (T,3), grip (T,1))."""
    for xyz_off, ypr_off, grip_i in ARM_BLOCKS:
        yield (
            traj[:, xyz_off : xyz_off + 3],
            traj[:, ypr_off : ypr_off + 3],
            traj[:, grip_i : grip_i + 1],
        )


def arm_travel(traj: np.ndarray) -> np.ndarray:
    """(T, 14) -> (2,) total translational arc length per arm, in metres."""
    return np.array(
        [float(cumulative_arc_length(pos)[-1]) for pos, _, _ in _arm_views(traj)]
    )


def tokenize_span(
    traj: np.ndarray, spans: np.ndarray, num_points: int, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize a time-indexed chunk over a caller-supplied PER-ARM span.

    This is ``arctokenize(traj, distance)`` from the matched-span rule, with the
    distance free rather than pinned to the tokenizer's D. Each arm is resampled
    to ``num_points`` samples spaced uniformly in ITS OWN arc length over
    [0, span_arm], using the same interpolators the real tokenizer uses (linear
    on xyz and gripper, SLERP on rotation), so the waypoints are the ones the
    data pipeline would have produced had D been ``span_arm``.

    The velocity row is recomputed here rather than read off a model output, so
    that a time-indexed run -- which predicts no velocity token at all -- still
    has one, and both run types get it by the same formula. It is the mean
    per-dim rate over the covered index range, matching ``MEAN_PER_DIM``.

    returns: (waypoints (num_points, 14), velocity (14,)).
    """
    waypoints = np.zeros((num_points, 14), dtype=np.float64)
    velocity = np.zeros(14, dtype=np.float64)
    for arm, (pos, ypr, grip) in enumerate(_arm_views(traj)):
        cum = cumulative_arc_length(pos)
        end_s = float(min(spans[arm], cum[-1]))
        p, y, g = resample_by_distance(
            pos, ypr, grip, cum, 0.0, end_s, num_points, start_idx=0
        )
        o = arm * 7
        waypoints[:, o : o + 3] = p
        waypoints[:, o + 3 : o + 6] = y
        waypoints[:, o + 6] = g[:, 0]

        # duration = (number of source steps the span covers) * dt. Falling back
        # to one step keeps a stationary arm's velocity finite and zero rather
        # than NaN, which is how the tokenizer handles the same case.
        start_i, end_i = _dist_interval_indices(cum, 0.0, end_s)
        dur = max(end_i - start_i, 1) * dt
        velocity[o : o + 3] = (p[-1] - p[0]) / dur
        velocity[o + 3 : o + 6] = (y[-1] - y[0]) / dur
        velocity[o + 6] = (g[-1, 0] - g[0, 0]) / dur
    return waypoints, velocity


def match_spans(pred_ti: np.ndarray, gt_ti: np.ndarray) -> np.ndarray:
    """Per-arm matched span: the shorter of the two travelled distances.

    Whichever side travels less sets the window, so both are re-tokenized over a
    stretch of motion they both actually cover. Cutting the longer one is what
    makes the comparison fair; without it a prediction that runs further is
    scored against ground truth it never reached.
    """
    return np.minimum(arm_travel(pred_ti), arm_travel(gt_ti))


def clip_to_distance(traj: np.ndarray, distance: float) -> np.ndarray:
    """Truncate a chunk at the first row where EITHER arm passes ``distance``.

    Used to build the dtw target. Both arms keep the same row count because the
    warping runs on whole rows.
    """
    ends = []
    for pos, _, _ in _arm_views(traj):
        cum = cumulative_arc_length(pos)
        over = np.nonzero(cum > distance)[0]
        ends.append(int(over[0]) + 1 if len(over) else len(traj))
    return traj[: max(2, min(ends))]


# ---------------------------------------------------------------------------
# dynamic time warping
# ---------------------------------------------------------------------------


def dtw_path(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Warping path between two (N, 3) position sequences.

    Standard DTW recurrence, but the forward pass sweeps ANTI-DIAGONALS instead
    of rows: every cell on i+j=k depends only on diagonals k-1 and k-2, so each
    diagonal is one vectorized numpy step. That turns an O(N*M) python loop into
    O(N+M) numpy calls, which is the difference between DTW being affordable in
    a per-epoch val loop and not.

    returns: (ia, ib) index arrays of equal length, the aligned pairs.
    """
    na, nb = len(A), len(B)
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    D = np.full((na + 1, nb + 1), np.inf)
    D[0, 0] = 0.0
    for k in range(2, na + nb + 1):
        i = np.arange(max(1, k - nb), min(na, k - 1) + 1)
        if not len(i):
            continue
        j = k - i
        D[i, j] = d[i - 1, j - 1] + np.minimum(
            np.minimum(D[i - 1, j], D[i, j - 1]), D[i - 1, j - 1]
        )
    ia, ib = [], []
    i, j = na, nb
    while i > 0 and j > 0:
        ia.append(i - 1)
        ib.append(j - 1)
        step = int(np.argmin((D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])))
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    return np.array(ia[::-1]), np.array(ib[::-1])


# ---------------------------------------------------------------------------
# rotation
# ---------------------------------------------------------------------------


def geodesic_deg(P: np.ndarray, G: np.ndarray) -> float:
    """Mean geodesic angle in degrees between two (..., 14) chunks.

    The angle of the single rotation carrying one orientation to the other --
    the shortest path on SO(3). Bounded in [0, 180], never wraps, unaffected by
    gimbal lock, and unchanged by expressing both rotations in another frame,
    none of which is true of a per-axis ypr difference.
    """
    per_arm = []
    for cols in ARM_YPR:
        rp = R.from_euler("ZYX", P[..., list(cols)].reshape(-1, 3))
        rg = R.from_euler("ZYX", G[..., list(cols)].reshape(-1, 3))
        per_arm.append(float(np.degrees((rp * rg.inv()).magnitude()).mean()))
    return float(np.mean(per_arm))


def pose_err_m(P: np.ndarray, G: np.ndarray, lever_m: float) -> float:
    """Position AND rotation error as one number, in metres.

    Summing squared metres and squared radians directly would make the score
    depend on an arbitrary unit choice -- rescaling the rotation representation
    would "improve" it. So rotation is converted into the length it actually
    costs: a geodesic error of theta radians displaces a point rigidly attached
    ``lever_m`` from the EEF origin by lever_m * theta. The result is the RMS
    displacement of that point,

        sqrt( mean( |dp|^2 + (lever_m * theta)^2 ) )

    which reads directly as "a point this far out on the gripper is off by X
    metres". Both terms are frame-invariant -- translation because the camera
    revert is rigid, rotation because conjugation preserves the geodesic angle
    -- so the combined number is too.

    ``lever_m`` is the one judgement call, and it is explicit rather than
    smuggled in through units: it says how much a radian is worth in metres.
    """
    per_arm = []
    for xyz_cols, ypr_cols in zip(ARM_XYZ, ARM_YPR):
        dp2 = ((P[..., list(xyz_cols)] - G[..., list(xyz_cols)]) ** 2).sum(-1)
        rp = R.from_euler("ZYX", P[..., list(ypr_cols)].reshape(-1, 3))
        rg = R.from_euler("ZYX", G[..., list(ypr_cols)].reshape(-1, 3))
        theta = (rp * rg.inv()).magnitude().reshape(dp2.shape)
        per_arm.append(dp2 + (float(lever_m) * theta) ** 2)
    return float(np.sqrt(np.mean(per_arm)))


def _mse(a: np.ndarray, b: np.ndarray, cols=None) -> float:
    if cols is not None:
        a, b = a[..., cols], b[..., cols]
    return float(np.mean((a - b) ** 2))


class EefArcMatchEval(ArcTokEvalVideo):
    """Arc-matched, DTW and time-domain EEF metrics for both run types."""

    def __init__(
        self,
        *args,
        chunk_length: int | dict | None = None,
        dtw_max_samples: int = 8,
        dtw_clip_gt_to_distance: bool = False,
        rot_lever_m: float = 0.10,
        untokenized_action_key: str = UNTOKENIZED_ACTION_KEY,
        **kwargs,
    ):
        """
        Args:
            chunk_length: how many steps of the time-indexed ground truth to
                score, as an int or a per-embodiment dict. Set it to the value
                that makes an arc run cover the same WALL-CLOCK span as its
                baseline twin -- equal sample counts are not enough when the two
                keymaps subsample at different strides. Left unset, the full
                available chunk is used and the time-domain families stop being
                cross-comparable (``_gt_chunk_travel_m_avg`` will show it).
            dtw_max_samples: DTW is O(N*M) per arm per sample, so only this many
                samples per batch are warped. The metric is a batch mean either
                way; this bounds the cost.
            dtw_clip_gt_to_distance: clip the dtw target to the first
                ``arc_match_distance`` metres. Off by default because it breaks
                the property that matters most in a ranking signal: with the
                target clipped and the prediction not, a PERFECT prediction
                scores 0.02 m rather than 0, purely because it kept going past
                where the target was cut. Against the full scored chunk a
                perfect prediction scores exactly 0 and a travel mismatch still
                costs, which is the behaviour this family exists for. Turn it on
                only when the two runs' ground-truth chunks cannot be equalized
                through ``chunk_length``.
            rot_lever_m: lever arm in metres for the combined ``pose_err_m``
                metrics -- how far out on the gripper the scored point sits, and
                so how many metres one radian of orientation error is worth.
                0.10 m is roughly a gripper's length. Two runs must share it to
                share those charts.
            untokenized_action_key: where the tokenizer stashed the untouched
                time-indexed chunk. Arc runs need it; time-indexed runs already
                have the real thing under ``ac_key`` and ignore this.
        """
        super().__init__(*args, **kwargs)
        # Normalize to a plain dict of ints up front. Hydra hands this over as an
        # omegaconf DictConfig, which is NOT a `dict` subclass, so an
        # `isinstance(cl, dict)` test downstream silently misses and `int()` gets
        # handed the container itself. Duck-typing on `.keys()` covers both, and
        # coercing here means the per-call path only ever sees int or None.
        if chunk_length is not None and hasattr(chunk_length, "keys"):
            chunk_length = {str(k): int(v) for k, v in chunk_length.items()}
        elif chunk_length is not None:
            chunk_length = int(chunk_length)
        self.chunk_length = chunk_length
        self.dtw_max_samples = int(dtw_max_samples)
        self.dtw_clip_gt_to_distance = bool(dtw_clip_gt_to_distance)
        self.rot_lever_m = float(rot_lever_m)
        self.untokenized_action_key = untokenized_action_key
        self._warned_missing_untokenized = False

    # -- plumbing -----------------------------------------------------------

    def _is_arc(self, t) -> bool:
        """True when a chunk is arc tokens rather than time steps.

        Both are 14 wide, so width cannot tell them apart -- the ROW COUNT can.
        An arc chunk is M waypoints plus one velocity token; a time-indexed one
        is ``rollout_horizon`` rows. A baseline whose horizon happened to equal
        M+1 would be misread as arc, so keep them distinct when adding configs.
        """
        return t is not None and t.ndim == 3 and t.shape[1] == self._M + 1

    def _detokenize_batch(self, tensor):
        """Detokenize arc tokens; pass a time-indexed chunk through untouched.

        Overridden rather than guarded at each call site because the parent
        detokenizes in three places -- the metric path, its own metrics, and
        ``_visualize_preds`` -- and the viz path is not otherwise reachable.
        Without this a baseline run dies in validation with "configured for
        M=100 (M+1=101 tokens), got 100 tokens".
        """
        t = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
        return super()._detokenize_batch(t) if self._is_arc(t) else t

    def _chunk_len(self, embodiment_name: str, available: int) -> int:
        cl = self.chunk_length
        if isinstance(cl, dict):
            cl = cl.get(embodiment_name)
        return min(cl, available) if cl else available

    def _time_indexed_gt(self, batch_unnorm: dict, ac_key: str, is_arc: bool):
        """The ground truth under normal action-chunk sampling, in metres."""
        if not is_arc:
            return batch_unnorm.get(ac_key)
        gt = batch_unnorm.get(self.untokenized_action_key)
        if gt is None and not self._warned_missing_untokenized:
            self._warned_missing_untokenized = True
            print(
                f"[EefArcMatchEval] '{self.untokenized_action_key}' is not in the "
                "batch, so the arc-matched, dtw and detok metrics are skipped. "
                "Rebuild the data config with an arc tokenizer that sets "
                "preserve_action_key (egomimic.rldb.embodiment.eva."
                "_append_arc_tokenizer does this by default)."
            )
        return gt

    @staticmethod
    def _np(t):
        return (
            t.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(t) else None
        )

    # -- metrics ------------------------------------------------------------

    def compute_metrics_and_viz(self, batch):
        metrics, images_dict = super().compute_metrics_and_viz(batch)
        # Both ancestors emit `_arcmatch_*` and `_detok_*` under these exact
        # names against different references (the base resamples to a fixed D;
        # the arc-tok parent compares two reconstructions). Ours replace them,
        # so drop theirs rather than let one name carry two definitions.
        metrics = {
            k: v
            for k, v in metrics.items()
            if "_arcmatch_" not in k and "_detok_" not in k
        }

        algo = self.model
        preds = algo.forward_eval(batch)
        M = self.arc_match_points
        D = self.arc_match_distance
        dt = self._detokenizer.tokenizer.config.dt

        for embodiment_id, _batch in batch.items():
            _batch = algo.norm_stats.unnormalize(_batch, embodiment_id)
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = algo.ac_keys[embodiment_id]
            pk = f"{embodiment_name}_{ac_key}"
            if pk not in preds or preds[pk] is None:
                continue

            is_arc = self._is_arc(preds[pk])
            gt_t = self._time_indexed_gt(_batch, ac_key, is_arc)
            if gt_t is None:
                continue
            T = self._chunk_len(embodiment_name, int(gt_t.shape[1]))
            gt = self._np(gt_t)[:, :T]
            # An arc prediction becomes time-indexed by the same reconstruction
            # a controller would run, so from here the two run types are one
            # code path over identical objects.
            pred_t = (
                self._detokenize_batch_to(preds[pk], T) if is_arc else preds[pk][:, :T]
            )
            pred = self._np(pred_t)
            if pred is None or pred.shape[1] < 2 or gt.shape[1] < 2:
                continue

            B = min(len(pred), len(gt))
            self._add_arcmatch(metrics, pk, pred[:B], gt[:B], M, dt)
            self._add_dtw(metrics, pk, pred[:B], gt[:B], D)
            # One name per action space, same computation: "detok" is the
            # reconstruction an arc run's controller would receive, "baseline"
            # is a time-indexed run's raw chunk. Both are scored against the
            # same raw ground truth, split the same way, so position, rotation
            # and gripper error can be told apart within a run.
            family = "detok" if is_arc else "baseline"
            p_, g_ = pred[:B], gt[:B]
            for suffix, value in (
                ("paired_mse_avg", _mse(p_, g_, PAIRED_COLS)),
                ("xyz_mse_avg", _mse(p_, g_, XYZ_COLS)),
                ("ypr_mse_avg", _mse(p_, g_, YPR_COLS)),
                ("gripper_mse_avg", _mse(p_, g_, GRIP_COLS)),
                ("final_mse_avg", _mse(p_[:, -1], g_[:, -1], PAIRED_COLS)),
                ("rot_geodesic_deg_avg", geodesic_deg(p_, g_)),
                ("pose_err_m_avg", pose_err_m(p_, g_, self.rot_lever_m)),
            ):
                metrics[f"Valid/{pk}_{family}_{suffix}"] = torch.tensor(value)
            # The comparability receipt: two runs whose GT travels the same
            # distance over the scored chunk are looking at the same motion.
            metrics[f"Valid/{pk}_gt_chunk_travel_m_avg"] = torch.tensor(
                float(np.mean([arm_travel(g).mean() for g in gt[:B]]))
            )

        return metrics, images_dict

    def _detokenize_batch_to(self, arc_tensor, horizon: int):
        """Detokenize to exactly ``horizon`` control steps.

        ``rollout_horizon`` is a video-length knob; the metrics need the
        reconstruction to line up row-for-row with the ground-truth chunk, so
        the horizon is taken from the data here, not from the config.
        """
        saved, self.rollout_horizon = self.rollout_horizon, int(horizon)
        try:
            return super()._detokenize_batch(arc_tensor)
        finally:
            self.rollout_horizon = saved

    def _add_arcmatch(self, metrics, pk, pred, gt, M, dt):
        """Re-tokenize both sides onto the matched per-arm span, then score."""
        Pw, Gw, Pv, Gv, spans, ratios = [], [], [], [], [], []
        for p, g in zip(pred, gt):
            s = match_spans(p, g)
            if not np.all(np.isfinite(s)):
                continue
            pw, pv = tokenize_span(p, s, M, dt)
            gw, gv = tokenize_span(g, s, M, dt)
            Pw.append(pw)
            Gw.append(gw)
            Pv.append(pv)
            Gv.append(gv)
            spans.append(s)
            gt_travel = arm_travel(g)
            ratios.append(arm_travel(p) / np.maximum(gt_travel, 1e-6))
        if not Pw:
            return
        Pw, Gw = np.stack(Pw), np.stack(Gw)
        # With-velocity variant: the velocity row appended as one more sample,
        # which is what the arc head actually predicts alongside the waypoints.
        Pf = np.concatenate([Pw, np.stack(Pv)[:, None, :]], axis=1)
        Gf = np.concatenate([Gw, np.stack(Gv)[:, None, :]], axis=1)

        m = {
            "arcmatch_paired_mse_avg": _mse(Pw, Gw, PAIRED_COLS),
            "arcmatch_withvel_paired_mse_avg": _mse(Pf, Gf, PAIRED_COLS),
            "arcmatch_xyz_mse_avg": _mse(Pw, Gw, XYZ_COLS),
            "arcmatch_rot_geodesic_deg_avg": geodesic_deg(Pw, Gw),
            # Position and rotation folded into one number, in metres.
            "arcmatch_pose_err_m_avg": pose_err_m(Pw, Gw, self.rot_lever_m),
            # Endpoint error in metres at the end of the matched span.
            "arcmatch_final_xyz_l2_m_avg": float(
                np.mean(
                    [
                        np.linalg.norm(Pw[:, -1, list(s)] - Gw[:, -1, list(s)], axis=-1)
                        for s in ARM_XYZ
                    ]
                )
            ),
            # How much motion the matched span covers. Arc-matching divides
            # travel out of the score, so read these two next to it: a policy
            # that stalls scores well on shape while span and ratio collapse.
            "arcmatch_span_m_avg": float(np.mean(spans)),
            "arcmatch_travel_ratio_avg": float(np.mean(ratios)),
        }
        for k, v in m.items():
            metrics[f"Valid/{pk}_{k}"] = torch.tensor(v)

    def _add_dtw(self, metrics, pk, pred, gt, distance):
        """Warp the prediction against the ground-truth chunk.

        The target is the same curve for both run types -- that is what makes
        the chart shared -- and ``chunk_length`` is what equalizes it, not a
        distance clip. Warping absorbs a speed difference but not a length one,
        so a prediction that stops short or runs long pays for it here. This is
        the family that sees the extent the arc-matched score divides out.
        """
        paired, l2 = [], []
        for p, g in zip(pred[: self.dtw_max_samples], gt[: self.dtw_max_samples]):
            gc = clip_to_distance(g, distance) if self.dtw_clip_gt_to_distance else g
            for arm, cols in enumerate(ARM_XYZ):
                cols = list(cols)
                ia, ib = dtw_path(p[:, cols], gc[:, cols])
                l2.append(
                    float(
                        np.mean(
                            np.linalg.norm(p[ia][:, cols] - gc[ib][:, cols], axis=-1)
                        )
                    )
                )
                block = list(range(arm * 7, arm * 7 + 7))
                keep = [c for c in block if c in PAIRED_COLS]
                paired.append(_mse(p[ia], gc[ib], keep))
        if not paired:
            return
        metrics[f"Valid/{pk}_dtw_paired_mse_avg"] = torch.tensor(float(np.mean(paired)))
        metrics[f"Valid/{pk}_dtw_xyz_l2_m_avg"] = torch.tensor(float(np.mean(l2)))
