"""Show the span confound and that the gt-span diagnostic removes it.

The cleanest case: a prediction that is the ground truth's OWN first 0.29 m of
path, perfectly. Its shape is exactly right over that span, but it covers only
43 % of the horizon's travel.

  * lab metric  span = min(travel(pred), travel(gt)) = 0.29 -> scores ~0.
    A row is rewarded for stopping short.
  * gt-span     span = min(travel(gt), D) = 0.40 for every row -> the short
    prediction is stretched over the M waypoints and pays for the travel it
    never made.
"""
import numpy as np
from egomimic.eval.e1_fold_tempo_eval import (
    PAIRED_COLS, arm_travel, match_spans, gt_spans, tokenize_span, _mse_cols,
    cumulative_arc_length)

T, M, dt, D = 100, 100, 1 / 30, 0.40


def curve(total_m, n=T):
    """A curved bimanual path of `total_m` arc length, sampled at n frames."""
    u = np.linspace(0, 1, 2000)
    p = np.stack([u, 0.15 * np.sin(np.pi * u), 0.05 * u ** 2], axis=1)
    c = cumulative_arc_length(p)
    p = p * (total_m / c[-1])
    c = cumulative_arc_length(p)
    s = np.linspace(0, c[-1], n)
    a = np.zeros((n, 14))
    for off in (0, 7):
        for j in range(3):
            a[:, off + j] = np.interp(s, c, p[:, j])
        a[:, off + 6] = np.linspace(0, 1, n)
    return a


gt = curve(0.68)
# the gt's own first 0.29 m, resampled to the full 100 frames: perfect shape, short travel
c_gt = cumulative_arc_length(gt[:, 0:3])
s_short = np.linspace(0, 0.29, T)
short = np.zeros((T, 14))
for off in (0, 7):
    for j in range(3):
        short[:, off + j] = np.interp(s_short, c_gt, gt[:, off + j])
    short[:, off + 6] = np.interp(s_short, c_gt, gt[:, off + 6])
full = gt.copy()

print(f"travel  gt={arm_travel(gt)[0]:.3f}  full-travel pred={arm_travel(full)[0]:.3f}  "
      f"short pred={arm_travel(short)[0]:.3f} m\n")
print("{:18} {:>9} {:>11} {:>9} {:>12}".format("row", "lab span", "lab MSE", "gt span", "gtspan MSE"))
for name, pr in (("full-travel pred", full), ("short pred", short)):
    sl = match_spans(pr, gt)
    pw, _ = tokenize_span(pr, sl, M, dt); gw, _ = tokenize_span(gt, sl, M, dt)
    sg = gt_spans(gt, D)
    pg, _ = tokenize_span(pr, sg, M, dt); gg, _ = tokenize_span(gt, sg, M, dt)
    print(f"{name:18} {sl[0]:>9.3f} {_mse_cols(pw, gw, PAIRED_COLS):>11.6f} "
          f"{sg[0]:>9.3f} {_mse_cols(pg, gg, PAIRED_COLS):>12.6f}")
