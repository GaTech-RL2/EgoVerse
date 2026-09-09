#!/usr/bin/env python3
"""Quantify the logdur clock bias on the EEF-FRAME chunks the model trains on.

For each chunk the token's span is traversed in a definite number of seconds.
`arcdur` stores those seconds and sums them, so its total is that number by
construction. `logdur` rebuilds the total as (waypoint-polyline length) x
exp(row 0), and row 0 was measured against the resampled target span -- the
polyline cuts corners, so the total comes out short by exactly that ratio.
This prints both, per arm, over real chunks.
"""
import os, sys
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

WT, root, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
EXP = os.environ.get("E1_EXP", "abc")
with initialize_config_dir(config_dir=f"{WT}/egomimic/hydra_configs", version_base=None):
    cfg = compose(config_name="train_zarr_cartesian",
                  overrides=[f"+experiment=e1/{EXP}_time", "e1.spread=smoke",
                             f"e1.train_root={root}", f"e1.valid_root={root}",
                             "e1.chunk_length=200", "e1.time_rows=200",
                             "e1.h_match_frames=40", "seed=0"])
ds_name = next(iter(cfg.data.train_datasets.keys()))
ds = instantiate(cfg.data.train_datasets[ds_name])
print(f"dataset: {len(ds)} samples (eef-frame, 200-frame chunks)")

from egomimic.rldb.zarr.e1_arc_tokenizer import (
    ARM_LAYOUT, TokenizeBimanualArcLengthE1, cumulative_arc_length)
D, M, dt = float(cfg.e1.D), int(cfg.e1.M), 1/30
mk = lambda m: TokenizeBimanualArcLengthE1(min_distance_unit=D, resampled_vector_length=M,
                                           dt=dt, velocity_norm="path", velocity_mode=m)
tk = {m: mk(m) for m in ("logdur", "dur")}

t_true, t_ld, t_du, poly_ratio = [], [], [], []
for idx in np.linspace(0, len(ds) - 1, n).astype(int):
    c = np.asarray(ds[int(idx)]["actions_time"], dtype=np.float64)   # (200, 14) eef-frame
    tok = {m: tk[m].transform({"actions_cartesian": c.copy()})["actions_cartesian"] for m in tk}
    clk = {m: tk[m].clock_at_waypoints(tok[m]) for m in tk}
    for k, (xo, _, _, _) in enumerate(ARM_LAYOUT):
        cum_src = cumulative_arc_length(c[:, xo:xo+3])
        span = min(D, float(cum_src[-1]))
        if span < 0.02:
            continue
        # true seconds to cover `span` of the source path
        tt = float(np.interp(span, cum_src, np.arange(len(cum_src)))) * dt
        poly = float(cumulative_arc_length(tok["dur"][:M, xo:xo+3])[-1])
        t_true.append(tt); t_ld.append(clk["logdur"][k][-1]); t_du.append(clk["dur"][k][-1])
        poly_ratio.append(poly / span)

t_true, t_ld, t_du = map(np.array, (t_true, t_ld, t_du))
poly_ratio = np.array(poly_ratio)
ok = t_true > 1e-6
print(f"\n{ok.sum()} arm-chunks with span >= 2 cm\n")
print(f"  waypoint-polyline / span                 median {np.median(poly_ratio):.5f}"
      f"  p05 {np.quantile(poly_ratio,0.05):.5f}  p95 {np.quantile(poly_ratio,0.95):.5f}")
print(f"  arcdur  total / TRUE seconds             median {np.median(t_du[ok]/t_true[ok]):.5f}"
      f"  p05 {np.quantile(t_du[ok]/t_true[ok],0.05):.5f}  p95 {np.quantile(t_du[ok]/t_true[ok],0.95):.5f}")
print(f"  logdur  total / TRUE seconds             median {np.median(t_ld[ok]/t_true[ok]):.5f}"
      f"  p05 {np.quantile(t_ld[ok]/t_true[ok],0.05):.5f}  p95 {np.quantile(t_ld[ok]/t_true[ok],0.95):.5f}")
r = t_ld[ok] / np.maximum(t_du[ok], 1e-12)
print(f"  logdur  total / arcdur total             median {np.median(r):.5f}"
      f"  p05 {np.quantile(r,0.05):.5f}  p95 {np.quantile(r,0.95):.5f}")
print(f"\n  -> logdur's clock is short by median {100*(1-np.median(r)):.2f}% "
      f"(p05 {100*(1-np.quantile(r,0.05)):.2f}%) on the eef-frame representation")
print(f"  -> correlation(logdur/arcdur, polyline/span) = "
      f"{np.corrcoef(r, poly_ratio[ok])[0,1]:.4f}")
