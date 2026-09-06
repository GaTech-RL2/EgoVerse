#!/usr/bin/env python3
"""Data-pipeline smoke for the E1 fold rows: build the dataset from the hydra
config, pull samples, print shapes, and round-trip the arc tokens through the
detokenizer against the carried ``actions_time`` ground truth."""
import sys
import time

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

WT, exp, root = sys.argv[1], sys.argv[2], sys.argv[3]  # exp = experiment suffix (arcvel, arclogdur_ps, ...)
with initialize_config_dir(config_dir=f"{WT}/egomimic/hydra_configs", version_base=None):
    cfg = compose(
        config_name="train_zarr_cartesian",
        overrides=[f"+experiment=e1/fold_{exp}", "e1.spread=smoke", f"e1.train_root={root}",
                   f"e1.valid_root={root}", "e1.h_match_frames=40", "seed=0"],
    )
variant = str(cfg.e1.variant)
t0 = time.time()
ds = instantiate(cfg.data.train_datasets.human_bimanual)
print(f"[{variant}] dataset built in {time.time()-t0:.1f}s: {len(ds)} samples, leaves={len(ds.datasets)}")
from egomimic.rldb.zarr.e1_arc_tokenizer import TokenizeBimanualArcLengthE1, ARM_LAYOUT
from egomimic.rldb.zarr.arc_length_tokenizer import cumulative_arc_length

detok = None
if variant != "time":
    from egomimic.rldb.embodiment.e1_fold import VELOCITY_MODES
    detok = TokenizeBimanualArcLengthE1(min_distance_unit=0.40, resampled_vector_length=100, dt=1/30,
                                        velocity_norm="path", velocity_mode=VELOCITY_MODES[variant],
                                        progress_smooth_hz=cfg.e1.progress_smooth_hz)
    # #3: exercise the anchor sampler the way the datamodule will
    if cfg.e1.get("anchor_sampler"):
        from omegaconf import OmegaConf
        from egomimic.rldb.zarr.e1_anchor_sampler import build_anchor_sampler
        smp = build_anchor_sampler(ds, **OmegaConf.to_container(cfg.e1.anchor_sampler, resolve=True))
        draws = np.array(list(iter(smp))[:20000])
        print(f"  anchor sampler: {len(smp)} draws/epoch, first 20k draws hit {len(np.unique(draws))} distinct anchors")
errs, clocks = [], []
for idx in np.linspace(0, len(ds) - 1, 12).astype(int):
    t0 = time.time()
    s = ds[int(idx)]
    if idx == 0:
        for k, v in s.items():
            shape = getattr(v, "shape", None)
            print(f"  {k}: {shape if shape is not None else type(v).__name__}")
        print(f"  one sample in {time.time()-t0:.2f}s")
    a = np.asarray(s["actions_cartesian"], dtype=np.float64)
    gt = np.asarray(s["actions_time"], dtype=np.float64)
    assert gt.shape == (100, 14), gt.shape
    if variant == "time":
        assert a.shape == (100, 14), a.shape
        assert np.allclose(a, gt)
        continue
    dec = detok.detokenize(a, action_horizon=40)
    pred_clocks = detok.clock_at_waypoints(a)
    for k, (xo, _, _, vsl) in enumerate(ARM_LAYOUT):
        e = np.sqrt(np.mean(np.sum((dec[:, xo:xo+3] - gt[:40, xo:xo+3]) ** 2, axis=1)))
        errs.append(e)
        wp = a[:100, xo:xo+3]
        cum = cumulative_arc_length(wp)
        gt_cum = cumulative_arc_length(gt[:, xo:xo+3])
        # clock check: predicted vs true time to reach min(span, what GT covers in 100 frames)
        s_chk = min(float(cum[-1]), float(gt_cum[-1])) * 0.999
        if s_chk > 1e-3:
            t_true = float(np.interp(s_chk, gt_cum, np.arange(len(gt_cum)))) / 30.0
            t_pred = float(np.interp(s_chk, cum, pred_clocks[k]))
            clocks.append((round(float(cum[-1]), 3), round(t_true, 2), round(t_pred, 2)))
if variant != "time":
    ratios = [p / t for _, t, p in clocks if t > 0]
    print(f"  token shape {a.shape}; codec E_time(40f) RMS over arms/samples = {np.sqrt(np.mean(np.square(errs))):.4f} m "
          f"(median {np.median(errs):.4f}); clock ratio pred/true median {np.median(ratios):.2f} "
          f"(q10 {np.quantile(ratios, .1):.2f}, q90 {np.quantile(ratios, .9):.2f}); (span, t_true, t_pred) = {clocks[:6]}")
print(f"[{variant}] SMOKE_OK")
