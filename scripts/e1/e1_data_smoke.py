#!/usr/bin/env python3
"""Data-pipeline smoke for the E1 fold rows: build the dataset from the hydra
config, pull samples, print shapes, and round-trip the arc tokens through the
detokenizer against the carried ``actions_time`` ground truth."""
import sys
import time

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

WT, variant, root = sys.argv[1], sys.argv[2], sys.argv[3]
with initialize_config_dir(config_dir=f"{WT}/egomimic/hydra_configs", version_base=None):
    cfg = compose(
        config_name="train_zarr_cartesian",
        overrides=[f"+experiment=e1/fold_{variant}", "e1.spread=smoke", f"e1.train_root={root}",
                   f"e1.valid_root={root}", "e1.h_match_frames=40", "seed=0"],
    )
t0 = time.time()
ds = instantiate(cfg.data.train_datasets.human_bimanual)
print(f"[{variant}] dataset built in {time.time()-t0:.1f}s: {len(ds)} samples, leaves={len(ds.datasets)}")
from egomimic.rldb.zarr.e1_arc_tokenizer import TokenizeBimanualArcLengthE1, ARM_LAYOUT
from egomimic.rldb.zarr.arc_length_tokenizer import cumulative_arc_length

detok = None
if variant != "time":
    detok = TokenizeBimanualArcLengthE1(min_distance_unit=0.40, resampled_vector_length=100, dt=1/30,
                                        velocity_norm="path", velocity_mode="mean" if variant == "arcmean" else "profile")
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
    for k, (xo, _, _, vsl) in enumerate(ARM_LAYOUT):
        e = np.sqrt(np.mean(np.sum((dec[:, xo:xo+3] - gt[:40, xo:xo+3]) ** 2, axis=1)))
        errs.append(e)
        cum = cumulative_arc_length(a[:100, xo:xo+3])
        if variant == "arcmean":
            clocks.append((float(cum[-1]), float(np.linalg.norm(a[100, vsl]))))
        else:
            clocks.append((float(cum[-1]), float(np.median(a[:, 14 + k]))))
if variant != "time":
    print(f"  token shape {a.shape}; codec E_time(40f) RMS over arms/samples = {np.sqrt(np.mean(np.square(errs))):.4f} m "
          f"(median {np.median(errs):.4f}); span/speed samples: {[(round(c,3), round(v,3)) for c, v in clocks[:6]]}")
print(f"[{variant}] SMOKE_OK")
