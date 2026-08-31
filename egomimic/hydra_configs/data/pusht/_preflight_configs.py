"""Instantiate every cotrain config's transform the way hydra will.

A config key with no matching factory parameter raises TypeError only once the
job is on the node, AFTER the data pull -- which is how 8 of 9 sweep runs died
on an unexpected 'rotation_radius'. Run this before submitting.
"""
import glob, os, sys
import numpy as np, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from egomimic.rldb.embodiment.pushshapes import (
    get_planar_arc_length_transform_list as arc,
    get_planar_dense_transform_list as dense)

bad = []
here = os.path.dirname(__file__)
for f in sorted(glob.glob(os.path.join(here, "cotrain11_*.yaml"))):
    t = next(iter(yaml.safe_load(open(f))["train_datasets"].values()))["resolver"]["transform_list"]
    kw = {k: v for k, v in t.items() if k != "_target_"}
    fn = arc if "arc" in t["_target_"] else dense
    try:
        out = fn(**kw)[0].transform(
            {"actions": np.cumsum(np.random.RandomState(0).randn(80, 4), 0)})["actions"]
        print(f"  ok   {os.path.basename(f):44} -> {out.shape}")
    except Exception as e:
        bad.append(f); print(f"  FAIL {os.path.basename(f):44} {type(e).__name__}: {e}")
raise SystemExit(1 if bad else 0)
