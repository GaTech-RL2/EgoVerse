"""Probe entries the way the baseline did: compose tree (no resolve), and
confirm any failures match the baseline's documented methods.json verbatim.

This separates 'the compose TREE builds' (structural) from 'resolve=True needs
Hydra runtime' (expected for path interpolations)."""
import json
import os

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CONFIG_DIR = os.path.join(REPO, "egomimic/hydra_configs")
METHODS = os.path.join(REPO, "scratch/config_phase2_baseline/resolved/methods.json")
ENTRIES = ["train_zarr_cartesian", "train_zarr_cartesian_pi",
           "train_zarr_keypoints", "train_zarr_keypoint_wrist", "viz_language"]

with open(METHODS) as f:
    methods = json.load(f)

with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
    for e in ENTRIES:
        base_method = methods.get("entry_" + e, "")
        # 1. does the compose TREE build (no resolve)?
        try:
            cfg = compose(config_name=e, overrides=[])
            tree_ok = True
            tree_err = ""
        except Exception as ex:
            tree_ok = False
            tree_err = repr(ex)[:120]
        # baseline expectation: 'rawload (compose failed:' means tree/resolve failed in baseline too
        base_failed = base_method.startswith("rawload (compose failed")
        verdict = "MATCHES-BASELINE" if (not tree_ok and base_failed) or (tree_ok and not base_failed) else "MISMATCH"
        print("%-28s tree_build=%s base_documented=%s -> %s" % (
            e, tree_ok, "FAILED" if base_failed else "OK", verdict))
        if not tree_ok:
            print("      tree_err: %s" % tree_err)
