"""Probe that all 5 entry yamls compose cleanly (like --cfg job)."""
import os
import traceback

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CONFIG_DIR = os.path.join(REPO, "egomimic/hydra_configs")
ENTRIES = ["train_zarr_cartesian", "train_zarr_cartesian_pi",
           "train_zarr_keypoints", "train_zarr_keypoint_wrist", "viz_language"]

with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
    for e in ENTRIES:
        try:
            cfg = compose(config_name=e, overrides=[])
            # force-resolve like --cfg job would render
            c = OmegaConf.to_container(cfg, resolve=True)
            nkeys = len(c) if isinstance(c, dict) else 0
            mt = c.get("model", {}).get("_target_", "?") if isinstance(c, dict) else "?"
            print("OK   %-28s top_keys=%d model._target_=%s" % (e, nkeys, mt))
        except Exception as ex:
            print("FAIL %-28s : %s" % (e, repr(ex)[:200]))
