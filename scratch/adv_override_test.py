"""Decisive test: does the smoke launcher's override set apply against the
CURRENT config dir AND against the PRISTINE-originals config dir identically?

If both fail the same way -> the failure is pre-existing (launcher/Hydra quirk),
NOT a refactor regression. If current fails but pristine succeeds -> regression.
"""
import os
import shutil
import sys
import tempfile

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CUR = os.path.join(REPO, "egomimic/hydra_configs")
ORIG = os.path.join(REPO, "scratch/config_phase2_baseline/originals")

NC3 = "/coc/flash7/paphiwetsa3/datasets/new_circle_3"
KM = "egomimic.rldb.embodiment.pushshapes.get_keymap_eval"

# the exact data.* overrides the launcher passes
OVERRIDES = [
    "data=tsimulation",
    "model=bc_rnn_pushshapes",
    "data.train_datasets.pushshapes_sim.resolver.folder_path=" + NC3,
    "data.train_datasets.pushshapes_sim.resolver.key_map._target_=" + KM,
    "data.valid_datasets.pushshapes_sim.resolver.folder_path=" + NC3,
    "data.valid_datasets.pushshapes_sim.resolver.key_map._target_=" + KM,
]


def build_temp_configdir(src):
    """Copy a pristine config tree into a temp dir so initialize_config_dir can use it."""
    tmp = tempfile.mkdtemp(prefix="adv_cfg_")
    dst = os.path.join(tmp, "hydra_configs")
    shutil.copytree(src, dst)
    return dst


def run(label, config_dir):
    try:
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="train_zarr_cartesian", overrides=OVERRIDES)
        print("%-10s : COMPOSE_OK" % label)
        return True, None
    except Exception as e:
        print("%-10s : FAIL %s" % (label, repr(e).split(chr(10))[0][:160]))
        return False, repr(e)


print("=== current config dir ===")
cur_ok, cur_err = run("CURRENT", CUR)
print("=== pristine-originals config dir (temp copy) ===")
tmp_orig = build_temp_configdir(ORIG)
orig_ok, orig_err = run("PRISTINE", tmp_orig)

print("\n=== VERDICT ===")
if cur_ok == orig_ok:
    if cur_ok:
        print("BOTH COMPOSE OK -> launcher fine, no issue")
    else:
        print("BOTH FAIL IDENTICALLY -> PRE-EXISTING launcher/Hydra issue, NOT a refactor regression")
else:
    print("DIVERGENCE -> current=%s pristine=%s : POSSIBLE REGRESSION" % (cur_ok, orig_ok))
