"""INSTANTIATION spot-check: for representative refactored models, build
cfg.model.robomimic_model from the CURRENT config tree and from the PRISTINE
originals tree, then compare state_dict() key-sets + shapes under a fixed seed.

Uses a real norm_stats loaded from the launcher's norm_stats.json. If a model
needs data-derived shapes that norm_stats doesn't cover, we fall back to
comparing the fully-resolved robomimic_model config dicts (already proven
identical) and note instantiation was config-equivalent.
"""
import json
import os
import shutil
import tempfile

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import hydra

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CUR = os.path.join(REPO, "egomimic/hydra_configs")
ORIG = os.path.join(REPO, "scratch/config_phase2_baseline/originals")
NORM = os.path.join(REPO, "logs/hnet_smoke/fused_nochunk_nc3_5ep_2026-05-30_23-34-47/norm_stats/norm_stats.json")

# representative per refactored cluster + the q-chained chunk4_q
REPS = [
    ("hnet_pushshapes_fused_windowed_crossattn_resnet", "tsimulation"),
    ("hnet_pushshapes_big", "tsimulation"),
    ("hpt_bc_flow_eva", "tsimulation_hpt"),
    ("bc_rnn_pushshapes_paperexact_tx_chunk4_q", "tsimulation"),
]
ENTRY = "train_zarr_cartesian"


def make_tmp(src):
    tmp = tempfile.mkdtemp(prefix="advinst_")
    dst = os.path.join(tmp, "hc")
    shutil.copytree(src, dst)
    return dst


def load_norm_stats():
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
    with open(NORM) as f:
        state = json.load(f)
    return MultiDataset, state


def resolved_model(config_dir, model, dgrp):
    # IMPORTANT: do not pass launcher data.valid_datasets overrides here; we only
    # need cfg.model resolved, which is data-group-driven for interpolations.
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=ENTRY, overrides=["model=%s" % model, "data=%s" % dgrp])
        return OmegaConf.to_container(cfg.model.robomimic_model, resolve=True), cfg


def state_summary(module):
    sd = module.state_dict()
    return {k: tuple(v.shape) for k, v in sd.items()}


def try_instantiate(cfg, MultiDataset, state):
    norm = MultiDataset.from_state(state)
    torch.manual_seed(0)
    return hydra.utils.instantiate(cfg.model.robomimic_model, norm_stats=norm)


def main():
    tmp_orig = make_tmp(ORIG)
    try:
        MultiDataset, state = load_norm_stats()
        ns_ok = True
    except Exception as e:
        ns_ok = False
        print("norm_stats load failed: %r -> config-equivalence fallback only" % e)

    for model, dgrp in REPS:
        cur_dict, cur_cfg = resolved_model(CUR, model, dgrp)
        orig_dict, orig_cfg = resolved_model(tmp_orig, model, dgrp)
        cfg_same = json.dumps(cur_dict, sort_keys=True) == json.dumps(orig_dict, sort_keys=True)
        line = "%-50s cfg_identical=%s" % (model, cfg_same)
        if ns_ok:
            try:
                m_cur = try_instantiate(cur_cfg, MultiDataset, state)
                m_org = try_instantiate(orig_cfg, MultiDataset, state)
                s_cur, s_org = state_summary(m_cur), state_summary(m_org)
                keys_same = set(s_cur) == set(s_org)
                shapes_same = s_cur == s_org
                line += "  | INSTANTIATED keys_same=%s shapes_same=%s nparams=%d" % (
                    keys_same, shapes_same, len(s_cur))
            except Exception as e:
                line += "  | instantiate_err=%r" % (repr(e)[:120])
        print(line)


if __name__ == "__main__":
    main()
