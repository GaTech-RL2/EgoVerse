"""Map exactly which modified data groups break the launcher's
valid_datasets.<ds>.resolver.folder_path override (current vs pristine).

For each data group + its valid dataset key, try the override on CURRENT and on
PRISTINE; classify regression vs pre-existing vs both-ok.
"""
import os
import shutil
import tempfile

from hydra import compose, initialize_config_dir

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CUR = os.path.join(REPO, "egomimic/hydra_configs")
ORIG = os.path.join(REPO, "scratch/config_phase2_baseline/originals")

# (data_group, valid_dataset_key)
CASES = [
    ("tsimulation", "pushshapes_sim"),
    ("tsimulation_delta", "pushshapes_sim"),
    ("tsimulation_hpt", "pushshapes_sim"),
    ("tsimulation_hpt_causal", "pushshapes_sim"),
    ("tsimulation_hpt_fast", "pushshapes_sim"),
    ("bc_pickplace_eva_qwen", "eva_bimanual"),
    ("cotrain_pickplace_qwen", "eva_bimanual"),
    ("cotrain_pickplace_qwen_objgen", "eva_bimanual"),
    ("cotrain_pickplace_qwen_wrist", "eva_bimanual"),
    ("cotrain_pi_pickplace_qwen", "eva_bimanual"),
]
ENTRY = "train_zarr_cartesian"


def make_tmp(src):
    tmp = tempfile.mkdtemp(prefix="advmap_")
    dst = os.path.join(tmp, "hc")
    shutil.copytree(src, dst)
    return dst


def try_override(config_dir, dgrp, vkey):
    ov = [
        "data=" + dgrp,
        "data.valid_datasets.%s.resolver.folder_path=/tmp/X" % vkey,
    ]
    try:
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            compose(config_name=ENTRY, overrides=ov)
        return True, None
    except Exception as e:
        return False, repr(e).split(chr(10))[0][:90]


tmp_orig = make_tmp(ORIG)
print("%-32s %-10s %-10s %s" % ("data_group", "CURRENT", "PRISTINE", "verdict"))
regressions = []
for dgrp, vkey in CASES:
    cok, cerr = try_override(CUR, dgrp, vkey)
    ook, oerr = try_override(tmp_orig, dgrp, vkey)
    if cok == ook:
        verdict = "both_ok" if cok else "both_fail(pre-existing)"
    elif ook and not cok:
        verdict = "*** REGRESSION ***"
        regressions.append(dgrp)
    else:
        verdict = "current_ok_pristine_fail(?)"
    print("%-32s %-10s %-10s %s" % (dgrp, cok, ook, verdict))

print("\nREGRESSED GROUPS (%d): %s" % (len(regressions), regressions))
