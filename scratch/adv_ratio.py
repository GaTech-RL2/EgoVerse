"""Ratio-leaf exception check + underscore-base abstractness.

The ratio leaf is gated against the BC-residuals agent's PRE-EDIT dump
(scratch/config_phase2_baseline/precheck/...ratio.json), not necessarily the
phase-2 baseline. Compare my freshly-resolved ratio dump to precheck.
"""
import json
import os

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
MINE = os.path.join(REPO, "scratch/adv_verify/resolved/model/bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist_ratio.json")
PRECHECK = os.path.join(REPO, "scratch/config_phase2_baseline/precheck/bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist_ratio.json")
POSTEDIT = os.path.join(REPO, "scratch/config_phase2_baseline/postedit/bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist_ratio.json")
BASE = os.path.join(REPO, "scratch/config_phase2_baseline/resolved/model/bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist_ratio.json")


def rd(p):
    with open(p) as f:
        return f.read()


for label, p in [("precheck(BC pre-edit)", PRECHECK), ("postedit(BC)", POSTEDIT), ("phase2-baseline", BASE)]:
    if os.path.exists(p):
        same = rd(MINE) == rd(p)
        print("ratio vs %-24s : %s" % (label, "IDENTICAL" if same else "DIFFERS"))
    else:
        print("ratio vs %-24s : MISSING" % label)
