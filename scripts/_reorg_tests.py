"""Rewrite hardcoded model config-name string lists in test files to the new
subfolder paths (handles both "name" and "name.yaml" quoted forms)."""
import os
import re

CFG = "egomimic/hydra_configs/model"
RULES = [
    ("bc_rnn", r"^_?bc_?rnn_", r"^(_?)bc_?rnn_"),
    ("dfot", r"^dfot_", r"^dfot_"),
    ("hnet", r"^_?hnet_", r"^(_?)hnet_"),
    ("hpt", r"^_?hpt_", r"^(_?)hpt_"),
    ("pi", r"^pi0\.5_", r"^pi0\.5_"),
    ("vae", r"^vae_", r"^vae_"),
]
KEEP_FLAT = {"act", "egobridge"}

# original flat stems = current (post-move) leaf names reverse-mapped is hard;
# instead derive mapping from the ORIGINAL names via git (pre-reorg HEAD list).
import subprocess
orig = subprocess.run(["git", "show", "HEAD:.", ], capture_output=True)  # noqa
# simpler: enumerate new tree and build new-path set; map by stripping fam.
mapping = {}  # old_flat_stem -> fam/newstem
for fam in os.listdir(CFG):
    d = os.path.join(CFG, fam)
    if not os.path.isdir(d):
        continue
    for fn in os.listdir(d):
        if not fn.endswith(".yaml") or fn.startswith("_"):
            continue
        newstem = fn[:-5]
        # reconstruct the old flat stem: fam + '_' + newstem, with pi/bcrnn quirks
        if fam == "pi":
            old = "pi0.5_" + newstem
        elif fam == "bc_rnn":
            old = "bc_rnn_" + newstem
        else:
            old = f"{fam}_{newstem}"
        mapping[old] = f"{fam}/{newstem}"

FILES = ["tests/test_config_compose.py", "tests/regression/test_hnet_yamls_load.py"]
keys = sorted(mapping, key=len, reverse=True)
for fp in FILES:
    if not os.path.exists(fp):
        continue
    t = open(fp).read()
    n = 0
    for old in keys:
        new = mapping[old]
        for suf in (".yaml", ""):
            pat = rf'(["\']){re.escape(old)}{re.escape(suf)}(["\'])'
            t2 = re.sub(pat, rf"\g<1>{new}{suf}\g<2>", t)
            if t2 != t:
                n += 1
                t = t2
    open(fp, "w").write(t)
    print(f"{fp}: {n} names rewritten")
