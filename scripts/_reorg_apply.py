"""Apply the hydra-config subfolder reorg for one group (model or data).

Steps: (1) compute flat-name -> fam/stripped-name mapping; (2) git mv files
into family subfolders; (3) rewrite intra-group `defaults:` refs in the moved
files to the new bare names; (4) rewrite external selectors (group=NAME /
group: NAME) across scripts/python/train-configs/tests/docs.

Run on a compute node:  srun ... .venv/bin/python scripts/_reorg_apply.py model
"""
import os
import re
import subprocess
import sys

GROUP = sys.argv[1]
ROOT = os.path.abspath(".")
CFG = f"egomimic/hydra_configs/{GROUP}"

MODEL_RULES = [
    ("bc_rnn", r"^_?bc_?rnn_", r"^(_?)bc_?rnn_"),
    ("dfot",   r"^dfot_",      r"^dfot_"),
    ("hnet",   r"^_?hnet_",    r"^(_?)hnet_"),
    ("hpt",    r"^_?hpt_",     r"^(_?)hpt_"),
    ("pi",     r"^pi0\.5_",    r"^pi0\.5_"),
    ("vae",    r"^vae_",       r"^vae_"),
]
KEEP_FLAT = {"act", "egobridge", "industry_eva_pi", "video_clips", "__init__"}
RULES = MODEL_RULES  # data handled in a later run with its own rules

def strip(stem, strip_re):
    has_grp = "(_?)" in strip_re
    out = re.sub(strip_re, (r"\1" if has_grp else ""), stem)
    return out

# 1. mapping: old_stem -> (fam, new_bare_stem, "fam/new_bare_stem")
mapping = {}
for fn in sorted(os.listdir(CFG)):
    if not fn.endswith(".yaml"):
        continue
    stem = fn[:-5]
    if stem in KEEP_FLAT:
        continue
    for folder, mre, sre in RULES:
        if re.match(mre, stem):
            new = strip(stem, sre) or folder
            mapping[stem] = (folder, new, f"{folder}/{new}")
            break

old_to_bare = {o: m[1] for o, m in mapping.items()}        # intra-file defaults
old_to_path = {o: m[2] for o, m in mapping.items()}        # external selectors
folders = sorted({m[0] for m in mapping.values()})

# 2. git mv
for folder in folders:
    os.makedirs(os.path.join(CFG, folder), exist_ok=True)
moves = 0
for old, (folder, new, path) in mapping.items():
    src = os.path.join(CFG, old + ".yaml")
    dst = os.path.join(CFG, folder, new + ".yaml")
    subprocess.run(["git", "mv", src, dst], check=True, cwd=ROOT)
    moves += 1

# 3. rewrite intra-group defaults refs in the moved files to the SUBFOLDER-
# QUALIFIED path (Hydra resolves bare defaults names at the group ROOT, so a
# same-subfolder sibling must be referenced as "fam/newstem").
path_keys = sorted(old_to_path, key=len, reverse=True)
def rewrite_defaults(text):
    out = text
    for old in path_keys:
        new = old_to_path[old]
        out = re.sub(
            rf"(^\s*-\s*){re.escape(old)}(?=\s*$|\s+#)",
            rf"\g<1>{new}",
            out, flags=re.M,
        )
    return out

moved_files = [os.path.join(CFG, m[0], m[1] + ".yaml") for m in mapping.values()]
defaults_edits = 0
pkg_added = 0
for fp in moved_files:
    with open(fp) as fh:
        t = fh.read()
    nt = rewrite_defaults(t)
    # Subfolder configs default to package model.<sub>; force the group package
    # so they populate cfg.<group> flat (both when selected and when composed).
    if not re.search(r"^#\s*@package", nt, re.M):
        nt = f"# @package {GROUP}\n" + nt
        pkg_added += 1
    if nt != t:
        with open(fp, "w") as fh:
            fh.write(nt)
        defaults_edits += 1

# 4. rewrite external selectors group=NAME and "group: NAME"
sel_keys = sorted(old_to_path, key=len, reverse=True)
BOUND = r"(?=[\s'\"`)\],#]|$)"
def rewrite_selectors(text):
    out = text
    for old in sel_keys:
        new = old_to_path[old]
        # group=NAME  (shell / cli / python f-strings)
        out = re.sub(rf"({GROUP}=){re.escape(old)}{BOUND}", rf"\g<1>{new}", out)
        # group: NAME  (yaml defaults list, e.g. "- model: hpt_bc_flow_eva")
        out = re.sub(rf"({GROUP}:\s*){re.escape(old)}{BOUND}", rf"\g<1>{new}", out)
    return out

ext_files = []
for base in ("egomimic", "scripts", "tests"):
    for dirpath, _dirs, files in os.walk(os.path.join(ROOT, base)):
        if "/.git" in dirpath or "/.venv" in dirpath or "/hydra_configs/" in dirpath:
            continue
        for f in files:
            if f.endswith((".sh", ".py", ".md")):
                ext_files.append(os.path.join(dirpath, f))
# repo-root docs + top-level train configs (model:/data: defaults live here)
for f in os.listdir(ROOT):
    if f.endswith((".sh", ".md")):
        ext_files.append(os.path.join(ROOT, f))
for f in os.listdir("egomimic/hydra_configs"):
    if f.endswith(".yaml"):
        ext_files.append(os.path.abspath(f"egomimic/hydra_configs/{f}"))

# don't rewrite the migrator/mapping helpers themselves
SKIP = {"_reorg_apply.py", "_reorg_map.py"}
sel_edits = 0
for fp in ext_files:
    if os.path.basename(fp) in SKIP:
        continue
    try:
        with open(fp) as fh:
            t = fh.read()
    except (UnicodeDecodeError, IsADirectoryError):
        continue
    nt = rewrite_selectors(t)
    if nt != t:
        with open(fp, "w") as fh:
            fh.write(nt)
        sel_edits += 1

print(f"{GROUP}: moved={moves}, files-edited={defaults_edits}, @package-added={pkg_added}, external-files-edited={sel_edits}")
