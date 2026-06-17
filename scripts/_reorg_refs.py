"""Dump each model config's defaults-list config refs + role classification.

role: 'both' (referenced AND directly-selected -> must split),
      'base' (referenced, not directly-selected -> composed-only -> bare refs),
      'leaf' (selectable -> qualified refs).
"""
import os
import re
import subprocess
import sys

import yaml

GROUP = sys.argv[1] if len(sys.argv) > 1 else "model"
CFG = f"egomimic/hydra_configs/{GROUP}"

# directly-selected set (from scripts/py): group=NAME
sel = set(re.findall(
    rf"{GROUP}=([a-zA-Z0-9_.]+)",
    subprocess.run(["grep", "-rhoE", rf"{GROUP}=[a-zA-Z0-9_.]+",
                    "--include=*.sh", "--include=*.py", "."],
                   capture_output=True, text=True).stdout,
))

refs_by = {}     # config -> set of config-name refs in its defaults
referenced = set()
for fn in sorted(os.listdir(CFG)):
    if not fn.endswith(".yaml"):
        continue
    stem = fn[:-5]
    try:
        doc = yaml.safe_load(open(os.path.join(CFG, fn))) or {}
    except Exception:
        doc = {}
    refs = set()
    for d in (doc.get("defaults") or []):
        if isinstance(d, str) and d not in ("_self_",) and not d.startswith("override"):
            refs.add(d)
        # dict-form entries (e.g. {override hydra/launcher: x}) ignored
    refs_by[stem] = refs
    referenced |= refs

def role(stem):
    r = stem in referenced
    s = stem in sel
    if r and s:
        return "both"
    if r and not s:
        return "base"
    return "leaf"

print("== referenced configs:", sorted(referenced))
print("== directly-selected (subset):", sorted(s for s in sel if s in refs_by))
print()
for stem in sorted(refs_by):
    if refs_by[stem]:
        print(f"[{role(stem):4}] {stem}  ->  defaults: {sorted(refs_by[stem])}")
print()
print("BOTH-CASE (must split):", sorted(s for s in refs_by if role(s) == "both"))
