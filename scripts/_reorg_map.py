"""Compute the flat-name -> subfolder/stripped-name mapping for the hydra
config reorg (model or data group). Prints the mapping; asserts no collisions.

Usage: python scripts/_reorg_map.py model
       python scripts/_reorg_map.py data
"""
import os
import re
import sys

GROUP = sys.argv[1]
CFG_DIR = os.path.abspath(f"egomimic/hydra_configs/{GROUP}")

# (family-folder, regex that matches the flat filename stem, prefix-strip regex).
# Order matters: first match wins. A stem that matches no rule stays flat.
MODEL_RULES = [
    ("bc_rnn", r"^_?bc_?rnn_", r"^(_?)bc_?rnn_"),
    ("dfot",   r"^dfot_",      r"^dfot_"),
    ("hnet",   r"^_?hnet_",    r"^(_?)hnet_"),
    ("hpt",    r"^_?hpt_",     r"^(_?)hpt_"),
    ("pi",     r"^pi0\.5_",    r"^pi0\.5_"),
    ("vae",    r"^vae_",       r"^vae_"),
]

# data: family folder keyed by EXPLICIT prefix; pickplace pulled out of cotrain.
DATA_RULES = [
    ("pickplace", r"^_?pickplace_qwen|^cotrain_pickplace_qwen|^cotrain_pi_pickplace_qwen|^bc_pickplace_eva_qwen|^aria_qwen",
     r"^(_?)(cotrain_)?(pi_)?(pickplace_qwen|pickplace_eva_qwen)_?|^bc_|^aria_qwen$"),
    ("tsimulation", r"^_?tsim", r"^(_?)tsim(ulation)?_?"),
    ("gmm",     r"^gmm_",     r"^gmm_"),
    ("cotrain", r"^cotrain_", r"^cotrain_"),
    ("aria",    r"^aria",     r"^aria_?"),
    ("eva",     r"^eva",      r"^eva_?"),
    ("mecka",   r"^mecka",    r"^mecka_?"),
    ("scale",   r"^scale",    r"^scale_?"),
]

RULES = MODEL_RULES if GROUP == "model" else DATA_RULES
# files we deliberately leave flat (singletons / out-of-family)
KEEP_FLAT = {"act", "egobridge", "industry_eva_pi", "video_clips", "__init__"}

mapping = {}
flat = []
for fn in sorted(os.listdir(CFG_DIR)):
    if not fn.endswith(".yaml"):
        continue
    stem = fn[:-5]
    if stem in KEEP_FLAT:
        flat.append(stem)
        continue
    fam = None
    new_stem = stem
    for folder, match_re, strip_re in RULES:
        if re.match(match_re, stem):
            fam = folder
            new_stem = re.sub(strip_re, r"\1" if "(_?)" in strip_re else "", stem)
            new_stem = new_stem.lstrip("_") and new_stem or new_stem  # noop guard
            break
    if fam is None:
        flat.append(stem)
        continue
    if new_stem == "" or new_stem == "_":
        new_stem = folder  # anchor whose name == family
    mapping[stem] = f"{fam}/{new_stem}"

# collision check: two old names mapping to the same new path
seen = {}
collisions = []
for old, new in mapping.items():
    if new in seen:
        collisions.append((seen[new], old, new))
    seen[new] = old

print(f"== {GROUP}: {len(mapping)} files mapped, {len(flat)} left flat ==")
for old in sorted(mapping):
    print(f"  {old}  ->  {mapping[old]}")
print(f"\n-- left flat: {sorted(flat)}")
if collisions:
    print(f"\n!! COLLISIONS: {collisions}")
    sys.exit(1)
print("\nno collisions")
