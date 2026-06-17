"""Full folderization (Path B) for one hydra config group.

Rules discovered empirically about Hydra defaults resolution:
  * a SELECTED (primary) config's defaults refs are group-ROOT-relative  -> qualify "fam/name"
  * a COMPOSED (base) config's defaults refs are SUBFOLDER-relative      -> bare "name"
  * subfolder configs need `# @package <group>` to populate cfg.<group> flat
A config that is BOTH selected and composed AND has its own defaults refs is
split into `_<name>_base` (composed) + `<name>` (thin selected leaf).

Run on a compute node:  srun ... .venv/bin/python scripts/_reorg_b.py model
"""
import os
import re
import subprocess
import sys

import yaml

GROUP = sys.argv[1]
CFG = f"egomimic/hydra_configs/{GROUP}"
ROOT = os.path.abspath(".")

RULES = {
    "model": [
        ("bc_rnn", r"^_?bc_?rnn_", r"^(_?)bc_?rnn_"),
        ("dfot", r"^dfot_", r"^dfot_"),
        ("hnet", r"^_?hnet_", r"^(_?)hnet_"),
        ("hpt", r"^_?hpt_", r"^(_?)hpt_"),
        ("pi", r"^pi0\.5_", r"^pi0\.5_"),
        ("vae", r"^vae_", r"^vae_"),
    ],
    "data": [
        ("tsimulation", r"^_?tsim", r"^(_?)tsim(ulation)?_?"),
        ("gmm", r"^gmm_", r"^gmm_"),
        ("cotrain", r"^cotrain_", r"^cotrain_"),
        ("aria", r"^aria", r"^aria_?"),
        ("eva", r"^eva", r"^eva_?"),
        ("mecka", r"^mecka", r"^mecka_?"),
        ("scale", r"^scale", r"^scale_?"),
    ],
}[GROUP]
KEEP_FLAT = {
    "model": {"act", "egobridge", "__init__"},
    "data": {"_pickplace_qwen_base", "bc_pickplace_eva_qwen",
             "industry_eva_pi", "video_clips", "__init__"},
}[GROUP]


def fam_of(stem):
    if stem in KEEP_FLAT:
        return None
    for folder, mre, _ in RULES:
        if re.match(mre, stem):
            return folder
    return None


def newstem(stem, folder):
    for f, _, sre in RULES:
        if f == folder:
            has = "(_?)" in sre
            return re.sub(sre, (r"\1" if has else ""), stem) or folder
    return stem


# ---- 1. ref graph + roles ------------------------------------------------
sel = set(re.findall(
    rf"{GROUP}=([a-zA-Z0-9_.]+)",
    subprocess.run(["grep", "-rhoE", rf"{GROUP}=[a-zA-Z0-9_.]+",
                    "--include=*.sh", "--include=*.py", "."],
                   capture_output=True, text=True).stdout))
refs_by = {}
referenced = set()
stems = []
for fn in sorted(os.listdir(CFG)):
    if not fn.endswith(".yaml"):
        continue
    stem = fn[:-5]
    stems.append(stem)
    try:
        doc = yaml.safe_load(open(os.path.join(CFG, fn))) or {}
    except Exception:
        doc = {}
    refs = [d for d in (doc.get("defaults") or [])
            if isinstance(d, str) and d != "_self_" and not d.startswith("override")]
    refs_by[stem] = refs
    referenced |= set(refs)


def role(stem):
    r, s = stem in referenced, stem in sel
    if r and s:
        return "both"
    return "base" if r else "leaf"


# Split any NON-underscore config that is used as a base (referenced) AND has
# its own defaults refs: it must be primary-safe (qualified refs via a thin
# leaf) AND composed-safe (bare refs in the _base). Underscore configs are
# composed-only by convention, so they stay as pure bases (no split).
SPLIT = {s for s in stems
         if s in referenced and refs_by[s] and not s.startswith("_")}

# ---- 2. plan: where each config ends up + its "ref target" ---------------
# fam[stem], new component name used in refs (post-prefix-drop), and the
# group-relative path. For SPLIT configs the ref-target is the _base.
fam = {s: fam_of(s) for s in stems}
ns = {s: (newstem(s, fam[s]) if fam[s] else s) for s in stems}

def base_comp(s):       # component name a referrer should point at
    return f"_{ns[s]}_base" if s in SPLIT else ns[s]

def target_qualified(s):
    if fam[s] is None:
        return base_comp(s)          # flat -> bare top-level name
    return f"{fam[s]}/{base_comp(s)}"

def target_bare(s):
    return base_comp(s)              # same-subfolder bare

# ---- 3. moves + split file creation --------------------------------------
for folder in sorted({f for f in fam.values() if f}):
    os.makedirs(os.path.join(CFG, folder), exist_ok=True)

created_leaves = []
moved_or_base = []   # (path, stem, file_role)  file_role in {leaf,base}
for s in stems:
    if fam[s] is None:
        moved_or_base.append((os.path.join(CFG, s + ".yaml"), s, "leaf"))
        continue
    src = os.path.join(CFG, s + ".yaml")
    if s in SPLIT:
        base_path = os.path.join(CFG, fam[s], f"_{ns[s]}_base.yaml")
        subprocess.run(["git", "mv", src, base_path], check=True, cwd=ROOT)
        moved_or_base.append((base_path, s, "base"))            # body lives here
        leaf_path = os.path.join(CFG, fam[s], ns[s] + ".yaml")
        with open(leaf_path, "w") as fh:
            fh.write(f"# @package {GROUP}\ndefaults:\n  - {fam[s]}/_{ns[s]}_base\n  - _self_\n")
        created_leaves.append(leaf_path)
    else:
        dst = os.path.join(CFG, fam[s], ns[s] + ".yaml")
        subprocess.run(["git", "mv", src, dst], check=True, cwd=ROOT)
        moved_or_base.append((dst, s, "leaf" if role(s) != "base" else "base"))

# ---- 4. rewrite defaults refs in every moved/base/flat file --------------
def rewrite_file(path, stem, file_role):
    with open(path) as fh:
        t = fh.read()
    for R in refs_by.get(stem, []):
        if R not in refs_by:        # ref to something not in this group (skip)
            continue
        newref = target_bare(R) if file_role == "base" else target_qualified(R)
        t = re.sub(rf"(^\s*-\s*){re.escape(R)}(?=\s*$|\s+#)", rf"\g<1>{newref}", t, flags=re.M)
    # add @package for subfolder files (flat files already default to group pkg)
    if fam[stem] is not None and not re.search(r"^#\s*@package", t, re.M):
        t = f"# @package {GROUP}\n" + t
    with open(path, "w") as fh:
        fh.write(t)

for path, stem, file_role in moved_or_base:
    rewrite_file(path, stem, file_role)

# ---- 5. external selectors (group=NAME / group: NAME) --------------------
# old flat name -> new selection path (SPLIT -> the leaf)
sel_map = {}
for s in stems:
    if fam[s] is None:
        continue
    sel_map[s] = f"{fam[s]}/{ns[s]}"
keys = sorted(sel_map, key=len, reverse=True)
BOUND = r"(?=[\s'\"`)\],#]|$)"
ext = []
for base in ("egomimic", "scripts", "tests"):
    for dp, _d, files in os.walk(os.path.join(ROOT, base)):
        if "/.git" in dp or "/.venv" in dp or "/hydra_configs/" in dp:
            continue
        ext += [os.path.join(dp, f) for f in files if f.endswith((".sh", ".py", ".md"))]
for f in os.listdir(ROOT):
    if f.endswith((".sh", ".md")):
        ext.append(os.path.join(ROOT, f))
for f in os.listdir("egomimic/hydra_configs"):
    if f.endswith(".yaml"):
        ext.append(os.path.abspath(f"egomimic/hydra_configs/{f}"))
SKIP = {"_reorg_b.py", "_reorg_apply.py", "_reorg_map.py", "_reorg_verify.py", "_reorg_refs.py"}
sel_edits = 0
for fp in ext:
    if os.path.basename(fp) in SKIP:
        continue
    try:
        t = open(fp).read()
    except (UnicodeDecodeError, IsADirectoryError):
        continue
    nt = t
    for old in keys:
        new = sel_map[old]
        nt = re.sub(rf"({GROUP}=){re.escape(old)}{BOUND}", rf"\g<1>{new}", nt)
        nt = re.sub(rf"({GROUP}:\s*){re.escape(old)}{BOUND}", rf"\g<1>{new}", nt)
    if nt != t:
        open(fp, "w").write(nt)
        sel_edits += 1

print(f"{GROUP}: moved/base={len(moved_or_base)} split={sorted(SPLIT)} "
      f"created_leaves={len(created_leaves)} ext_edits={sel_edits}")
