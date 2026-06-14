"""Launcher + underscore-base reference audit.

1. Grep scripts/ for model=/data=/evaluator=/callbacks=/trainer=/logger= override names.
2. Confirm each referenced config file exists in egomimic/hydra_configs/<group>/.
3. Confirm NO script references an underscore-prefixed (abstract base) config.
4. Confirm no underscore base is referenced anywhere under scripts/.
"""
import os
import re

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
SCRIPTS = os.path.join(REPO, "scripts")
CFG = os.path.join(REPO, "egomimic/hydra_configs")

GROUPS = ("model", "data", "evaluator", "callbacks", "trainer", "logger",
          "paths", "data_schematic")
# match e.g. model=foo  +model=foo  model=foo.bar (hydra group override)
pat = re.compile(r"(?<![\w./-])(" + "|".join(GROUPS) + r")=([A-Za-z0-9_./-]+)")

refs = {}  # (group,name) -> set(files)
for root, _d, files in os.walk(SCRIPTS):
    for fn in files:
        p = os.path.join(root, fn)
        try:
            with open(p, "r", errors="replace") as f:
                txt = f.read()
        except Exception:
            continue
        for m in pat.finditer(txt):
            g, name = m.group(1), m.group(2)
            name = name.strip().strip("\"'")
            refs.setdefault((g, name), set()).add(os.path.relpath(p, REPO))

missing = []
underscore_refs = []
for (g, name), where in sorted(refs.items()):
    base = name.split("/")[-1]
    yaml_path = os.path.join(CFG, g, name + ".yaml")
    exists = os.path.isfile(yaml_path)
    if base.startswith("_"):
        underscore_refs.append((g, name, sorted(where)))
    if not exists:
        # tolerate values that are clearly not config names (e.g. numeric)
        missing.append((g, name, sorted(where)))

print("TOTAL distinct group-override refs in scripts/: %d" % len(refs))
print("=== MISSING (referenced but no yaml) ===")
if missing:
    for g, name, where in missing:
        print("  MISSING %s=%s  in %s" % (g, name, where))
else:
    print("  none")
print("=== UNDERSCORE-BASE refs from scripts/ (should be NONE) ===")
if underscore_refs:
    for g, name, where in underscore_refs:
        print("  UNDERSCORE %s=%s  in %s" % (g, name, where))
else:
    print("  none")

# also: list all underscore bases on disk and grep scripts for the bare stem
print("=== underscore bases on disk + any scripts/ mention of the stem ===")
for g in GROUPS:
    d = os.path.join(CFG, g)
    if not os.path.isdir(d):
        continue
    for fn in sorted(os.listdir(d)):
        if fn.startswith("_") and fn.endswith(".yaml"):
            stem = fn[:-5]
            hits = []
            for root, _dd, files in os.walk(SCRIPTS):
                for sf in files:
                    try:
                        with open(os.path.join(root, sf), errors="replace") as fh:
                            if stem in fh.read():
                                hits.append(os.path.relpath(os.path.join(root, sf), REPO))
                    except Exception:
                        pass
            print("  %s/%s : scripts mentions=%s" % (g, stem, hits if hits else "NONE"))
