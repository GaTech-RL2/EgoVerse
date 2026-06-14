"""Full config-tree sweep: compare CURRENT hydra_configs against pristine
originals backup, byte-for-byte. Classify every yaml as:
  IDENTICAL  - unchanged from pristine
  MODIFIED   - content changed (a refactor edit)
  NEW        - exists now, not in pristine (new base / new leaf)
  DELETED    - in pristine, gone now (would break launchers)
Also flag stray .json/.bak backups and broken symlinks inside config dirs.
"""
import os

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CUR = os.path.join(REPO, "egomimic/hydra_configs")
ORIG = os.path.join(REPO, "scratch/config_phase2_baseline/originals")


def all_yaml(root):
    out = {}
    for r, _d, files in os.walk(root):
        for fn in files:
            if fn.endswith(".yaml") and not fn.startswith("._"):
                rel = os.path.relpath(os.path.join(r, fn), root)
                out[rel] = os.path.join(r, fn)
    return out


def rd(p):
    with open(p, "rb") as f:
        return f.read()


cur = all_yaml(CUR)
orig = all_yaml(ORIG)

identical, modified, new, deleted = [], [], [], []
for rel in sorted(set(cur) | set(orig)):
    if rel in cur and rel in orig:
        if rd(cur[rel]) == rd(orig[rel]):
            identical.append(rel)
        else:
            modified.append(rel)
    elif rel in cur:
        new.append(rel)
    else:
        deleted.append(rel)

print("IDENTICAL=%d  MODIFIED=%d  NEW=%d  DELETED=%d" % (
    len(identical), len(modified), len(new), len(deleted)))
print("=== MODIFIED (refactor edits) ===")
for r in modified:
    print("  M " + r)
print("=== NEW (added by refactor) ===")
for r in new:
    print("  N " + r)
print("=== DELETED (MUST be empty) ===")
for r in deleted:
    print("  D " + r)

# stray non-yaml junk in config dirs
print("=== STRAY non-yaml/.bak/.json/broken-symlink in hydra_configs ===")
stray = []
for r, _d, files in os.walk(CUR):
    for fn in files:
        p = os.path.join(r, fn)
        rel = os.path.relpath(p, CUR)
        if fn.endswith((".bak", ".json", ".orig", ".tmp", "~")) or ".bak" in fn:
            stray.append("backup: " + rel)
    # broken symlinks
    for name in os.listdir(r):
        fp = os.path.join(r, name)
        if os.path.islink(fp) and not os.path.exists(fp):
            stray.append("broken-symlink: " + os.path.relpath(fp, CUR))
if stray:
    for s in sorted(set(stray)):
        print("  " + s)
else:
    print("  none")
