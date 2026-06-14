#!/usr/bin/env python
"""Build /coc/flash7/paphiwetsa3/datasets/new_circle_3_bal200 as 200 symlinks
to the selected obs0 episodes, and write SELECTION.md inside it.
"""
import json
import shutil
from pathlib import Path

SRC = Path("/coc/flash7/paphiwetsa3/datasets/new_circle_3")
DST = Path("/coc/flash7/paphiwetsa3/datasets/new_circle_3_bal200")
SEL = Path("/coc/flash7/paphiwetsa3/projects/EgoVerse2/scripts/bal200_selection.json")
MD = Path("/coc/flash7/paphiwetsa3/projects/EgoVerse2/scripts/bal200_SELECTION.md")

sel = json.loads(SEL.read_text())
names = sel["selected"]
assert len(names) == 200 and len(set(names)) == 200, f"bad selection: {len(names)}"

if DST.exists():
    # only safe to clear if it's our symlink dir (no real data inside)
    for p in DST.iterdir():
        if p.is_symlink():
            p.unlink()
        elif p.name == "SELECTION.md":
            p.unlink()
        else:
            raise RuntimeError(f"refusing to remove non-symlink {p}")
else:
    DST.mkdir(parents=True)

n = 0
for name in names:
    target = SRC / name
    assert target.exists(), f"missing source {target}"
    link = DST / name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target)
    n += 1

# copy SELECTION.md into the dataset dir
shutil.copy(MD, DST / "SELECTION.md")

# verify
links = sorted(p for p in DST.iterdir() if p.is_symlink())
print(f"Created {n} symlinks; {len(links)} symlinks present in {DST}")
# verify each resolves to a real .zarr with zarr.json
ok = 0
for p in links:
    rp = p.resolve()
    if (rp / "zarr.json").exists():
        ok += 1
print(f"{ok}/{len(links)} symlinks resolve to a valid .zarr (zarr.json present)")
print(f"SELECTION.md present: {(DST / 'SELECTION.md').exists()}")
