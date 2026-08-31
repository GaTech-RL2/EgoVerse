"""Build a small local mirror of what the launcher stages on the node.

Same shape as the real thing: several control modes flattened into ONE
directory with mode-prefixed episode names. Prefixing is the part worth
rehearsing — every cell uses identical episode filenames, so an unprefixed
copy silently keeps only the last mode.
"""
import os
import shutil
import sys

SRC = os.path.expanduser("~/Desktop/GEAR/sim_run/ds_gen")
DST = os.path.expanduser("~/Desktop/GEAR/sim_run/local_stage/train/gripper/T")
# ideal is excluded: 714/1000 of its episodes are unreadable (zarr 3.1.0).
# jittery is excluded because it is the held-out mode.
MODES = ["tight", "loose", "laggy", "sticky"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 25

if os.path.isdir(DST):
    shutil.rmtree(DST)
os.makedirs(DST)

for mode in MODES:
    src = f"{SRC}/{mode}/gripper/T"
    eps = sorted(os.listdir(src))[:N]
    for ep in eps:
        shutil.copytree(f"{src}/{ep}", f"{DST}/{mode}__{ep}")
    print(f"staged {mode}: {len(eps)}")

names = os.listdir(DST)
print(f"TOTAL {len(names)} episodes in {DST}")
assert len(names) == len(MODES) * N, "prefix collision — episodes were lost"
assert not any(n.startswith("jittery__") for n in names), "held-out mode leaked"
