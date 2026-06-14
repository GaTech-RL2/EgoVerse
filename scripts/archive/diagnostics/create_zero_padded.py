"""Create zero-padded version by copying matching episodes from new_circle_2.
new_circle_clean has 491 episodes (tails removed). The same episodes exist in
new_circle_2 WITH zero-padded tails. Copy those to new_circle_zero_padded."""
import shutil
from pathlib import Path

clean_dir = Path("/coc/cedarp-dxu345-0/Tsim_datasets2/new_circle_clean")
src_dir = Path("/coc/cedarp-dxu345-0/Tsim_datasets2/new_circle_2/new_circle_2")
dst_dir = Path("/coc/cedarp-dxu345-0/Tsim_datasets2/new_circle_zero_padded")

# Get episode names from clean (these are the 491 that passed quality filter)
clean_eps = sorted([p.name for p in clean_dir.iterdir() if p.name.endswith(".zarr")])
print(f"Found {len(clean_eps)} episodes in new_circle_clean")

# Verify they exist in new_circle_2
missing = [e for e in clean_eps if not (src_dir / e).exists()]
if missing:
    print(f"WARNING: {len(missing)} episodes not found in new_circle_2:")
    for m in missing[:5]:
        print(f"  {m}")
    print("Skipping those.")

# Create destination
dst_dir.mkdir(parents=True, exist_ok=True)

copied = 0
for ep_name in clean_eps:
    src_path = src_dir / ep_name
    dst_path = dst_dir / ep_name
    if src_path.exists():
        if not dst_path.exists():
            shutil.copytree(str(src_path), str(dst_path))
        copied += 1

print(f"Copied {copied} episodes to {dst_dir}")
