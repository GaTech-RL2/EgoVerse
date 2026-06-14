import zarr
import numpy as np
from pathlib import Path

data_dir = Path("/coc/cedarp-dxu345-0/Tsim_datasets2/new_circle_clean")
eps = sorted([p for p in data_dir.iterdir() if p.name.endswith(".zarr")])[:5]

for ep in eps:
    z = zarr.open_group(str(ep), mode="r")
    actions = np.asarray(z["actions"][:])
    # Check last 15 actions
    print(f"\n{ep.name} (T={len(actions)}):")
    print("  Last 15 actions:")
    for i in range(max(0, len(actions)-15), len(actions)):
        print(f"    t={i}: [{actions[i][0]:.1f}, {actions[i][1]:.1f}]")
    # Check if tail is zeros or repeated
    n_trailing_zeros = 0
    for i in range(len(actions)-1, -1, -1):
        if np.allclose(actions[i], 0.0):
            n_trailing_zeros += 1
        else:
            break
    n_trailing_repeat = 0
    if len(actions) > 1:
        last = actions[-1]
        for i in range(len(actions)-2, -1, -1):
            if np.allclose(actions[i], last):
                n_trailing_repeat += 1
            else:
                break
    print(f"  trailing_zeros={n_trailing_zeros}, trailing_repeat={n_trailing_repeat}")
