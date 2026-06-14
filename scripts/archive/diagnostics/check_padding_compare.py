import zarr
import numpy as np
from pathlib import Path

print("=== new_circle_2 (original) ===")
data_dir = Path("/coc/cedarp-dxu345-0/Tsim_datasets2/new_circle_2/new_circle_2")
eps = sorted([p for p in data_dir.iterdir() if p.name.endswith(".zarr")])[:3]
for ep in eps:
    z = zarr.open_group(str(ep), mode="r")
    actions = np.asarray(z["actions"][:])
    n_trailing_zeros = 0
    for i in range(len(actions)-1, -1, -1):
        if np.allclose(actions[i], 0.0):
            n_trailing_zeros += 1
        else:
            break
    print(f"  {ep.name}: T={len(actions)}, trailing_zeros={n_trailing_zeros}, last_action={actions[-1]}")

print("\n=== new_circle_clean (cleaned) ===")
data_dir = Path("/coc/cedarp-dxu345-0/Tsim_datasets2/new_circle_clean")
eps = sorted([p for p in data_dir.iterdir() if p.name.endswith(".zarr")])[:3]
for ep in eps:
    z = zarr.open_group(str(ep), mode="r")
    actions = np.asarray(z["actions"][:])
    n_trailing_zeros = 0
    for i in range(len(actions)-1, -1, -1):
        if np.allclose(actions[i], 0.0):
            n_trailing_zeros += 1
        else:
            break
    print(f"  {ep.name}: T={len(actions)}, trailing_zeros={n_trailing_zeros}, last_action={actions[-1]}")

print("\n=== new_circle_clean_padded (padded) ===")
data_dir = Path("/coc/cedarp-dxu345-0/Tsim_datasets2/new_circle_clean_padded")
eps = sorted([p for p in data_dir.iterdir() if p.name.endswith(".zarr")])[:3]
for ep in eps:
    z = zarr.open_group(str(ep), mode="r")
    actions = np.asarray(z["actions"][:])
    n_trailing_zeros = 0
    for i in range(len(actions)-1, -1, -1):
        if np.allclose(actions[i], 0.0):
            n_trailing_zeros += 1
        else:
            break
    n_trailing_repeat = 0
    last = actions[-1]
    for i in range(len(actions)-2, -1, -1):
        if np.allclose(actions[i], last):
            n_trailing_repeat += 1
        else:
            break
    print(f"  {ep.name}: T={len(actions)}, trailing_zeros={n_trailing_zeros}, trailing_repeat={n_trailing_repeat}, last_action={actions[-1]}")
