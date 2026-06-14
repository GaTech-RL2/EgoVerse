import numpy as np
import zarr
from pathlib import Path

data_dir = Path("/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle")
eps = sorted([p for p in data_dir.iterdir() if p.name.endswith(".zarr")])[:10]

for ep in eps:
    z = zarr.open_group(str(ep), mode="r")
    reward = np.asarray(z["reward"][:]) if "reward" in z else None
    actions = np.asarray(z["actions"][:])
    
    if reward is not None:
        print(f"{ep.name}: T={actions.shape[0]}, reward.shape={reward.shape}, "
              f"final_reward={reward[-1]}, max_reward={reward.max()}")
    else:
        print(f"{ep.name}: T={actions.shape[0]}, no reward key")
