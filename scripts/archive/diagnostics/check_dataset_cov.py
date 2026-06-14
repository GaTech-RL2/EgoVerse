import numpy as np
import zarr
from pathlib import Path

data_dir = Path("/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle")
eps = sorted([p for p in data_dir.iterdir() if p.name.endswith(".zarr")])[:10]

for ep in eps:
    z = zarr.open_group(str(ep), mode="r")
    reward = np.asarray(z["reward"][:]) if "reward" in z else None
    actions = np.asarray(z["actions"][:])
    state = np.asarray(z["observations.state"][:])
    
    max_rew = float(reward.max()) if reward is not None else -1
    final_rew = float(reward[-1]) if reward is not None else -1
    
    # Check attrs
    attrs = dict(z.attrs) if hasattr(z, 'attrs') else {}
    
    print(f"{ep.name}: T={actions.shape[0]}, final_reward={final_rew:.3f}, "
          f"max_reward={max_rew:.3f}, attrs={list(attrs.keys())[:5]}")
