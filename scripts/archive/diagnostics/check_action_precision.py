import numpy as np
import zarr
from pathlib import Path

ep = sorted([p for p in Path("/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle").iterdir() if p.name.endswith(".zarr")])[10]
z = zarr.open_group(str(ep), mode="r")
actions = np.asarray(z["actions"][:])
state = np.asarray(z["observations.state"][:])

print(f"actions dtype={actions.dtype}")
print(f"state dtype={state.dtype}")
print(f"\nActions around t=55-65 (where divergence starts):")
for t in range(55, 66):
    print(f"  t={t}: action={actions[t]} agent_state={state[t,:2]} "
          f"obj_state={state[t,2:5].round(4)}")

# Check if actions are integer-valued (rounded from mouse/screen coords)
frac = actions - np.round(actions)
print(f"\nAction fractional parts: max={np.abs(frac).max():.6f}")
print(f"All integer-valued: {np.allclose(frac, 0)}")
