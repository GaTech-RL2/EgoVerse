import numpy as np
import zarr
from pathlib import Path

data_dir = Path("/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle")
eps = sorted([p for p in data_dir.iterdir() if p.name.endswith(".zarr")])

for idx in [0, 1, 3]:
    ep = eps[idx]
    z = zarr.open_group(str(ep), mode="r")
    reward = np.asarray(z["reward"][:]).squeeze()
    actions = np.asarray(z["actions"][:])
    state = np.asarray(z["observations.state"][:])

    # Find when reward first exceeds 0.95
    above = np.where(reward >= 0.95)[0]
    peak_idx = int(np.argmax(reward))

    print(f"\n=== {ep.name} (T={len(reward)}) ===")
    print(f"Peak reward: {reward[peak_idx]:.4f} at t={peak_idx}")
    if len(above) > 0:
        print(f"First >=0.95 at t={above[0]}, last >=0.95 at t={above[-1]}")
    else:
        print("Never reaches 0.95!")

    # Show reward around the peak
    start = max(0, peak_idx - 3)
    end = min(len(reward), peak_idx + 10)
    print(f"Reward around peak [{start}:{end}]: {reward[start:end].round(3).tolist()}")

    # Check if actions are zero-padded after peak
    if peak_idx + 1 < len(actions):
        post_peak = actions[peak_idx+1:]
        n_zeros = np.sum(np.all(post_peak == 0, axis=1))
        print(f"Actions after peak: {len(post_peak)} steps, {n_zeros} are [0,0]")
        print(f"  actions[peak-1:peak+5] = {actions[max(0,peak_idx-1):peak_idx+5].tolist()}")

    # Check state movement after peak
    if peak_idx + 5 < len(state):
        print(f"  state agent_pos around peak:")
        for t in range(max(0, peak_idx-1), min(len(state), peak_idx+5)):
            print(f"    t={t}: agent={state[t,:2].round(1)}, obj={state[t,2:4].round(1)}")
