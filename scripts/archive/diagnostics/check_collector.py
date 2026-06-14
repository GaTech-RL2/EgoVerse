import zarr, json
from pathlib import Path

data_dir = Path("/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle")
eps = sorted([p for p in data_dir.iterdir() if p.name.endswith(".zarr")])

# Check a few episodes for collector type and action precision
for idx in [0, 10, 100, 500]:
    if idx >= len(eps):
        continue
    z = zarr.open_group(str(eps[idx]), mode="r")
    desc = json.loads(z.attrs.get("task_description", "{}"))
    collector = desc.get("env_args", {}).get("collector", "unknown")
    actions = z["actions"][:]
    frac = actions - actions.round()
    has_frac = (abs(frac) > 0.001).any()
    print(f"ep[{idx}] {eps[idx].name}: collector={collector} "
          f"has_fractional_actions={has_frac} "
          f"action_sample={actions[50].tolist() if len(actions)>50 else actions[0].tolist()}")
