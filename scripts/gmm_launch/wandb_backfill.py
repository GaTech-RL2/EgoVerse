"""Backfill the 7-run 3000-ep campaign's metrics.csv into a wandb project.
Re-syncable: a sidecar state file tracks the last-logged step per run, so re-running
only appends new rows (resume='allow'). Run on PACE (wandb auth'd) via srun --overlap.
"""
import glob, json, os
import pandas as pd
import wandb

PROJECT = "gmm-3000ep-campaign"
REPO = "/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm"
STATE = os.path.join(REPO, ".wandb_backfill_state.json")
RUNS = [
    "hnet_big_cot3000",
    "hnet_cot3000_c4r4", "hnet_cot3000_c4r8", "hnet_cot3000_c8r4",
    "hnet_cot3000_taper", "hnet_cot3000_taper_c8r2", "hnet_cot3000_taper_c8r4", "hnet_cot3000_taper_c4r8",
    "hnet_cot3000_taper_c4r8_regoff", "hnet_cot3000_taper_c8r2_regoff",
]

state = json.load(open(STATE)) if os.path.exists(STATE) else {}
urls = {}
for run in RUNS:
    paths = sorted(
        glob.glob(f"{REPO}/logs/{run}/*/csv_logs/lightning_logs/version_0/metrics.csv"),
        key=os.path.getmtime,
    )
    if not paths:
        print(f"  {run}: no metrics.csv yet — skip")
        continue
    df = pd.read_csv(paths[-1])
    if "step" not in df.columns:
        print(f"  {run}: no step column — skip")
        continue
    df = df.sort_values("step")
    w = wandb.init(project=PROJECT, id=run, name=run, resume="allow", reinit=True)
    urls[run] = w.url
    last = state.get(run, -1)
    new_last = last
    n = 0
    for step, grp in df.groupby("step"):
        if pd.isna(step) or int(step) <= last:
            continue
        data = {}
        for _, row in grp.iterrows():
            for k, v in row.items():
                if k not in ("step", "epoch") and pd.notna(v):
                    data[k] = float(v) if isinstance(v, (int, float)) else v
        if data:
            wandb.log(data, step=int(step))
            new_last = int(step)
            n += 1
    w.finish()
    state[run] = new_last
    print(f"  {run}: logged {n} new steps (up to {new_last})  {w.url}")

json.dump(state, open(STATE, "w"))
print("\n=== DONE. Project:", PROJECT, "===")
for r, u in urls.items():
    print(f"  {r}: {u}")
