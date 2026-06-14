"""Full data-group identity sweep: re-dump EVERY data config (baseline method)
and diff against scratch/config_phase2_baseline/resolved/data/<name>.json.
Confirms the refactor introduced ZERO resolved-output changes anywhere in the
group (touched AND untouched files)."""
import json
import os
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CONFIG_DIR = os.path.join(REPO, "egomimic/hydra_configs")
BASELINE = os.path.join(REPO, "scratch/config_phase2_baseline/resolved/data")
DATA_DIR = os.path.join(CONFIG_DIR, "data")
ENTRY = "train_zarr_cartesian"

names = []
for fn in sorted(os.listdir(DATA_DIR)):
    if fn.startswith("._") or not fn.endswith(".yaml"):
        continue
    if fn.startswith("_"):
        continue  # base files have no baseline json (new) and aren't composed standalone
    names.append(fn[:-5])

passed, failed, nobaseline = [], [], []
for name in names:
    bp = os.path.join(BASELINE, name + ".json")
    if not os.path.exists(bp):
        nobaseline.append(name)
        continue
    with open(bp) as f:
        baseline = f.read()
    try:
        with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
            cfg = compose(config_name=ENTRY, overrides=[f"data={name}"])
            got = json.dumps(OmegaConf.to_container(cfg.data, resolve=True),
                             sort_keys=True, indent=2) + "\n"
    except Exception as e:  # noqa: BLE001
        failed.append((name, f"compose raised: {repr(e)[:160]}"))
        continue
    if got == baseline:
        passed.append(name)
    else:
        failed.append((name, "DIFF"))

print(f"PASSED ({len(passed)}): {', '.join(passed)}")
print(f"NO-BASELINE-JSON ({len(nobaseline)}): {', '.join(nobaseline)}")
print(f"FAILED ({len(failed)}):")
for n, why in failed:
    print(f"   {n}: {why}")
print("VERDICT:", "ALL DATA CONFIGS IDENTICAL TO BASELINE" if not failed else "REGRESSION DETECTED")
