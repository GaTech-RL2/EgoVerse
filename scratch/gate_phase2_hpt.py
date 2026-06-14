"""Self-gate for the phase-2 HPT cluster inheritance refactor.

For each refactored leaf: compose train_zarr_cartesian with model=<name>, dump
cfg.model resolved (sort_keys, indent=2) exactly the way the baseline dumper did,
and diff against scratch/config_phase2_baseline/resolved/model/<name>.json.

Identity bar: every diff EMPTY.
"""
import json
import os
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CONFIG_DIR = os.path.join(REPO, "egomimic/hydra_configs")
BASE_DIR = os.path.join(REPO, "scratch/config_phase2_baseline/resolved/model")
OUT_DIR = os.path.join(REPO, "scratch/config_phase2_hpt_cluster/postedit")
ENTRY = "train_zarr_cartesian"

# 9 refactored leaves (compose method, per methods.json)
LEAVES = [
    "hpt_pushshapes_circle",
    "hpt_pushshapes_circle_regression",
    "hpt_pushshapes_simpleconv",
    "hpt_bc_flow_aria",
    "hpt_bc_flow_mecka",
    "hpt_bc_flow_scale",
    "hpt_bc_flow_eva",
    "hpt_bc_pickplace_qwen_pertoken",
    "hpt_bc_pickplace_qwen_pooled",
]


def dump_resolved(name):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name=ENTRY, overrides=[f"model={name}"])
        return OmegaConf.to_container(cfg.model, resolve=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    all_ok = True
    for name in LEAVES:
        try:
            container = dump_resolved(name)
        except Exception as e:  # noqa: BLE001
            results[name] = f"COMPOSE_FAIL: {e}"
            all_ok = False
            print(f"FAIL {name}: compose error {e}", flush=True)
            continue
        out_path = os.path.join(OUT_DIR, name + ".json")
        with open(out_path, "w") as f:
            json.dump(container, f, sort_keys=True, indent=2)
            f.write("\n")
        with open(os.path.join(BASE_DIR, name + ".json")) as f:
            baseline = json.load(f)
        if container == baseline:
            results[name] = "IDENTICAL"
            print(f"OK   {name}: IDENTICAL", flush=True)
        else:
            results[name] = "DIFF"
            all_ok = False
            print(f"DIFF {name}: NOT identical -> see {out_path}", flush=True)
    print("\n==== SUMMARY ====", flush=True)
    for k, v in results.items():
        print(f"{v:12s} {k}", flush=True)
    print(f"\nALL_IDENTICAL={all_ok}", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
