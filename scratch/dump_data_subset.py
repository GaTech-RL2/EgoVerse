"""Dump resolved cfg.data JSON for the composable data configs only.

Mirrors dump_phase2_configs.dump_group exactly for the data group:
  compose(train_zarr_cartesian, data=<name>) -> OmegaConf.to_container(cfg.data, resolve=True)
  sort_keys=True, indent=2, trailing newline.

Usage: python dump_data_subset.py <outdir> <name1> <name2> ...
Writes <outdir>/<name>.json for each name that composes; prints FAIL on any that don't.
"""
import json
import os
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CONFIG_DIR = os.path.join(REPO, "egomimic/hydra_configs")
ENTRY = "train_zarr_cartesian"


def main():
    outdir = sys.argv[1]
    names = sys.argv[2:]
    os.makedirs(outdir, exist_ok=True)
    ok, fail = [], []
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        for name in names:
            try:
                cfg = compose(config_name=ENTRY, overrides=[f"data={name}"])
                container = OmegaConf.to_container(cfg.data, resolve=True)
                with open(os.path.join(outdir, name + ".json"), "w") as f:
                    json.dump(container, f, sort_keys=True, indent=2)
                    f.write("\n")
                ok.append(name)
                print(f"OK   {name}", flush=True)
            except Exception as e:  # noqa: BLE001
                fail.append(name)
                print(f"FAIL {name}: {e}", flush=True)
    print(f"SUMMARY ok={len(ok)} fail={len(fail)}", flush=True)
    if fail:
        print("FAILED: " + ", ".join(fail), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
