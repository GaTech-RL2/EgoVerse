"""Dump fully-RESOLVED cfg.model for each BC-family leaf to sorted-keys JSON.

Mirrors scratch/dump_baseline_configs.py exactly:
  config_dir = egomimic/hydra_configs (abs)
  config_name = train_zarr_cartesian
  override = model=<name>
  OmegaConf.to_container(cfg.model, resolve=True) -> json.dump(sort_keys=True, indent=2)

OUT_DIR is taken from argv[1] so the same script dumps pre-edit and post-edit sets.
"""
import json
import os
import sys
import traceback

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
CONFIG_DIR = os.path.join(REPO, "egomimic/hydra_configs")
CONFIG_NAME = "train_zarr_cartesian"
OUT_DIR = sys.argv[1]

MODELS = [
    "bc_rnn_pushshapes",
    "bc_rnn_pushshapes_minmax",
    "bc_rnn_pushshapes_minmax_crop",
    "bc_rnn_pushshapes_cos",
    "bc_rnn_pushshapes_minmax_crop_cos",
    "bc_rnn_pushshapes_paperexact",
    "bc_rnn_pushshapes_paperexact_tx",
    "bc_rnn_pushshapes_paperexact_tx_cos",
    "bc_rnn_pushshapes_paperexact_tx_cos_lowlr",
    "bc_rnn_pushshapes_paperexact_tx_chunk8",
    "bc_rnn_pushshapes_paperexact_tx_chunk16",
    "bc_rnn_pushshapes_paperexact_tx_chunk8_q",
    "bc_rnn_pushshapes_paperexact_tx_chunk4_q",
    "bc_rnn_pushshapes_paperexact_hnet",
    "bc_rnn_pushshapes_paperexact_hnet_chunk8",
    "bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist",
    "bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist_ratio",
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ok, fail = [], []
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        for name in MODELS:
            try:
                cfg = compose(
                    config_name=CONFIG_NAME,
                    overrides=[f"model={name}"],
                    return_hydra_config=False,
                )
                container = OmegaConf.to_container(cfg.model, resolve=True)
                out_path = os.path.join(OUT_DIR, f"{name}.json")
                with open(out_path, "w") as f:
                    json.dump(container, f, sort_keys=True, indent=2)
                    f.write("\n")
                ok.append(name)
                print(f"OK   {name}", flush=True)
            except Exception as e:
                fail.append(name)
                print(f"FAIL {name}: {e}", flush=True)
                traceback.print_exc()
    print(f"\nSUMMARY ok={len(ok)} fail={len(fail)}", flush=True)
    if fail:
        print("FAILED: " + ", ".join(fail), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
