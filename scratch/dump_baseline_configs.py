"""Dump fully-RESOLVED cfg.model for each baseline model config to sorted-keys JSON.

Compose pattern mirrors egomimic.trainHydra:
  config_path = egomimic/hydra_configs  (relative to package)
  config_name = train_zarr_cartesian
  override    = model=<name>
Then OmegaConf.to_container(cfg.model, resolve=True) -> json.dump(sort_keys=True).

Only cfg.model is resolved/dumped; the BC model configs contain no cross-group
interpolations (the only interpolation is the relative ${..rnn_horizon}, internal
to the model subtree), so composing the default data/trainer/etc groups is safe and
does not affect resolved model values.
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
OUT_DIR = os.path.join(REPO, "scratch/config_refactor_baseline/resolved")

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
    "hnet_pushshapes_big",
    "hnet_pushshapes_recipe",
    "hnet_pushshapes_crossattn",
    "hnet_pushshapes_fused_lowlr",
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ok, fail = [], []
    # initialize_config_dir needs an absolute path; version_base=None for legacy behavior.
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
                print(f"OK   {name} -> {out_path}", flush=True)
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
