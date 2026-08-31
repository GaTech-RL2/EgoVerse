"""Did the checkpoint's weights actually load, or did strict=False swallow it?

trainHydra's eval path does:

    model.load_state_dict(checkpoint["state_dict"], strict=False)
    log.info(f"Loaded weights from {ckpt_path}")

The log line prints unconditionally and the return value — which names every
missing and unexpected key — is discarded. So "Loaded weights from ..." in a log
is NOT evidence that any weight was loaded. With strict=False a complete key
mismatch loads nothing at all and evaluates a randomly initialised network,
which looks exactly like a model that learned nothing.

This rebuilds the model from the same config and reports the real overlap.

    python scripts/control_modes/check_ckpt_keys.py <ckpt> <model_cfg_name>
"""

from __future__ import annotations

import sys
import pathlib

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

REPO = pathlib.Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "egomimic/hydra_configs"


def main() -> int:
    ckpt_path = sys.argv[1]
    model_cfg = sys.argv[2] if len(sys.argv) > 2 else \
        "bf/bf_ctrlmode_arm2_causal_bidir_small"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    print(f"checkpoint tensors: {len(sd)}")
    print(f"epoch: {ckpt.get('epoch')}  global_step: {ckpt.get('global_step')}")
    print("first 5 checkpoint keys:")
    for k in list(sd)[:5]:
        print(f"   {k}   {tuple(sd[k].shape)}")

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="train_zarr_cartesian.yaml", overrides=[
            "data=pusht/control_modes_gripper_arc_D10_M16_append_r0",
            f"model={model_cfg}",
            "~evaluator", "~logger", "~callbacks",
            "launch_params.nodes=1", "launch_params.gpus_per_node=1",
        ])
    model = instantiate(cfg.model, norm_stats=None, _recursive_=False)
    msd = model.state_dict()
    print(f"\nmodel tensors: {len(msd)}")
    print("first 5 model keys:")
    for k in list(msd)[:5]:
        print(f"   {k}   {tuple(msd[k].shape)}")

    ck, mk = set(sd), set(msd)
    both = ck & mk
    shape_ok = sum(1 for k in both if sd[k].shape == msd[k].shape)
    print(f"\nkeys in both      : {len(both)}")
    print(f"  of those, shapes match: {shape_ok}")
    print(f"missing from ckpt : {len(mk - ck)}")
    print(f"unexpected in ckpt: {len(ck - mk)}")
    for k in list(mk - ck)[:5]:
        print(f"   MISSING   {k}")
    for k in list(ck - mk)[:5]:
        print(f"   UNEXPECTED {k}")

    if not both:
        print("\nVERDICT: NOTHING loaded. strict=False silently evaluated a "
              "randomly initialised model.")
        return 1
    if shape_ok < 0.9 * len(mk):
        print(f"\nVERDICT: only {shape_ok}/{len(mk)} model tensors were "
              f"restored — the evaluation was largely random.")
        return 1
    print(f"\nVERDICT: {shape_ok}/{len(mk)} model tensors restored. The weights "
          f"loaded; a zero success rate is a REAL result, not a loading bug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
