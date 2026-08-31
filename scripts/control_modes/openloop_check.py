"""Did the model learn the token distribution, or not learn at all?

A zero success rate has two very different causes and they need different
fixes:

  * the model never learned to predict the arc token   -> a TRAINING problem
  * it predicts tokens well but fails when its own output is executed
    step after step                                    -> a CLOSED-LOOP problem

Rollout success cannot distinguish them. Open-loop prediction error can: run the
trained weights on real held-out samples, teacher-forced, and compare against
the ground-truth token. This is the same quantity the training loss optimizes,
so a low number here with 0% rollout success localizes the fault to deployment,
and a high number localizes it to training.

    python scripts/control_modes/openloop_check.py <ckpt> <norm_stats.json> [nbatch]
"""

from __future__ import annotations

import pathlib
import sys

import hydra
import torch
import torch.nn as nn
from omegaconf import OmegaConf

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from egomimic.pipeline.core import Pipeline  # noqa: E402
from egomimic.rldb.embodiment.pushshapes import (  # noqa: E402
    get_keymap_hpt, get_planar_arc_length_transform_list,
)
from egomimic.rldb.zarr.zarr_dataset_multi import (  # noqa: E402
    LocalEpisodeResolverWithEmbodimentOverride, MultiDataset,
)

DOMAIN = "pushshapes_sim_gripper"
MODEL_CFG = REPO / "egomimic/hydra_configs/model/bf/bf_ctrlmode_arm2_causal_bidir_small.yaml"
STAGE = pathlib.Path("~/Desktop/GEAR/sim_run/local_stage/train/gripper/T").expanduser()


def main() -> int:
    ckpt_path = sys.argv[1]
    norm_path = sys.argv[2]
    n_batch = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    cfg = OmegaConf.load(MODEL_CFG)
    stages = [hydra.utils.instantiate(s) for s in cfg.robomimic_model.stages]
    nets = nn.ModuleDict({"policy": Pipeline(stages)})

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    stripped = {k[len("nets."):]: v for k, v in sd.items() if k.startswith("nets.")}
    missing, unexpected = nets.load_state_dict(stripped, strict=False)
    print(f"loaded: missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        print("  REFUSING to report numbers from a partially loaded model")
        return 1
    nets.eval()

    res = LocalEpisodeResolverWithEmbodimentOverride(
        folder_path=str(STAGE), embodiment_override=DOMAIN,
        key_map=get_keymap_hpt(action_horizon=200),
        transform_list=get_planar_arc_length_transform_list(
            min_distance_unit=10.0, resampled_vector_length=16, dt=1 / 30.,
            rotation_radius=0.0, velocity_mode="mean_scalar",
            velocity_layout="append"),
    )
    ds = MultiDataset._from_resolver(resolver=res, mode="train",
                                     valid_ratio=0.02, bounds_check=False)
    # NORMALIZE with the stats the checkpoint was TRAINED under. Without this
    # the targets come back in raw pixel units (x, y std ~134) while the model
    # emits normalized values, and the resulting MSE is a units mismatch on two
    # channels rather than a statement about learning. The tell is per-channel:
    # x and y in the tens of thousands while cos/sin/grip sit near 1.
    ds.infer_norm_from_dataset(dataset=None, dataset_name=DOMAIN,
                               precomputed_norm_path=norm_path)
    print(f"dataset samples: {len(ds)}  (normalized with {norm_path})")

    step = max(1, len(ds) // (n_batch * 16))
    idx = list(range(0, len(ds), step))[: n_batch * 16]
    samples = [ds[i] for i in idx]

    def stack(key):
        return torch.stack([torch.as_tensor(s[key]) for s in samples])

    target = stack("actions")
    batch = {
        "obs/state_agent_obj": stack("state_agent_obj"),
        "obs/front_img_1": stack("front_img_1"),
        "actions": target.clone(),
        "embodiment": DOMAIN,
    }
    with torch.no_grad():
        out = nets["policy"](batch)
    pred = out["pred_action"]

    mse = torch.nn.functional.mse_loss(pred, target).item()
    # A model that ignores its input and emits a constant would score the
    # target's own variance; predicting the per-batch MEAN token is the honest
    # trivial baseline to beat.
    trivial = torch.nn.functional.mse_loss(
        target.mean(dim=0, keepdim=True).expand_as(target), target).item()
    print(f"\nsamples            : {len(samples)}")
    print(f"target scale (std) : {target.std().item():.4f}")
    print(f"model MSE          : {mse:.6f}")
    print(f"predict-the-mean   : {trivial:.6f}")
    print(f"ratio (model/mean) : {mse / max(trivial, 1e-9):.3f}")

    print("\nper-channel MSE  [x, y, cos, sin, grip]:")
    per = ((pred - target) ** 2).mean(dim=(0, 1))
    print("   " + "  ".join(f"{v:.4f}" for v in per.tolist()))

    print()
    if mse < 0.25 * trivial:
        print("VERDICT: the model predicts tokens well ABOVE the trivial baseline.")
        print("         Training worked; 0% rollout success is a CLOSED-LOOP or")
        print("         action-decoding problem, not a learning one.")
    elif mse < trivial:
        print("VERDICT: barely better than predicting the mean — weak but not null.")
    else:
        print("VERDICT: NO better than predicting the mean. Training did not")
        print("         learn a useful mapping; the 0% is downstream of that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
