"""End-to-end: real staged episodes -> data config -> every arm -> loss.

The config preflight drives the head from a synthetic `condition`, so it never
touches the observation encoder or the real tensors. This closes that gap: it
runs the actual DPStyleObsEncoder over real images from the staged directory,
so a shape or key mismatch between the loader and the encoder surfaces here
rather than on the node.
"""
import os

import hydra
import torch
from omegaconf import OmegaConf

os.environ.setdefault(
    "PUSHSHAPES_ROOT", os.path.expanduser("~/Desktop/GEAR/sim_run/local_stage")
)

REPO = os.path.dirname(os.path.abspath(__file__))
DATA_CFG = os.path.join(
    REPO, "egomimic/hydra_configs/data/pusht",
    "control_modes_gripper_arc_D10_M16_append_r0.yaml")
MODEL_DIR = os.path.join(REPO, "egomimic/hydra_configs/model/bf")
DOMAIN = "pushshapes_sim_gripper"

data = OmegaConf.load(DATA_CFG)
ds = hydra.utils.instantiate(data.train_datasets[DOMAIN], _convert_="all")
print(f"dataset samples: {len(ds)}")

BATCH = 4
samples = [ds[i] for i in range(0, 4 * BATCH, 4)]


def collate(key):
    return torch.stack([torch.as_tensor(s[key]) for s in samples])


base = {
    "obs/state_agent_obj": collate("state_agent_obj"),
    "obs/front_img_1": collate("front_img_1"),
    "actions": collate("actions"),
    "embodiment": DOMAIN,
}
print("actions        ", tuple(base["actions"].shape))
print("state_agent_obj", tuple(base["obs/state_agent_obj"].shape))
print("front_img_1    ", tuple(base["obs/front_img_1"].shape))

# Only the small capacity: this is a wiring check, and instantiating four
# 300M models on a loaded laptop buys nothing the preflight has not covered.
for name in sorted(os.listdir(MODEL_DIR)):
    if not (name.startswith("bf_ctrlmode_") and name.endswith("_small.yaml")):
        continue
    cfg = OmegaConf.load(os.path.join(MODEL_DIR, name))
    stages = [hydra.utils.instantiate(s) for s in cfg.robomimic_model.stages]

    batch = dict(base)
    for st in stages:
        st.eval()
        with torch.no_grad():
            batch = st(batch)
    train_loss = batch["loss/native_action"].item()

    # Rollout obs carry an explicit singleton history axis — PipelineAlgo's
    # _append_sim_observation adds it, and FusedObsEncoder requires it under
    # the rollout marker. Reproduce that here rather than calling the encoder
    # in a shape it never sees in production.
    rollout = {
        k: (v.unsqueeze(1) if k.startswith("obs/") else v)
        for k, v in base.items()
        if k != "actions"
    }
    rollout["rollout_t"] = 0
    for st in stages:
        if getattr(st, "train_only", False):
            continue
        with torch.no_grad():
            rollout = st(rollout)

    arm = name.replace("bf_ctrlmode_", "").replace("_small.yaml", "")
    print(f"{arm:22s} pred={tuple(batch['pred_action'].shape)} "
          f"loss={train_loss:.4f} "
          f"rollout={tuple(rollout['pred_action'].shape)}")
