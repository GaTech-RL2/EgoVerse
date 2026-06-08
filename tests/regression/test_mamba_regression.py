"""Is the mamba_encdec packed-forward error a refactor regression?
Run BOTH HNetPolicy (old) and HNetOuterStage (new) with the same mamba
config + same input; compare error shape. If both fail with the same
error -> pre-existing mamba_ssm runtime issue, NOT a refactor bug."""
import pytest  # noqa: E402

pytest.skip(
    "manual regression smoke: hardcoded EgoVerse-clone-3 paths + configs "
    "removed from this repo + needs GPU/checkpoints; run directly with "
    "python, skipped under pytest collection",
    allow_module_level=True,
)

import sys, traceback, torch
sys.path.insert(0, "/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3")
from omegaconf import OmegaConf
from hydra.utils import instantiate

from egomimic.algo.hnet import HNetPolicy
from egomimic.algo.hnet import HNetOuterStage
from egomimic.models.stems.input_modules import ActionInToken
from egomimic.models.hnet.context import HNetContext


cfg = OmegaConf.load(
    "/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3/"
    "egomimic/hydra_configs/model/hnet_pushshapes_mamba_encdec.yaml"
)
rcfg = cfg.robomimic_model.outer_stage  # mamba lives under outer_stage now

action_dim = int(rcfg.action_dim)
action_horizon = int(rcfg.action_horizon)
d_model = int(rcfg.d_model)
cond_encoder = instantiate(rcfg.cond_encoder)
hnet_core = instantiate(rcfg.hnet)

device = torch.device("cuda")
policy = HNetPolicy(
    action_dim=action_dim, action_horizon=action_horizon, d_model=d_model,
    cond_encoder=cond_encoder, hnet=hnet_core, action_head_type="linear",
    input_modules=[ActionInToken(action_dim, d_model)],
).to(device)
outer = HNetOuterStage(
    action_dim=action_dim, action_horizon=action_horizon, d_model=d_model,
    cond_encoder=cond_encoder, hnet=hnet_core, action_head_type="linear",
    input_modules=[policy.input_modules[0]],
).to(device)
outer.action_out = policy.action_out

# Packed batch.
T1, T2 = 12, 20
T_total = T1 + T2
cu = torch.tensor([0, T1, T_total], dtype=torch.long, device=device)
actions = torch.randn(T_total, action_dim, device=device)
obs = {
    "state_agent_obj": torch.randn(T_total, 5, device=device),
    "front_img_1": torch.rand(T_total, 3, 96, 96, device=device),
}

print("=== HNetPolicy (old class) ===")
try:
    pred, _ = policy.forward_packed(actions, obs, cu, T2)
    print(f"  OK — shape {pred.shape}")
except Exception:
    traceback.print_exc()

print("\n=== HNetOuterStage (new class) ===")
try:
    batch = {"actions": actions, "__obs": obs}
    ctx = HNetContext(cond_dict={}, aux=[], inference_params=None, cu_seqlens=cu, max_seqlen=T2)
    outer(batch, ctx)
    print(f"  OK — shape {batch['pred_action'].shape}")
except Exception:
    traceback.print_exc()
