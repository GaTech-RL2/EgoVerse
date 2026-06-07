"""Smoke: verify HNetOuterStage produces the same forward output as the
current HNetPolicy for identical inputs + the same weights (deepcopy).

Same hnet stage config, same input_modules, same cond_encoder — only the
class wrapping them changes. The action_out and stages and input_modules
all share the SAME instances between the two paths, so any output diff
is a logic bug in the new class."""
import pytest  # noqa: E402

pytest.skip(
    "manual regression smoke: hardcoded EgoVerse-clone-3 paths + configs "
    "removed from this repo + needs GPU/checkpoints; run directly with "
    "python, skipped under pytest collection",
    allow_module_level=True,
)

import sys, torch, copy
sys.path.insert(0, "/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3")

from omegaconf import OmegaConf
from hydra.utils import instantiate

from egomimic.algo.packed_base import HNetPolicy
from egomimic.algo.packed_outer_stage import HNetOuterStage
from egomimic.models.stems.input_modules import ActionInToken
from egomimic.models.hnet.context import HNetContext

torch.manual_seed(0)

cfg = OmegaConf.load(
    "/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3/"
    "egomimic/hydra_configs/model/hnet_pushshapes.yaml"
)
rcfg = cfg.robomimic_model
action_dim = int(rcfg.action_dim)
action_horizon = int(rcfg.action_horizon)
d_model = int(rcfg.d_model)

cond_encoder = instantiate(rcfg.cond_encoder)
hnet_core = instantiate(rcfg.hnet)

# Build HNetPolicy (old) with these submodules.
policy = HNetPolicy(
    action_dim=action_dim,
    action_horizon=action_horizon,
    d_model=d_model,
    cond_encoder=cond_encoder,
    hnet=hnet_core,
    action_head_type="linear",
    input_modules=[ActionInToken(action_dim, d_model)],
)

# Build HNetOuterStage with the SAME submodule instances.
outer = HNetOuterStage(
    action_dim=action_dim,
    action_horizon=action_horizon,
    d_model=d_model,
    cond_encoder=cond_encoder,
    hnet=hnet_core,
    action_head_type="linear",
    input_modules=[policy.input_modules[0]],
)
# Tie the action heads so both classes have identical output layers.
outer.action_out = policy.action_out

policy.eval(); outer.eval()

# ---- padded forward smoke ----
B, T = 2, 16
actions = torch.randn(B, T, action_dim)
obs = {
    "state_agent_obj": torch.randn(B, T, 5),
    "front_img_1": torch.rand(B, T, 3, 96, 96),
}

# Path A: HNetPolicy
torch.manual_seed(42)
pred_a, aux_a = policy(actions, obs)

# Path B: HNetOuterStage
torch.manual_seed(42)
batch = {"actions": actions, "__obs": obs}
ctx = HNetContext(cond_dict={}, aux=[], inference_params=None)
outer(batch, ctx)
pred_b = batch["pred_action"]
aux_b = ctx.aux

print(f"Path A pred[0,0]: {pred_a[0, 0].tolist()}")
print(f"Path B pred[0,0]: {pred_b[0, 0].tolist()}")
print(f"Max abs diff: {(pred_a - pred_b).abs().max().item():.2e}")
assert torch.allclose(pred_a, pred_b, atol=1e-6), \
    f"Padded forward mismatch: {(pred_a - pred_b).abs().max()}"
print("PASS — padded forward matches")

# ---- step smoke ----
state = outer.init_step_state(batch_size=2, T_max=16, device=torch.device("cpu"))
obs_step = {
    "state_agent_obj": torch.randn(2, 5),
    "front_img_1": torch.rand(2, 3, 96, 96),
}
a_step = outer.step(state, obs_step, t=0)
print(f"\nStep output shape: {a_step.shape}  finite: {torch.isfinite(a_step).all().item()}")
assert a_step.shape == (2, 1, action_dim)
print("PASS — step inference works")

print("\nALL SMOKES PASS — HNetOuterStage is functionally equivalent to HNetPolicy")
