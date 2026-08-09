"""End-to-end smoke: instantiate refactored DFoT via hydra config, run
forward_training, check loss is finite and a scalar tensor."""
import pytest  # noqa: E402

pytest.skip(
    "manual regression smoke: hardcoded EgoVerse-clone-3 paths + configs "
    "removed from this repo + needs GPU/checkpoints; run directly with "
    "python, skipped under pytest collection",
    allow_module_level=True,
)

import sys, torch
sys.path.insert(0, "/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3")
from omegaconf import OmegaConf
from hydra.utils import instantiate


class MockNormStats:
    """Stub that gives DFoT the bare minimum it needs."""
    def __init__(self):
        self._action_keys = {0: ["actions"]}
        self._proprio_keys = {0: ["state_agent_obj"]}
        self._lang_keys = {0: []}
        self._camera_keys = {0: ["front_img_1"]}

    def keys_of_type(self, kind, emb_id):
        return {
            "action_keys": self._action_keys,
            "proprio_keys": self._proprio_keys,
            "lang_keys": self._lang_keys,
            "camera_keys": self._camera_keys,
        }[kind].get(emb_id, [])

    def is_key_with_embodiment(self, key, emb_id):
        return True

    def zarr_key_to_keyname(self, key, emb_id):
        return key

    def normalize(self, batch, emb_id):
        return batch

    def unnormalize(self, batch, emb_id):
        return batch


# Patch the embodiment lookup to return our pushshapes_sim id.
import egomimic.rldb.embodiment.embodiment as _emb
orig_get_id = _emb.get_embodiment_id
_emb.get_embodiment_id = lambda n: 0 if n == "pushshapes_sim" else orig_get_id(n)


cfg = OmegaConf.load(
    "/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3/"
    "egomimic/hydra_configs/model/dfot/base.yaml"
)
model_wrapper_cfg = cfg.robomimic_model

# Instantiate DFoT via hydra (which builds outer_stage + auto-builds the loss).
norm_stats = MockNormStats()
dfot = instantiate(model_wrapper_cfg, norm_stats=norm_stats, _recursive_=True)
dfot.nets = dfot.nets.cpu()  # avoid CUDA assumption for the test
dfot.device = torch.device("cpu")

print(f"[init] DFoT built. nets keys: {list(dfot.nets.keys())}")
print(f"[init] outer_stage class: {type(dfot.outer_stage).__name__}")
print(f"[init] loss class: {type(dfot.loss).__name__}")
print(f"[init] backbone (via property) class: {type(dfot.backbone).__name__}")
print(f"[init] cond_encoder (via property) class: {type(dfot.cond_encoder).__name__}")
print(f"[init] diffusion (via property) class: {type(dfot.diffusion).__name__}")

# Build a synthetic processed batch (padded mode).
B, T, A = 2, 8, 2
batch = {
    0: {  # emb_id
        "actions": torch.randn(B, T, A),
        "state_agent_obj": torch.randn(B, T, 5),
        "front_img_1": torch.rand(B, T, 3, 96, 96),
        "_packed": False,
    },
}

torch.manual_seed(42)
predictions = dfot.forward_training(batch)
loss_dict = dfot.compute_losses(predictions, batch)

print(f"\n[forward_training] predictions keys: {list(predictions.keys())}")
print(f"[forward_training] 0_action_loss: {predictions['0_action_loss'].item():.6f}")
print(f"[compute_losses] action_loss: {loss_dict['action_loss'].item():.6f}")
assert torch.isfinite(predictions["0_action_loss"]).all()
assert predictions["0_action_loss"].ndim == 0, "loss should be scalar"
print("\nPASS — refactored DFoT instantiates from yaml + runs forward_training")
