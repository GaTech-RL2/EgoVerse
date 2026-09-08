"""Unit tests for the BPP adapter (egomimic/algo/bpp.py).

Instantiates the real hydra model config (model/bpp_prompt_dit.yaml) with a
small ViT backbone override and runs the full Algo contract on a synthetic
batch on CPU. Requires network/HF-cache access for the timm backbone weights.
"""

import os

import hydra
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "hydra_configs", "model")
MODEL_YAML = os.path.join(_MODEL_DIR, "bpp_prompt_dit.yaml")
NOPROMPT_MODEL_YAML = os.path.join(_MODEL_DIR, "bpp_dit.yaml")

B, T, D = 2, 100, 12
EMBODIMENT = "human_bimanual"


def _build_norm_stats():
    emb_id = get_embodiment_id(EMBODIMENT)
    keys = {
        "observations.images.front_img_1": "camera_keys",
        "observations.state.ee_pose": "proprio_keys",
        "actions_cartesian": "action_keys",
    }
    stats = {
        key: {
            "quantile_1": -np.ones(D, dtype=np.float32),
            "quantile_99": np.ones(D, dtype=np.float32),
        }
        for key in ("observations.state.ee_pose", "actions_cartesian")
    }
    state = {
        "norm_mode": "quantile",
        "embodiments": [emb_id],
        "key_types": {emb_id: dict(keys)},
        "zarr_keys": {emb_id: {k: k for k in keys}},
        "shapes": {emb_id: {}},
        "norm_stats": {emb_id: stats},
    }
    return MultiDataset.from_state(state), emb_id


def _build_batch():
    torch.manual_seed(0)
    return {
        EMBODIMENT: {
            "observations.images.front_img_1": torch.rand(B, 3, 360, 640),
            "observations.state.ee_pose": torch.randn(B, D).clamp(-1, 1),
            "actions_cartesian": torch.randn(B, T, D).clamp(-1, 1),
        }
    }


def _load_small_cfg(yaml_path):
    """Real model yaml with a small ViT backbone + fewer DDIM steps for CPU
    test speed; everything else (prompt path, decoder, adapter wiring) is the
    real config."""
    cfg = OmegaConf.create({"model": OmegaConf.load(yaml_path)})
    enc = cfg.model.robomimic_model.policy.obs_encoder
    # prompted: PairPromptObsEncoder -> Tokenizer -> TransformerObsEncoder;
    # unprompted: TransformerObsEncoder directly.
    while "model_name" not in enc:
        enc = enc.obs_encoder
    enc.model_name = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
    cfg.model.robomimic_model.policy.num_inference_steps = 2
    return cfg


@pytest.fixture(scope="module")
def algo():
    cfg = _load_small_cfg(MODEL_YAML)
    norm_stats, _ = _build_norm_stats()
    return hydra.utils.instantiate(cfg.model.robomimic_model, norm_stats=norm_stats)


@pytest.fixture(scope="module")
def algo_noprompt():
    cfg = _load_small_cfg(NOPROMPT_MODEL_YAML)
    norm_stats, _ = _build_norm_stats()
    return hydra.utils.instantiate(cfg.model.robomimic_model, norm_stats=norm_stats)


@pytest.fixture()
def emb_id():
    return get_embodiment_id(EMBODIMENT)


def test_training_path_and_grads(algo, emb_id):
    algo.nets.train()
    processed = algo.process_batch_for_training(_build_batch())
    assert emb_id in processed
    assert "pad_mask" in processed[emb_id]

    predictions = algo.forward_training(processed)
    losses = algo.compute_losses(predictions, processed)
    assert torch.isfinite(losses["action_loss"])

    algo.nets.zero_grad(set_to_none=True)
    losses["action_loss"].backward()
    trainable = [p for p in algo.nets.parameters() if p.requires_grad]
    with_grad = [p for p in trainable if p.grad is not None]
    # Every trainable tensor must receive a grad or DDP
    # (find_unused_parameters=False) breaks at train time.
    assert len(with_grad) == len(trainable)
    assert sum(p.grad.abs().sum().item() for p in with_grad) > 0

    log = algo.log_info({"losses": losses})
    assert "Loss" in log and "action_loss" in log


def test_policy_normalizer_is_identity_and_frozen(algo):
    x = torch.randn(4, T, D, device=algo.device)
    normalizer = algo.nets["policy"].normalizer
    assert torch.allclose(x, normalizer["action"].normalize(x))
    prompt_normalizer = normalizer.get_prompt_normalizer()
    assert torch.allclose(x, prompt_normalizer["action"].normalize(x))
    assert all(not p.requires_grad for p in normalizer.parameters())


def test_forward_eval_shapes(algo, emb_id):
    algo.nets.eval()
    with torch.no_grad():
        processed = algo.process_batch_for_training(_build_batch())
        preds = algo.forward_eval(processed)
    assert torch.isfinite(preds[f"{EMBODIMENT}_loss"])
    act = preds[f"{EMBODIMENT}_actions_cartesian"]
    assert act.shape == (B, T, D)
    assert torch.isfinite(act).all()


def test_prompting_deployment_cycle(algo, emb_id):
    algo.nets.eval()
    assert algo.supports_prompting()
    with torch.no_grad():
        processed = algo.process_batch_for_training(_build_batch())
        obs_dict = algo._build_obs_dict(processed[emb_id], emb_id, training=False)
    prompt_dict = obs_dict.pop("prompt")
    algo.prompt(prompt_dict)
    with torch.no_grad():
        result = algo.nets["policy"].predict_action(obs_dict)
    assert result["action"].shape == (B, T, D)
    algo.reset()
    assert not algo.nets["policy"].obs_encoder.is_prompted


def test_batch_roll_prompt_structure(algo, emb_id):
    algo.nets.train()
    processed = algo.process_batch_for_training(_build_batch())
    obs_dict = algo._build_obs_dict(processed[emb_id], emb_id, training=True)
    prompt = obs_dict["prompt"]
    P = T // algo.chunk_n_actions
    assert prompt["action"].shape == (B, P, algo.chunk_n_actions, D)
    assert prompt["metadata"]["mask"].shape == (B, P)
    assert not prompt["metadata"]["mask"].any()  # batch-roll prompts are unpadded
    # prompt obs carries only prompt-participating keys, repeated across P steps
    assert set(prompt["obs"].keys()) == {"front_img_1", "state_ee_pose"}
    assert prompt["obs"]["front_img_1"].shape[:2] == (B, P)
    assert prompt["obs"]["state_ee_pose"].shape == (B, P, D)
    # roll-by-one pairing: sample 0's prompt actions == sample B-1's actions
    actions = processed[emb_id][algo.ac_keys[emb_id]]
    assert torch.allclose(prompt["action"][0].reshape(T, D), actions[B - 1])


# ---------------------------------------------------------------------------
# Unprompted variant (model/bpp_dit.yaml): plain TransformerObsEncoder, no
# prompt is built, encoded, or attended to.
# ---------------------------------------------------------------------------


def test_noprompt_training_path_and_grads(algo_noprompt, emb_id):
    algo = algo_noprompt
    assert not algo.use_prompt
    assert algo.prompt_chunker is None
    algo.nets.train()
    processed = algo.process_batch_for_training(_build_batch())

    predictions = algo.forward_training(processed)
    losses = algo.compute_losses(predictions, processed)
    assert torch.isfinite(losses["action_loss"])

    algo.nets.zero_grad(set_to_none=True)
    losses["action_loss"].backward()
    trainable = [p for p in algo.nets.parameters() if p.requires_grad]
    with_grad = [p for p in trainable if p.grad is not None]
    assert len(with_grad) == len(trainable)
    assert sum(p.grad.abs().sum().item() for p in with_grad) > 0


def test_noprompt_obs_dict_has_no_prompt(algo_noprompt, emb_id):
    algo = algo_noprompt
    algo.nets.train()
    processed = algo.process_batch_for_training(_build_batch())
    obs_dict = algo._build_obs_dict(processed[emb_id], emb_id, training=True)
    assert "prompt" not in obs_dict
    assert set(obs_dict) == {"front_img_1", "state_ee_pose"}
    assert obs_dict["front_img_1"].shape == (B, 1, 3, 224, 224)
    assert obs_dict["state_ee_pose"].shape == (B, 1, D)
    # no prompt encoder anywhere in the policy
    assert not any("Prompt" in type(m).__name__ for m in algo.nets["policy"].modules())


def test_noprompt_forward_eval_shapes(algo_noprompt, emb_id):
    algo = algo_noprompt
    algo.nets.eval()
    with torch.no_grad():
        processed = algo.process_batch_for_training(_build_batch())
        preds = algo.forward_eval(processed)
    assert torch.isfinite(preds[f"{EMBODIMENT}_loss"])
    act = preds[f"{EMBODIMENT}_actions_cartesian"]
    assert act.shape == (B, T, D)
    assert torch.isfinite(act).all()


def test_noprompt_prompting_refused(algo_noprompt):
    algo = algo_noprompt
    assert not algo.supports_prompting()
    with pytest.raises(RuntimeError, match="without a prompt encoder"):
        algo.prompt({})
    algo.reset()  # reset is a no-op on the plain encoder and must not raise


def test_noprompt_rejects_prompt_block():
    cfg = _load_small_cfg(NOPROMPT_MODEL_YAML)
    cfg.model.robomimic_model.prompt = {"mode": "batch_roll"}
    norm_stats, _ = _build_norm_stats()
    with pytest.raises(Exception, match="does not support prompting"):
        hydra.utils.instantiate(cfg.model.robomimic_model, norm_stats=norm_stats)
