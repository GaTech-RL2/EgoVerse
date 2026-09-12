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

B, T, D = 2, 100, 18
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


def _load_yaml_with_defaults(yaml_path):
    """``OmegaConf.load`` plus a minimal hydra ``defaults:`` merge (bare names
    resolved in the yaml's own directory, merged in order, the file's own
    keys last), since these tests bypass hydra compose."""
    cfg = OmegaConf.load(yaml_path)
    defaults = cfg.pop("defaults", None)
    if not defaults:
        return cfg
    base = OmegaConf.create({})
    for entry in defaults:
        if entry == "_self_":
            continue
        if not isinstance(entry, str):
            raise ValueError(f"unsupported defaults entry in {yaml_path}: {entry}")
        base = OmegaConf.merge(
            base,
            _load_yaml_with_defaults(
                os.path.join(os.path.dirname(yaml_path), entry + ".yaml")
            ),
        )
    return OmegaConf.merge(base, cfg)


def _load_small_cfg(yaml_path):
    """Real model yaml with a small ViT backbone + fewer DDIM steps for CPU
    test speed; everything else (prompt path, decoder, adapter wiring) is the
    real config."""
    cfg = OmegaConf.create({"model": _load_yaml_with_defaults(yaml_path)})
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


# ---------------------------------------------------------------------------
# Whole-episode prompting (model/bpp_prompt_dit_episode.yaml, prompt.mode
# episode_pair): the batch carries a collated, variable-length prompt.
# ---------------------------------------------------------------------------

EPISODE_MODEL_YAML = os.path.join(_MODEL_DIR, "bpp_prompt_dit_episode.yaml")
CHUNK_EP = 30


@pytest.fixture(scope="module")
def algo_episode():
    cfg = _load_small_cfg(EPISODE_MODEL_YAML)
    norm_stats, _ = _build_norm_stats()
    return hydra.utils.instantiate(cfg.model.robomimic_model, norm_stats=norm_stats)


def _build_episode_batch(lengths=(3, 5, 4, 2), groups=(0, 0, 1, 1)):
    from egomimic.pl_utils.pl_data_utils import prompt_collate

    torch.manual_seed(0)
    Bn = len(lengths)
    prompts = [
        {
            "obs": {
                "observations.images.front_img_1": torch.rand(L, 3, 224, 224),
                "observations.state.ee_pose": torch.randn(L, D).clamp(-1, 1),
            },
            "action": torch.randn(L, CHUNK_EP, D).clamp(-1, 1),
            "length": L,
        }
        for L in lengths
    ]
    return {
        EMBODIMENT: {
            "observations.images.front_img_1": torch.rand(Bn, 3, 360, 640),
            "observations.state.ee_pose": torch.randn(Bn, D).clamp(-1, 1),
            "actions_cartesian": torch.randn(Bn, T, D).clamp(-1, 1),
            "group_idx": torch.tensor(groups),
            "prompt": prompt_collate(prompts),
        }
    }


def test_episode_mode_config(algo_episode):
    assert algo_episode.prompt_mode == "episode_pair"
    assert algo_episode.chunk_n_actions == CHUNK_EP
    assert algo_episode.max_prompt_len == 15  # 450 / 30
    assert algo_episode.prompt_chunker is None


def test_episode_prompt_structure(algo_episode, emb_id):
    algo = algo_episode
    algo.nets.eval()
    processed = algo.process_batch_for_training(_build_episode_batch())
    assert torch.is_tensor(processed[emb_id]["prompt"]["action"])
    obs_dict = algo._build_obs_dict(processed[emb_id], emb_id, training=False)
    prompt = obs_dict["prompt"]
    Bn, P = 4, 5
    assert set(prompt["obs"].keys()) == {"front_img_1", "state_ee_pose"}
    assert prompt["obs"]["front_img_1"].shape == (Bn, P, 3, 224, 224)
    assert prompt["obs"]["state_ee_pose"].shape == (Bn, P, D)
    assert prompt["action"].shape == (Bn, P, CHUNK_EP, D)
    mask = prompt["metadata"]["mask"]
    assert mask.shape == (Bn, P) and mask.dtype == torch.bool
    assert mask.sum(1).tolist() == [2, 0, 1, 3]


def test_episode_training_path_and_grads(algo_episode, emb_id):
    algo = algo_episode
    algo.nets.train()
    processed = algo.process_batch_for_training(_build_episode_batch())
    predictions = algo.forward_training(processed)
    losses = algo.compute_losses(predictions, processed)
    assert torch.isfinite(losses["action_loss"])
    algo.nets.zero_grad(set_to_none=True)
    losses["action_loss"].backward()
    trainable = [p for p in algo.nets.parameters() if p.requires_grad]
    with_grad = [p for p in trainable if p.grad is not None]
    assert len(with_grad) == len(trainable)


def test_episode_forward_eval_and_seen_unseen_losses(algo_episode, emb_id):
    algo = algo_episode
    algo.nets.eval()
    batch = _build_episode_batch()
    batch[EMBODIMENT]["operator_seen"] = torch.tensor([1, 1, 0, 0])
    with torch.no_grad():
        processed = algo.process_batch_for_training(batch)
        preds = algo.forward_eval(processed)
    assert torch.isfinite(preds[f"{EMBODIMENT}_loss"])
    assert torch.isfinite(preds[f"{EMBODIMENT}_loss_seen_operator"])
    assert torch.isfinite(preds[f"{EMBODIMENT}_loss_unseen_operator"])
    assert preds[f"{EMBODIMENT}_actions_cartesian"].shape == (4, T, D)

    # all-seen batch: no unseen key; untagged batch: neither key
    batch = _build_episode_batch()
    batch[EMBODIMENT]["operator_seen"] = torch.tensor([1, 1, 1, 1])
    with torch.no_grad():
        preds2 = algo.forward_eval(algo.process_batch_for_training(batch))
    assert f"{EMBODIMENT}_loss_seen_operator" in preds2
    assert f"{EMBODIMENT}_loss_unseen_operator" not in preds2
    with torch.no_grad():
        preds3 = algo.forward_eval(
            algo.process_batch_for_training(_build_episode_batch())
        )
    assert f"{EMBODIMENT}_loss_seen_operator" not in preds3


def test_episode_prompt_dropout_zeroes_content(algo_episode, emb_id):
    algo = algo_episode
    algo.nets.train()
    processed = algo.process_batch_for_training(_build_episode_batch())
    old = algo.p_drop_prompt
    algo.p_drop_prompt = 1.0
    try:
        obs_dict = algo._build_obs_dict(processed[emb_id], emb_id, training=True)
    finally:
        algo.p_drop_prompt = old
    assert (obs_dict["prompt"]["action"] == 0).all()
    assert (obs_dict["prompt"]["obs"]["state_ee_pose"] == 0).all()
    assert not obs_dict["prompt"]["metadata"]["mask"].all(1).any()


def test_episode_mode_requires_prompt_in_batch(algo_episode, emb_id):
    algo = algo_episode
    batch = _build_episode_batch()
    del batch[EMBODIMENT]["prompt"]
    processed = algo.process_batch_for_training(batch)
    with pytest.raises(ValueError, match="carries no `prompt`"):
        algo._build_obs_dict(processed[emb_id], emb_id, training=False)


def test_episode_rejects_wrong_chunk_size(algo_episode, emb_id):
    algo = algo_episode
    batch = _build_episode_batch()
    bad = batch[EMBODIMENT]["prompt"]["action"]
    batch[EMBODIMENT]["prompt"]["action"] = bad[:, :, : CHUNK_EP - 1]
    processed = algo.process_batch_for_training(batch)
    with pytest.raises(ValueError, match="prompt chunk size"):
        algo._build_obs_dict(processed[emb_id], emb_id, training=False)


def test_noprompt_policy_drops_episode_prompt_batch(algo_noprompt, emb_id):
    """Arm A of the video-context ablation: the unprompted policy trains on
    data built by EpisodePromptMultiDataset, so the batch carries a collated
    prompt plus group / operator tags. The adapter must drop the prompt and
    keep the tags."""
    algo = algo_noprompt
    algo.nets.train()
    batch = _build_episode_batch()
    batch[EMBODIMENT]["operator_seen"] = torch.tensor([1, 1, 0, 0])
    processed = algo.process_batch_for_training(batch)
    assert "prompt" not in processed[emb_id]
    assert processed[emb_id]["group_idx"].tolist() == [0, 0, 1, 1]
    assert processed[emb_id]["operator_seen"].tolist() == [1, 1, 0, 0]
    obs_dict = algo._build_obs_dict(processed[emb_id], emb_id, training=True)
    assert "prompt" not in obs_dict
    predictions = algo.forward_training(processed)
    losses = algo.compute_losses(predictions, processed)
    assert torch.isfinite(losses["action_loss"])
    algo.nets.zero_grad(set_to_none=True)
    losses["action_loss"].backward()
    algo.nets.eval()
    with torch.no_grad():
        preds = algo.forward_eval(algo.process_batch_for_training(batch))
    assert preds[f"{EMBODIMENT}_actions_cartesian"].shape == (4, T, D)
    # seen / unseen losses are computed from the tags, prompt or not
    assert torch.isfinite(preds[f"{EMBODIMENT}_loss_seen_operator"])
    assert torch.isfinite(preds[f"{EMBODIMENT}_loss_unseen_operator"])


def test_grad_checkpointing_flag_enables_vit_checkpointing(algo_episode, emb_id):
    # default (algo_episode fixture) stays off
    assert not any(
        getattr(m, "grad_checkpointing", False)
        for m in algo_episode.nets["policy"].modules()
    )
    cfg = _load_small_cfg(EPISODE_MODEL_YAML)
    cfg.model.robomimic_model.grad_checkpointing = True
    norm_stats, _ = _build_norm_stats()
    algo = hydra.utils.instantiate(cfg.model.robomimic_model, norm_stats=norm_stats)
    vits = [
        m for m in algo.nets["policy"].modules() if hasattr(m, "set_grad_checkpointing")
    ]
    assert vits and all(getattr(m, "grad_checkpointing", False) for m in vits)
    import timm.layers

    assert not timm.layers.use_reentrant_ckpt()
    # training path still backprops into every trainable parameter
    algo.nets.train()
    processed = algo.process_batch_for_training(_build_episode_batch())
    losses = algo.compute_losses(algo.forward_training(processed), processed)
    algo.nets.zero_grad(set_to_none=True)
    losses["action_loss"].backward()
    trainable = [p for p in algo.nets.parameters() if p.requires_grad]
    assert all(p.grad is not None for p in trainable)


@pytest.mark.parametrize("fixture_name", ["algo_episode", "algo_history"])
def test_optimizer_param_groups_cover_all_and_scale_backbone(fixture_name, request):
    algo = request.getfixturevalue(fixture_name)
    lr, wd = 5e-5, 1e-6
    groups = algo.optimizer_param_groups(lr=lr, weight_decay=wd)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))  # no param in two groups
    trainable = {id(p) for p in algo.nets.parameters() if p.requires_grad}
    assert set(ids) == trainable
    lrs = {float(g.get("lr", lr)) for g in groups}
    assert lr in lrs
    assert any(abs(g_lr - lr * 0.1) < 1e-12 for g_lr in lrs)  # backbone at 0.1x
    backbone = [g for g in groups if abs(float(g.get("lr", lr)) - lr * 0.1) < 1e-12]
    assert all(g["weight_decay"] == 0 for g in backbone)
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
    assert len(opt.param_groups) == len(groups)
    if fixture_name == "algo_history":
        enc = algo.nets["policy"].obs_encoder
        no_decay = {
            id(p) for g in groups if g["weight_decay"] == 0 for p in g["params"]
        }
        assert id(enc.segment_emb) in no_decay
        assert id(enc.history_pos_proj.bias) in no_decay
        assert id(enc.history_pos_proj.weight) not in no_decay


# ---------------------------------------------------------------------------
# Own-episode history (model/bpp_prompt_dit_episode_history.yaml): the batch
# also carries a collated, variable-length `history`; the adapter nests it in
# the prompt metadata and HistoryPairPromptObsEncoder appends it to the
# prompt decoder's memory. See docs/plan/2026-09-08_bpp_rollout_history.md.
# ---------------------------------------------------------------------------

HISTORY_MODEL_YAML = os.path.join(_MODEL_DIR, "bpp_prompt_dit_episode_history.yaml")
HIST_MAX = 15


def _history_small_cfg():
    cfg = _load_small_cfg(HISTORY_MODEL_YAML)
    # the folding defaults (60-step chunks, 2700-frame table, ViT grad
    # checkpointing) -> the short-task geometry the episode tests use
    cfg.model.robomimic_model.shape_meta.prompt_chunk_n_actions = CHUNK_EP
    cfg.model.robomimic_model.shape_meta.max_sequence_length = 450
    cfg.model.robomimic_model.grad_checkpointing = False
    return cfg


@pytest.fixture(scope="module")
def algo_history():
    cfg = _history_small_cfg()
    norm_stats, _ = _build_norm_stats()
    return hydra.utils.instantiate(cfg.model.robomimic_model, norm_stats=norm_stats)


def _history_payloads(hist_lengths, seed=1):
    torch.manual_seed(seed)
    return [
        {
            "obs": {
                "observations.images.front_img_1": torch.rand(L, 3, 224, 224),
                "observations.state.ee_pose": torch.randn(L, D).clamp(-1, 1),
            },
            "action": torch.randn(L, CHUNK_EP, D).clamp(-1, 1),
            "length": L,
        }
        for L in hist_lengths
    ]


def _build_history_batch(
    lengths=(3, 5, 4, 2), hist_lengths=(0, 2, 1, 3), groups=(0, 0, 1, 1)
):
    from egomimic.pl_utils.pl_data_utils import prompt_collate

    batch = _build_episode_batch(lengths, groups)
    batch[EMBODIMENT]["history"] = prompt_collate(_history_payloads(hist_lengths))
    return batch


def _one_chunk_payload(seed=7):
    """A single completed chunk with batch keys, B = 1 (what a rollout buffer
    hands to ``BPP.push_history_chunk``)."""
    torch.manual_seed(seed)
    return {
        "obs": {
            "observations.images.front_img_1": torch.rand(1, 1, 3, 224, 224),
            "observations.state.ee_pose": torch.randn(1, 1, D).clamp(-1, 1),
        },
        "action": torch.randn(1, 1, CHUNK_EP, D).clamp(-1, 1),
        "length": 1,
    }


def _assert_all_grads(algo):
    trainable = [p for p in algo.nets.parameters() if p.requires_grad]
    with_grad = [p for p in trainable if p.grad is not None]
    assert len(with_grad) == len(trainable)


def test_history_config_and_encoder(algo_history, algo_episode):
    from egomimic.algo.bpp_history_encoder import HistoryPairPromptObsEncoder

    algo = algo_history
    assert algo.use_history and algo.prompt_mode == "episode_pair"
    enc = algo.nets["policy"].obs_encoder
    assert isinstance(enc, HistoryPairPromptObsEncoder)
    assert algo.history_max_chunks == HIST_MAX == enc.history_max_chunks
    assert not enc.prompt_encoder_enabled and enc.prompt_with_obs_decoder_enabled
    # the diffusion head's conditioning size is independent of the memory
    assert (
        enc.get_max_token_count()
        == algo_episode.nets["policy"].obs_encoder.get_max_token_count()
    )
    assert algo.p_drop_history == 0.2 and not algo.history_only
    assert algo.eval_history_chunks is None


def test_history_obs_dict_structure(algo_history, emb_id):
    algo = algo_history
    algo.nets.eval()
    processed = algo.process_batch_for_training(_build_history_batch())
    assert torch.is_tensor(processed[emb_id]["history"]["action"])
    obs_dict = algo._build_obs_dict(processed[emb_id], emb_id, training=False)
    hist = obs_dict["prompt"]["metadata"]["history"]
    Bn, Hn = 4, 3
    assert set(hist["obs"].keys()) == {"front_img_1", "state_ee_pose"}
    assert hist["obs"]["front_img_1"].shape == (Bn, Hn, 3, 224, 224)
    assert hist["obs"]["state_ee_pose"].shape == (Bn, Hn, D)
    assert hist["action"].shape == (Bn, Hn, CHUNK_EP, D)
    mask = hist["metadata"]["mask"]
    assert mask.shape == (Bn, Hn) and mask.dtype == torch.bool
    assert mask.sum(1).tolist() == [3, 1, 2, 0]
    # the demo prompt is untouched
    assert obs_dict["prompt"]["metadata"]["mask"].sum(1).tolist() == [2, 0, 1, 3]

    batch = _build_history_batch(hist_lengths=(HIST_MAX + 1, 1, 1, 1))
    with pytest.raises(ValueError, match="history has 16 chunks"):
        algo._build_obs_dict(
            algo.process_batch_for_training(batch)[emb_id], emb_id, training=False
        )
    batch = _build_history_batch()
    bad = batch[EMBODIMENT]["history"]["action"]
    batch[EMBODIMENT]["history"]["action"] = bad[:, :, : CHUNK_EP - 1]
    with pytest.raises(ValueError, match="history chunk size"):
        algo._build_obs_dict(
            algo.process_batch_for_training(batch)[emb_id], emb_id, training=False
        )


def test_history_training_path_and_grads(algo_history, emb_id):
    algo = algo_history
    algo.nets.train()
    processed = algo.process_batch_for_training(_build_history_batch())
    losses = algo.compute_losses(algo.forward_training(processed), processed)
    assert torch.isfinite(losses["action_loss"])
    algo.nets.zero_grad(set_to_none=True)
    losses["action_loss"].backward()
    _assert_all_grads(algo)
    enc = algo.nets["policy"].obs_encoder
    assert enc.history_pos_proj.weight.grad.abs().sum() > 0


def test_zero_history_batch_trains_without_nan(algo_history, emb_id):
    algo = algo_history
    algo.nets.train()
    batch = _build_history_batch(hist_lengths=(0, 0, 0, 0))
    assert batch[EMBODIMENT]["history"]["action"].shape[1] == 0
    processed = algo.process_batch_for_training(batch)
    obs_dict = algo._build_obs_dict(processed[emb_id], emb_id, training=True)
    hist = obs_dict["prompt"]["metadata"]["history"]
    assert hist["obs"]["front_img_1"].shape == (4, 0, 3, 224, 224)
    assert hist["metadata"]["mask"].shape == (4, 0)
    losses = algo.compute_losses(algo.forward_training(processed), processed)
    assert torch.isfinite(losses["action_loss"])
    algo.nets.zero_grad(set_to_none=True)
    losses["action_loss"].backward()
    _assert_all_grads(algo)
    algo.nets.eval()
    with torch.no_grad():
        preds = algo.forward_eval(algo.process_batch_for_training(batch))
    assert torch.isfinite(preds[f"{EMBODIMENT}_loss"])
    assert torch.isfinite(preds[f"{EMBODIMENT}_actions_cartesian"]).all()


def test_history_dropout_masks_and_zeroes(algo_history, emb_id):
    algo = algo_history
    algo.nets.train()
    processed = algo.process_batch_for_training(_build_history_batch())
    saved = (
        algo.p_drop_history,
        algo.p_drop_history_state,
        algo.p_drop_history_action,
        algo.history_action_noise_std,
    )
    try:
        # whole-history dropout: content zeroed AND every chunk masked
        algo.p_drop_history = 1.0
        obs_dict = algo._build_obs_dict(processed[emb_id], emb_id, training=True)
        hist = obs_dict["prompt"]["metadata"]["history"]
        assert (hist["action"] == 0).all()
        assert (hist["obs"]["state_ee_pose"] == 0).all()
        assert (hist["obs"]["front_img_1"] == 0).all()
        assert hist["metadata"]["mask"].all()
        # the demo tokens keep the memory valid: finite loss
        loss = algo.nets["policy"].compute_loss(
            {"obs": obs_dict, "action": processed[emb_id]["actions_cartesian"]}
        )
        assert torch.isfinite(loss)
        # eval leaves the content alone
        obs_eval = algo._build_obs_dict(processed[emb_id], emb_id, training=False)
        hist_eval = obs_eval["prompt"]["metadata"]["history"]
        assert not (hist_eval["action"] == 0).all()
        assert hist_eval["metadata"]["mask"].sum(1).tolist() == [3, 1, 2, 0]

        # per-modality: state only
        algo.p_drop_history = 0.0
        algo.p_drop_history_state = 1.0
        hist = algo._build_obs_dict(processed[emb_id], emb_id, training=True)["prompt"][
            "metadata"
        ]["history"]
        assert (hist["obs"]["state_ee_pose"] == 0).all()
        assert not (hist["action"] == 0).all()
        assert hist["metadata"]["mask"].sum(1).tolist() == [3, 1, 2, 0]
        # per-modality: action only
        algo.p_drop_history_state = 0.0
        algo.p_drop_history_action = 1.0
        hist = algo._build_obs_dict(processed[emb_id], emb_id, training=True)["prompt"][
            "metadata"
        ]["history"]
        assert (hist["action"] == 0).all()
        assert not (hist["obs"]["state_ee_pose"] == 0).all()
        # action noise, train only
        algo.p_drop_history_action = 0.0
        algo.history_action_noise_std = 0.1
        train_hist = algo._build_obs_dict(processed[emb_id], emb_id, training=True)[
            "prompt"
        ]["metadata"]["history"]
        eval_hist = algo._build_obs_dict(processed[emb_id], emb_id, training=False)[
            "prompt"
        ]["metadata"]["history"]
        assert not torch.allclose(train_hist["action"], eval_hist["action"])
        assert torch.allclose(
            train_hist["obs"]["state_ee_pose"], eval_hist["obs"]["state_ee_pose"]
        )
    finally:
        (
            algo.p_drop_history,
            algo.p_drop_history_state,
            algo.p_drop_history_action,
            algo.history_action_noise_std,
        ) = saved


def test_eval_history_slice(algo_history, emb_id):
    algo = algo_history
    algo.nets.eval()
    processed = algo.process_batch_for_training(
        _build_history_batch(hist_lengths=(0, 2, 3, 1))
    )
    full = algo._build_obs_dict(processed[emb_id], emb_id, training=False)["prompt"][
        "metadata"
    ]["history"]
    try:
        algo.eval_history_chunks = 1
        hist = algo._build_obs_dict(processed[emb_id], emb_id, training=False)[
            "prompt"
        ]["metadata"]["history"]
        assert hist["action"].shape[1] == 1
        assert hist["metadata"]["mask"].tolist() == [[True], [False], [False], [False]]
        # the newest valid chunk of each row survives
        for b, src in ((1, 1), (2, 2), (3, 0)):
            assert torch.equal(hist["action"][b, 0], full["action"][b, src])
            assert torch.equal(
                hist["obs"]["front_img_1"][b, 0], full["obs"]["front_img_1"][b, src]
            )
            assert torch.equal(
                hist["obs"]["state_ee_pose"][b, 0], full["obs"]["state_ee_pose"][b, src]
            )
        # training ignores the eval slice
        algo.nets.train()
        train_hist = algo._build_obs_dict(processed[emb_id], emb_id, training=True)[
            "prompt"
        ]["metadata"]["history"]
        assert train_hist["action"].shape[1] == 3
        algo.nets.eval()
        # k = 0: no history at all
        algo.eval_history_chunks = 0
        hist0 = algo._build_obs_dict(processed[emb_id], emb_id, training=False)[
            "prompt"
        ]["metadata"]["history"]
        assert hist0["action"].shape[1] == 0 and hist0["metadata"]["mask"].shape == (
            4,
            0,
        )
        with torch.no_grad():
            preds = algo.forward_eval(processed)
        assert torch.isfinite(preds[f"{EMBODIMENT}_loss"])
    finally:
        algo.eval_history_chunks = None


def test_variable_lengths_share_positions(algo_history):
    """Age indexing: the newest chunk's embedded token is the same whether
    the history holds 1 chunk or many."""
    enc = algo_history.nets["policy"].obs_encoder
    torch.manual_seed(0)
    raw = torch.randn(1, HIST_MAX, enc.n_emb)
    mask = torch.zeros(1, HIST_MAX, dtype=torch.bool)
    with torch.no_grad():
        full = enc._embed_history(raw, mask)
        one = enc._embed_history(raw[:, -1:], mask[:, -1:])
        # left-aligned padding: a row with 3 valid chunks then padding
        mask3 = torch.ones(1, HIST_MAX, dtype=torch.bool)
        mask3[:, :3] = False
        three = enc._embed_history(raw, mask3)
    assert torch.allclose(full[:, -1], one[:, -1], atol=1e-6)
    assert torch.allclose(three[:, 2], full[:, -1] - raw[:, -1] + raw[:, 2], atol=1e-6)
    # different ages get different positions
    assert not torch.allclose(full[:, -1] - raw[:, -1], full[:, -2] - raw[:, -2])


def test_history_deploy_cycle(algo_history, emb_id):
    algo = algo_history
    algo.nets.eval()
    enc = algo.nets["policy"].obs_encoder
    chunk = _one_chunk_payload()
    with pytest.raises(RuntimeError, match="prompt\\(\\) before"):
        algo.push_history_chunk(chunk)
    batch = _build_history_batch(lengths=(4,), hist_lengths=(1,), groups=(0,))
    with torch.no_grad():
        processed = algo.process_batch_for_training(batch)
        obs_dict = algo._build_obs_dict(processed[emb_id], emb_id, training=False)
    prompt_dict = obs_dict.pop("prompt")
    algo.prompt(prompt_dict)
    assert enc.is_prompted and algo.history_len == 1
    # the caller's metadata is not mutated by prompt()
    assert "history" in prompt_dict["metadata"]
    assert algo.push_history_chunk(chunk) == 2
    assert algo.push_history_chunk(chunk) == 3
    with torch.no_grad():
        result = algo.nets["policy"].predict_action(obs_dict)
    assert result["action"].shape == (1, T, D)
    assert torch.isfinite(result["action"]).all()
    act = algo.predict_action_deployed(processed[emb_id], emb_id)
    assert act.shape == (1, T, D) and torch.isfinite(act).all()
    algo.reset()
    assert not enc.is_prompted and algo.history_len == 0
    with pytest.raises(RuntimeError, match="prompt\\(\\) before"):
        algo.push_history_chunk(chunk)


def test_incremental_cache_matches_batch_encoding(algo_history, emb_id):
    algo = algo_history
    algo.nets.eval()
    enc = algo.nets["policy"].obs_encoder
    batch = _build_history_batch(lengths=(4,), hist_lengths=(3,), groups=(0,))
    with torch.no_grad():
        processed = algo.process_batch_for_training(batch)
        obs_dict = algo._build_obs_dict(processed[emb_id], emb_id, training=False)
        hist = obs_dict["prompt"]["metadata"]["history"]
        raw_batch, mask_batch = enc._encode_history_raw(hist, 1, hist["action"].device)
        mem_batch, _ = enc._merge_memory(
            torch.zeros(1, 2, enc.n_emb), None, raw_batch, mask_batch
        )
    assert raw_batch.shape == (1, 3, enc.n_emb) and not mask_batch.any()

    # deploy: prompt without history, then push the three chunks one by one
    prompt_dict = dict(obs_dict["prompt"])
    metadata = dict(prompt_dict["metadata"])
    metadata.pop("history")
    prompt_dict["metadata"] = metadata
    algo.prompt(prompt_dict)
    assert algo.history_len == 0
    for k in range(3):
        enc.push_history_chunk(
            {
                "obs": {key: v[:, k : k + 1] for key, v in hist["obs"].items()},
                "action": hist["action"][:, k : k + 1],
                "metadata": {"mask": hist["metadata"]["mask"][:, k : k + 1]},
            }
        )
    assert algo.history_len == 3
    with torch.no_grad():
        raw_cache, mask_cache = enc._history_from_cache(raw_batch.device)
        mem_cache, _ = enc._merge_memory(
            torch.zeros(1, 2, enc.n_emb), None, raw_cache, mask_cache
        )
    assert torch.allclose(raw_cache, raw_batch, atol=1e-5)
    assert torch.allclose(mem_cache, mem_batch, atol=1e-5)

    # sliding window: only the newest max_chunks survive
    for _ in range(HIST_MAX):
        enc.push_history_chunk(
            {
                "obs": {key: v[:, :1] for key, v in hist["obs"].items()},
                "action": hist["action"][:, :1],
                "metadata": {"mask": hist["metadata"]["mask"][:, :1]},
            }
        )
    assert algo.history_len == HIST_MAX
    with torch.no_grad():
        raw_cache, _ = enc._history_from_cache(raw_batch.device)
    assert torch.allclose(raw_cache[:, -1], raw_batch[:, 0], atol=1e-5)
    algo.reset()


def test_history_seen_unseen_losses(algo_history, emb_id):
    algo = algo_history
    algo.nets.eval()
    batch = _build_history_batch()
    batch[EMBODIMENT]["operator_seen"] = torch.tensor([1, 1, 0, 0])
    with torch.no_grad():
        preds = algo.forward_eval(algo.process_batch_for_training(batch))
    assert torch.isfinite(preds[f"{EMBODIMENT}_loss"])
    assert torch.isfinite(preds[f"{EMBODIMENT}_loss_seen_operator"])
    assert torch.isfinite(preds[f"{EMBODIMENT}_loss_unseen_operator"])
    assert preds[f"{EMBODIMENT}_actions_cartesian"].shape == (4, T, D)


def test_history_disabled_drops_history_batch(algo_episode, algo_history, emb_id):
    # the plain episode model never moves the history to the device
    processed = algo_episode.process_batch_for_training(_build_history_batch())
    assert "history" not in processed[emb_id]
    algo_episode.nets.train()
    losses = algo_episode.compute_losses(
        algo_episode.forward_training(processed), processed
    )
    assert torch.isfinite(losses["action_loss"])
    # the history model needs it
    processed = algo_history.process_batch_for_training(_build_episode_batch())
    with pytest.raises(ValueError, match="carries no `history`"):
        algo_history._build_obs_dict(processed[emb_id], emb_id, training=False)


def test_history_encoder_requires_enabled_flag():
    cfg = _history_small_cfg()
    cfg.model.robomimic_model.prompt.history.enabled = False
    norm_stats, _ = _build_norm_stats()
    with pytest.raises(Exception, match="prompt.history.enabled is false"):
        hydra.utils.instantiate(cfg.model.robomimic_model, norm_stats=norm_stats)


def test_history_only_masks_demo(emb_id):
    cfg = _history_small_cfg()
    cfg.model.robomimic_model.prompt.history.history_only = True
    norm_stats, _ = _build_norm_stats()
    with pytest.raises(Exception, match="use_attention_sink"):
        hydra.utils.instantiate(cfg.model.robomimic_model, norm_stats=norm_stats)
    cfg.model.robomimic_model.policy.obs_encoder.use_attention_sink = True
    algo = hydra.utils.instantiate(cfg.model.robomimic_model, norm_stats=norm_stats)
    assert algo.history_only
    algo.nets.train()
    batch = _build_history_batch(hist_lengths=(0, 2, 1, 3))
    processed = algo.process_batch_for_training(batch)
    obs_dict = algo._build_obs_dict(processed[emb_id], emb_id, training=True)
    prompt = obs_dict["prompt"]
    # the demo prompt collapses to one fully-masked zero chunk
    assert (
        prompt["metadata"]["mask"].shape == (4, 1) and prompt["metadata"]["mask"].all()
    )
    assert (
        prompt["action"].shape == (4, 1, CHUNK_EP, D) and (prompt["action"] == 0).all()
    )
    assert prompt["obs"]["front_img_1"].shape == (4, 1, 3, 224, 224)
    assert all((v == 0).all() for v in prompt["obs"].values())
    # the batch's own mask was not written in place
    assert not processed[emb_id]["prompt"]["metadata"]["mask"].all()
    # history still present, row 0 has none: only the sinks are attendable
    assert prompt["metadata"]["history"]["metadata"]["mask"].sum(1).tolist() == [
        3,
        1,
        2,
        0,
    ]
    losses = algo.compute_losses(algo.forward_training(processed), processed)
    assert torch.isfinite(losses["action_loss"])
    algo.nets.zero_grad(set_to_none=True)
    losses["action_loss"].backward()
    _assert_all_grads(algo)
