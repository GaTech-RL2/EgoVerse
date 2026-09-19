"""PaliGemma-only init: the VLM prefix comes from the public PaliGemma release
and everything pi0.5 adds on top starts from its PyTorch init.

No network and no 3B model: the loader is exercised against a miniature module
tree and a two-shard safetensors file written on the fly, and the download
resolution is driven through stubs. The one thing that needs the real thing --
that the released checkpoint covers openpi's prefix exactly -- is asserted by
``load_paligemma_weights`` itself (it raises on any missing or unexpected
tensor), so a rename upstream cannot pass silently.
"""

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from egomimic.models import paligemma_init as pgi

CANONICAL = pgi.CANONICAL_REPO


# ---------------------------------------------------------------- key remapping


# Spelled exactly as google/paligemma-3b-pt-224 ships them.
@pytest.mark.parametrize(
    ("checkpoint_key", "module_name"),
    [
        (
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
        ),
        (
            "language_model.model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
        ),
        ("language_model.model.norm.weight", "model.language_model.norm.weight"),
        ("language_model.lm_head.weight", "lm_head.weight"),
        # transformers >= 5 flattened SiglipVisionModel: the `.vision_model.` hop
        # in the checkpoint has no counterpart in the module tree.
        (
            "vision_tower.vision_model.encoder.layers.26.mlp.fc2.bias",
            "model.vision_tower.encoder.layers.26.mlp.fc2.bias",
        ),
        (
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "model.vision_tower.embeddings.patch_embedding.weight",
        ),
        (
            "multi_modal_projector.linear.bias",
            "model.multi_modal_projector.linear.bias",
        ),
    ],
)
def test_checkpoint_keys_map_onto_the_module_tree(checkpoint_key, module_name):
    assert pgi._remap_key(checkpoint_key) == module_name


def test_unknown_prefix_is_left_alone():
    assert pgi._remap_key("action_in_proj.weight") == "action_in_proj.weight"


# ------------------------------------------------------------- vocab truncation


def test_padded_vocabulary_is_truncated_to_the_model_width():
    # The release pads 257152 -> 257216; the extra rows are <image> + padding.
    tensor, param = torch.arange(12.0).reshape(6, 2), torch.zeros(4, 2)
    fitted = pgi._fit("model.language_model.embed_tokens.weight", tensor, param)
    assert fitted.shape == param.shape
    assert torch.equal(fitted, tensor[:4])  # padding is at the END


def test_truncation_is_not_applied_to_other_tensors():
    with pytest.raises(ValueError, match="Shape mismatch"):
        pgi._fit(
            "model.vision_tower.encoder.layers.0.mlp.fc1.weight",
            torch.zeros(6, 2),
            torch.zeros(4, 2),
        )


def test_a_narrower_checkpoint_is_never_padded_out():
    with pytest.raises(ValueError, match="Shape mismatch"):
        pgi._fit("lm_head.weight", torch.zeros(3, 2), torch.zeros(4, 2))


# ------------------------------------------------------------------ init source


def test_pi05_base_wins_when_only_it_is_set():
    assert pgi.select_init_source("/ckpt/pi05_base_pytorch", None) == "pi05_base"


def test_paligemma_when_the_base_checkpoint_is_off():
    assert pgi.select_init_source(None, CANONICAL) == "paligemma"


def test_neither_is_reported_as_no_weights():
    assert pgi.select_init_source(None, None) == "none"


def test_both_at_once_is_refused():
    # The pi0.5 base already contains a trained PaliGemma; silently letting one
    # win would make the run's provenance unreadable from its config.
    with pytest.raises(ValueError, match="both set"):
        pgi.select_init_source("/ckpt/pi05_base_pytorch", CANONICAL)


# ------------------------------------------------------------ the loader itself


class _Prefix(nn.Module):
    """Shaped like the part of ``PaliGemmaForConditionalGeneration`` the loader
    walks: a ``model.`` subtree, a tied ``lm_head`` and a derived buffer."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, vocab=4, width=3):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(vocab, width)
        self.model.vision_tower = nn.Module()
        self.model.vision_tower.post_layernorm = nn.Linear(width, width)
        self.lm_head = nn.Linear(width, vocab, bias=False)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight  # tied
        # config-derived, absent from every released checkpoint
        self.model.language_model.register_buffer("inv_freq", torch.ones(width))


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = _Prefix()
        self.action_in_proj = nn.Linear(3, 3)


def _write_checkpoint(tmp_path, tensors, shards=2):
    items = list(tensors.items())
    size = -(-len(items) // shards)
    for i in range(shards):
        chunk = dict(items[i * size : (i + 1) * size])
        if chunk:
            save_file(chunk, str(tmp_path / f"model-{i}.safetensors"))
    return str(tmp_path)


def _full_checkpoint(vocab=4, width=3):
    torch.manual_seed(1)
    return {
        "language_model.model.embed_tokens.weight": torch.randn(vocab, width),
        "vision_tower.vision_model.post_layernorm.weight": torch.randn(width, width),
        "vision_tower.vision_model.post_layernorm.bias": torch.randn(width),
    }


def test_the_prefix_loads_and_nothing_else_moves(tmp_path):
    model = _Model()
    before = model.action_in_proj.weight.detach().clone()
    tensors = _full_checkpoint()
    directory = _write_checkpoint(tmp_path, tensors)

    assert pgi.load_paligemma_weights(model, directory) == directory

    prefix = model.paligemma_with_expert.paligemma
    assert torch.equal(
        prefix.model.language_model.embed_tokens.weight,
        tensors["language_model.model.embed_tokens.weight"],
    )
    assert torch.equal(
        prefix.model.vision_tower.post_layernorm.bias,
        tensors["vision_tower.vision_model.post_layernorm.bias"],
    )
    # everything outside the prefix is untouched
    assert torch.equal(model.action_in_proj.weight, before)


def test_the_tied_head_follows_the_embedding(tmp_path):
    # The checkpoint carries no lm_head; tying must still leave it consistent.
    model = _Model()
    tensors = _full_checkpoint()
    pgi.load_paligemma_weights(model, _write_checkpoint(tmp_path, tensors))
    prefix = model.paligemma_with_expert.paligemma
    assert torch.equal(
        prefix.lm_head.weight, prefix.model.language_model.embed_tokens.weight
    )


def test_a_derived_buffer_is_not_required(tmp_path):
    model = _Model()
    pgi.load_paligemma_weights(model, _write_checkpoint(tmp_path, _full_checkpoint()))
    assert torch.equal(
        model.paligemma_with_expert.paligemma.model.language_model.inv_freq,
        torch.ones(3),
    )


def test_a_partial_checkpoint_is_refused(tmp_path):
    # Silently leaving part of the VLM at its random init is the failure this
    # whole module exists to avoid.
    tensors = _full_checkpoint()
    del tensors["vision_tower.vision_model.post_layernorm.weight"]
    with pytest.raises(RuntimeError, match="missing"):
        pgi.load_paligemma_weights(_Model(), _write_checkpoint(tmp_path, tensors))


def test_an_unknown_tensor_is_refused(tmp_path):
    tensors = _full_checkpoint()
    tensors["vision_tower.vision_model.renamed_upstream.weight"] = torch.zeros(3)
    with pytest.raises(RuntimeError, match="unexpected"):
        pgi.load_paligemma_weights(_Model(), _write_checkpoint(tmp_path, tensors))


def test_an_empty_directory_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="safetensors"):
        pgi.load_paligemma_weights(_Model(), str(tmp_path))


def test_a_padded_vocabulary_checkpoint_loads(tmp_path):
    # The real release is 64 rows wider than openpi's table.
    tensors = _full_checkpoint()
    wide = torch.randn(6, 3)
    tensors["language_model.model.embed_tokens.weight"] = wide
    pgi.load_paligemma_weights(_Model(), _write_checkpoint(tmp_path, tensors))


# ------------------------------------------------------- download / mirror path


def test_a_local_directory_is_used_as_is(tmp_path):
    assert pgi.resolve_paligemma_dir(str(tmp_path)) == str(tmp_path)


def _gated_error(repo_id):
    """The 401 huggingface_hub raises for a license-gated repo."""
    import requests
    from huggingface_hub.errors import GatedRepoError

    response = requests.Response()
    response.status_code = 401
    return GatedRepoError(f"401 for {repo_id}", response=response)


def _stub_hub(monkeypatch, *, gated, digests, downloads):
    import huggingface_hub

    def snapshot_download(repo_id, **_):
        if repo_id in gated:
            raise _gated_error(repo_id)
        return downloads[repo_id]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(pgi, "_shard_digests", lambda repo: digests[repo])
    monkeypatch.setattr(pgi, "_verify_shards", lambda directory, expected: None)


def test_the_canonical_repo_is_preferred(monkeypatch, tmp_path):
    _stub_hub(
        monkeypatch, gated=set(), digests={}, downloads={CANONICAL: str(tmp_path)}
    )
    assert pgi.resolve_paligemma_dir(CANONICAL) == str(tmp_path)


def test_a_gated_canonical_repo_falls_back_to_a_matching_mirror(monkeypatch, tmp_path):
    good = pgi.MIRROR_REPOS[0]
    digests = {
        CANONICAL: {"model.safetensors": "abc"},
        good: {"model.safetensors": "abc"},
    }
    digests.update({m: {"model.safetensors": "abc"} for m in pgi.MIRROR_REPOS})
    _stub_hub(
        monkeypatch,
        gated={CANONICAL},
        digests=digests,
        downloads={m: str(tmp_path) for m in pgi.MIRROR_REPOS},
    )
    assert pgi.resolve_paligemma_dir(CANONICAL) == str(tmp_path)


def test_a_mirror_whose_shards_differ_is_rejected(monkeypatch, tmp_path):
    # A mirror is transport only: different bytes means it is not PaliGemma.
    digests = {CANONICAL: {"model.safetensors": "abc"}}
    digests.update({m: {"model.safetensors": "TAMPERED"} for m in pgi.MIRROR_REPOS})
    _stub_hub(
        monkeypatch,
        gated={CANONICAL},
        digests=digests,
        downloads={m: str(tmp_path) for m in pgi.MIRROR_REPOS},
    )
    with pytest.raises(RuntimeError, match="no verified mirror"):
        pgi.resolve_paligemma_dir(CANONICAL)


def test_the_mirror_fallback_can_be_switched_off(monkeypatch, tmp_path):
    from huggingface_hub.errors import GatedRepoError

    _stub_hub(monkeypatch, gated={CANONICAL}, digests={}, downloads={})
    with pytest.raises(GatedRepoError):
        pgi.resolve_paligemma_dir(CANONICAL, allow_mirror=False)


def test_a_non_canonical_repo_never_falls_back(monkeypatch, tmp_path):
    # Only the gated canonical repo has verified stand-ins; anything else the
    # user names must fail loudly rather than silently load other weights.
    from huggingface_hub.errors import GatedRepoError

    _stub_hub(
        monkeypatch, gated={"someone/private-paligemma"}, digests={}, downloads={}
    )
    with pytest.raises(GatedRepoError):
        pgi.resolve_paligemma_dir("someone/private-paligemma")


# -------------------------------------------------------------------- the config


def test_the_overlay_flips_the_init_source_and_keeps_its_parent(compose_resolve):
    cfg = compose_resolve(
        "train_zarr_cartesian_pi", ["model=pi0.5_bc_mecka_6d_paligemma_init"]
    )
    model = cfg.model.robomimic_model
    assert model.config.pytorch_weight_path is None
    assert model.config.paligemma_weight_path == CANONICAL
    assert (
        pgi.select_init_source(
            model.config.pytorch_weight_path, model.config.paligemma_weight_path
        )
        == "paligemma"
    )
    # the two keys are all the overlay touches: pi0.5_bc_mecka_6d's own choices
    # (cartesian 6D action, no Embodiment prompt block) must survive it
    assert model.ac_keys.human_bimanual == "actions_cartesian"
    assert model.embodiment_label is False
    assert model.config.model.action_dim == 32
    assert (
        model.action_converters.rules.HUMAN_BIMANUAL._target_
        == "egomimic.utils.action_utils.HumanBimanualCartesianEuler"
    )


def test_the_base_pi_config_still_starts_from_pi05_base(compose_resolve):
    cfg = compose_resolve("train_zarr_cartesian_pi", ["model=pi0.5_bc_mecka_6d"])
    config = cfg.model.robomimic_model.config
    assert config.paligemma_weight_path is None
    assert (
        pgi.select_init_source(config.pytorch_weight_path, config.paligemma_weight_path)
        == "pi05_base"
    )
