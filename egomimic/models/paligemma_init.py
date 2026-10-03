"""PaliGemma-only initialization for the pi0.5 chassis.

``egomimic.algo.pi.PI`` normally starts from the full pi0.5 base checkpoint.
This module is the other entry point: the VLM half (SigLIP tower, multimodal
projector, Gemma-2B prefix) comes from the public PaliGemma release, and
everything pi0.5 adds on top -- the action expert, the action in/out
projections, the adaRMS time MLPs -- keeps its PyTorch init.

openpi builds the prefix as a stock ``PaliGemmaForConditionalGeneration`` whose
config is the ``gemma_2b`` variant (width 2048, depth 18, head_dim 256) next to
a SigLIP-so400m/14 tower, which is exactly the ``paligemma-3b-*-224``
architecture, so the released checkpoint drops straight in.
"""

import logging
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)

# License-gated: pulling it needs an HF token that has accepted the Gemma terms.
CANONICAL_REPO = "google/paligemma-3b-pt-224"

# Released PaliGemma checkpoint keys -> the module tree. These are what
# transformers applies via
# ``PaliGemmaForConditionalGeneration._checkpoint_conversion_mapping``; spelled
# out here because we stream the shards ourselves rather than going through
# ``from_pretrained`` (which would build a second full-size model alongside the
# one openpi already made). The vision rule is the transformers >= 5 spelling,
# which flattened ``SiglipVisionModel``; below 5 the ``vision_model.`` hop is
# still there, and ``_module_name`` picks whichever the live tree has.
#
# ``load_paligemma_weights`` fails loudly if these leave any parameter
# uncovered, so a future rename surfaces as an error, not a silent partial load.
_KEY_REMAP = (
    (re.compile(r"^language_model\.model\."), "model.language_model."),
    (re.compile(r"^language_model\.lm_head\."), "lm_head."),
    (re.compile(r"^vision_tower\.vision_model\."), "model.vision_tower."),
    (re.compile(r"^vision_tower\."), "model.vision_tower."),
    (re.compile(r"^multi_modal_projector\."), "model.multi_modal_projector."),
)


# The released checkpoint pads the token embedding from PaliGemma's 257152-token
# vocabulary up to 257216 rows. Row 257152 is <image> -- which openpi never looks
# up, because ``embed_prefix`` concatenates SigLIP features directly instead of
# scattering them into an <image> placeholder -- and the 63 rows after it are
# unused padding. openpi builds the table at exactly 257152, so these two
# tensors are truncated on their vocabulary axis; every real text token keeps its
# row, since the padding is at the end.
_VOCAB_TRUNCATABLE = frozenset(
    {"model.language_model.embed_tokens.weight", "lm_head.weight"}
)


def _remap_key(key: str) -> str:
    for pattern, replacement in _KEY_REMAP:
        new, n = pattern.subn(replacement, key, count=1)
        if n:
            return new
    return key


def _module_name(key: str, own: dict) -> str:
    """Where checkpoint ``key`` lives in ``own``, the model's named tensors."""
    name = _remap_key(key)
    if name not in own and key.startswith("vision_tower.vision_model."):
        nested = "model." + key
        if nested in own:
            return nested
    return name


def _fit(name: str, tensor: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    if tensor.shape == param.shape:
        return tensor
    if (
        name in _VOCAB_TRUNCATABLE
        and tensor.ndim == param.ndim == 2
        and tensor.shape[0] > param.shape[0]
        and tensor.shape[1] == param.shape[1]
    ):
        logger.info(
            "Truncating %s from %d to %d vocabulary rows (padding tokens).",
            name,
            tensor.shape[0],
            param.shape[0],
        )
        return tensor[: param.shape[0]]
    raise ValueError(
        f"Shape mismatch for {name}: checkpoint {tuple(tensor.shape)} "
        f"vs model {tuple(param.shape)}"
    )


def resolve_paligemma_dir(source: str) -> str:
    """Local directory holding the PaliGemma checkpoint named by ``source``: a
    local path as-is, else the Hugging Face repo's shards."""
    if os.path.isdir(source):
        return source
    from huggingface_hub import snapshot_download

    # Only the shards: the loader reads nothing else and openpi builds the
    # config in code.
    return snapshot_download(source, allow_patterns=["*.safetensors"])


def load_paligemma_weights(model, source: str) -> str:
    """Copy a PaliGemma checkpoint into ``model``'s VLM prefix, in place.

    ``model`` is an ``openpi.models_pytorch.pi0_pytorch.PI0Pytorch``. Only
    ``paligemma_with_expert.paligemma`` is touched; the action expert and the
    action/time projections are left at their init. Returns the directory the
    weights came from.
    """
    directory = resolve_paligemma_dir(source)
    target = model.paligemma_with_expert.paligemma

    # remove_duplicate=False so a tied lm_head.weight is reachable under both
    # names; the checkpoint only carries one of them.
    params = dict(target.named_parameters(remove_duplicate=False))
    # Buffers are accepted if the checkpoint ships them, but are not required:
    # rotary inv_freq, Gemma's embed_scale and the SigLIP position_ids are all
    # derived from the config at construction time and are absent from the
    # release.
    own = dict(target.named_buffers(remove_duplicate=False))
    own.update(params)

    shards = sorted(Path(directory).glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors under {directory}")

    loaded, unexpected = set(), []
    for shard in shards:
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open is not a Mapping
                name = _module_name(key, own)
                param = own.get(name)
                if param is None:
                    unexpected.append(key)
                    continue
                tensor = _fit(name, handle.get_tensor(key), param)
                with torch.no_grad():
                    param.copy_(tensor.to(param.dtype))
                loaded.add(name)

    # lm_head is tied to embed_tokens and, either way, dead weight here: no code
    # path in openpi's PyTorch pi0 reads it.
    tied = set(getattr(type(target), "_tied_weights_keys", None) or ())
    missing = {n for n in params if n not in loaded} - tied
    if missing or unexpected:
        raise RuntimeError(
            f"PaliGemma load from {directory} did not cover the prefix exactly: "
            f"{len(missing)} missing (e.g. {sorted(missing)[:5]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:5]})."
        )

    logger.info(
        "Initialized the VLM prefix from PaliGemma at %s (%d tensors, %d parameters). "
        "The action expert and the action/time projections stay randomly initialized.",
        directory,
        len(loaded),
        sum(p.numel() for p in target.parameters()),
    )
    return directory
