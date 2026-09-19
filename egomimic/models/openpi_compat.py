"""Adapters that let openpi's PyTorch pi0 run against transformers 5.

`external/openpi` pins `transformers==4.53.2` and ships a `transformers_replace`
patch written against it (its own `check.py` asserts that version), which is also
this repo's pin. A venv on transformers 5 works too, but there the stock modules
have moved on, so a few of openpi's call sites no longer line up. Nothing here
runs below transformers 5, and each shim is a no-op when it finds openpi's own
spelling already in place -- so porting the fork's patch forward makes this
module inert. Delete it then.

Applied from `egomimic.algo.pi`, which is the only importer of openpi.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_APPLIED = False


def _patch_embed_image() -> bool:
    """`get_image_features` returns a `BaseModelOutputWithPooling`, not a tensor.

    transformers 5's `PaliGemmaModel.get_image_features` returns the vision
    output object with the projected features stashed on `pooler_output`, while
    openpi's `embed_prefix` reads `.shape` off the result. Unwrap it.
    """
    from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel

    original = PaliGemmaWithExpertModel.embed_image

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        features = original(self, image)
        if isinstance(features, torch.Tensor):
            return features
        return features.pooler_output

    PaliGemmaWithExpertModel.embed_image = embed_image
    return True


def _patch_paligemma_bc_properties() -> bool:
    """`paligemma.language_model` / `.vision_tower` / `.multi_modal_projector`.

    transformers 5 moved those submodules under `PaliGemmaModel` and dropped the
    back-compat properties that used to forward to them from
    `PaliGemmaForConditionalGeneration`. openpi still reads all three off the
    outer model. Put the forwarders back.
    """
    from transformers import PaliGemmaForConditionalGeneration

    added = []
    for name in ("language_model", "vision_tower", "multi_modal_projector"):
        if hasattr(PaliGemmaForConditionalGeneration, name):
            continue
        setattr(
            PaliGemmaForConditionalGeneration,
            name,
            property(lambda self, _name=name: getattr(self.model, _name)),
        )
        added.append(name)
    return bool(added)


# openpi keeps SigLIP's patch and position embeddings in float32 by substring on
# `vision_tower.vision_model.embeddings.*`; the flattened 5.x spelling of the same
# three tensors.
_FP32_KEEP_FLAT = (
    "vision_tower.embeddings.patch_embedding.",
    "vision_tower.embeddings.position_embedding.",
)


def _patch_fp32_keep_list() -> bool:
    """openpi's float32 keep-list misses the SigLIP embeddings.

    `to_bfloat16_for_selected_params` casts everything to bf16 and then restores
    float32 on names containing `vision_tower.vision_model.embeddings.*`.
    transformers 5 flattened `SiglipVisionModel`, so nothing matches and the
    patch embedding and position embedding stay bf16 -- the tensors openpi
    protects because small updates vanish in bf16. Re-apply the keep under the
    flattened names; with the nested names present they match nothing.
    """
    from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel

    original = PaliGemmaWithExpertModel.to_bfloat16_for_selected_params

    def to_bfloat16_for_selected_params(self, precision="bfloat16"):
        original(self, precision)
        if precision != "bfloat16":
            return
        for name, param in self.named_parameters():
            if any(selector in name for selector in _FP32_KEEP_FLAT):
                param.data = param.data.to(dtype=torch.float32)

    PaliGemmaWithExpertModel.to_bfloat16_for_selected_params = (
        to_bfloat16_for_selected_params
    )
    return True


def apply() -> None:
    """Install every shim once. Safe to call repeatedly."""
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    import transformers

    major = int(transformers.__version__.split(".")[0])
    if major < 5:
        return

    shims = (
        ("embed_image", _patch_embed_image),
        ("paligemma_bc_properties", _patch_paligemma_bc_properties),
        ("fp32_keep_list", _patch_fp32_keep_list),
    )
    applied = [name for name, fn in shims if fn()]
    if applied:
        logger.info(
            "openpi targets transformers 4.53 and this environment has %s; "
            "applied compatibility shims: %s",
            transformers.__version__,
            ", ".join(applied),
        )
