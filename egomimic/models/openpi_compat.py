"""Adapters that let openpi's PyTorch pi0 run against transformers 5.

`external/openpi` pins `transformers==4.53.2` and ships a `transformers_replace`
patch written against it (its own `check.py` asserts that version). This repo
runs transformers 5, where the stock modules have moved on, so a few of openpi's
call sites no longer line up. Nothing here runs below transformers 5, and each
shim is a no-op when it finds openpi's own spelling already in place -- so
porting the fork's patch forward makes this module inert. Delete it then.

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
    )
    applied = [name for name, fn in shims if fn()]
    if applied:
        logger.info(
            "openpi targets transformers 4.53 and this environment has %s; "
            "applied compatibility shims: %s",
            transformers.__version__,
            ", ".join(applied),
        )
