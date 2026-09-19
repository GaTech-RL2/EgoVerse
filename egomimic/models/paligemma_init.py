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

import hashlib
import logging
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)

CANONICAL_REPO = "google/paligemma-3b-pt-224"

# `CANONICAL_REPO` is license-gated: a machine without an HF token that has
# accepted the Gemma terms gets a 401 on the weight shards (the repo *metadata*
# stays anonymously readable, which is what makes the check below possible).
# These repos carry the same three shards. Every one is sha256-checked against
# the canonical repo's file metadata before it is used, so a mirror cannot
# substitute different weights -- it is only a transport.
MIRROR_REPOS = (
    "leo009/paligemma-3b-pt-224",
    "hehe156/paligemma-3b-pt-224",
)

_ALLOW_PATTERNS = ["*.json", "*.safetensors"]

# Released PaliGemma checkpoint keys -> this transformers version's module tree.
# The first three are what transformers applies via
# ``PaliGemmaForConditionalGeneration._checkpoint_conversion_mapping``; spelled
# out here because we stream the shards ourselves rather than going through
# ``from_pretrained`` (which would build a second full-size model alongside the
# one openpi already made). The fourth is the extra hop that mapping does not
# cover: transformers >= 5 flattened ``SiglipVisionModel``, so the checkpoint's
# ``vision_tower.vision_model.*`` is just ``vision_tower.*`` here.
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


def select_init_source(
    pytorch_weight_path: str | None, paligemma_weight_path: str | None
) -> str:
    """Which set of starting weights a PI run uses: the full pi0.5 base
    checkpoint (``"pi05_base"``), PaliGemma with a fresh action expert
    (``"paligemma"``), or nothing at all (``"none"``)."""
    if pytorch_weight_path is not None and paligemma_weight_path is not None:
        raise ValueError(
            "pytorch_weight_path and paligemma_weight_path are both set. The "
            "pi0.5 base checkpoint already contains a trained PaliGemma; set "
            "pytorch_weight_path=null to start from PaliGemma with a fresh "
            "action expert."
        )
    if pytorch_weight_path is not None:
        return "pi05_base"
    if paligemma_weight_path is not None:
        return "paligemma"
    return "none"


def _shard_digests(repo: str) -> dict[str, str]:
    """``{shard filename: sha256}`` from a repo's file metadata."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, files_metadata=True)
    digests = {}
    for sibling in info.siblings:
        if not sibling.rfilename.endswith(".safetensors"):
            continue
        lfs = getattr(sibling, "lfs", None)
        sha = (
            lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        )
        if sha:
            digests[sibling.rfilename] = sha
    return digests


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_shards(directory: str, expected: dict[str, str]) -> None:
    root = Path(directory)
    for name, sha in expected.items():
        shard = root / name
        if not shard.is_file():
            raise FileNotFoundError(f"{shard} is missing from the mirror snapshot")
        actual = _sha256(shard)
        if actual != sha:
            raise ValueError(
                f"{shard} does not match {CANONICAL_REPO}: expected sha256 {sha}, "
                f"got {actual}. Refusing to initialize from it."
            )


def resolve_paligemma_dir(source: str, *, allow_mirror: bool = True) -> str:
    """Local directory holding the PaliGemma checkpoint named by ``source``.

    ``source`` is a local path (returned as-is) or a Hugging Face repo id. When
    it is the gated canonical repo and this machine is not authenticated for it,
    fall back to a sha256-verified mirror.
    """
    if os.path.isdir(source):
        return source

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    try:
        return snapshot_download(source, allow_patterns=_ALLOW_PATTERNS)
    except (GatedRepoError, HfHubHTTPError) as gated:
        if not allow_mirror or source != CANONICAL_REPO:
            raise
        logger.warning(
            "%s is gated for this machine (%s). Falling back to a mirror, "
            "sha256-verified against %s. Set HF_TOKEN to a token that has "
            "accepted the Gemma terms to pull the canonical repo instead.",
            CANONICAL_REPO,
            type(gated).__name__,
            CANONICAL_REPO,
        )

    expected = _shard_digests(CANONICAL_REPO)
    if not expected:
        raise RuntimeError(
            f"Cannot read shard checksums for {CANONICAL_REPO}; refusing to use "
            "an unverified mirror."
        )

    failures = []
    for mirror in MIRROR_REPOS:
        try:
            if _shard_digests(mirror) != expected:
                raise ValueError("shard checksums differ from the canonical repo")
            directory = snapshot_download(mirror, allow_patterns=_ALLOW_PATTERNS)
            _verify_shards(directory, expected)
        except Exception as exc:  # try the next mirror, report them all if none works
            failures.append(f"{mirror}: {type(exc).__name__}: {exc}")
            continue
        logger.warning(
            "Initializing PaliGemma from mirror %s (all %d shards sha256-match %s).",
            mirror,
            len(expected),
            CANONICAL_REPO,
        )
        return directory

    raise RuntimeError(
        f"{CANONICAL_REPO} is gated and no verified mirror worked:\n  "
        + "\n  ".join(failures)
    )


def load_paligemma_weights(model, source: str, *, allow_mirror: bool = True) -> str:
    """Copy a PaliGemma checkpoint into ``model``'s VLM prefix, in place.

    ``model`` is an ``openpi.models_pytorch.pi0_pytorch.PI0Pytorch``. Only
    ``paligemma_with_expert.paligemma`` is touched; the action expert and the
    action/time projections are left at their init. Returns the directory the
    weights came from.
    """
    directory = resolve_paligemma_dir(source, allow_mirror=allow_mirror)
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
                name = _remap_key(key)
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
