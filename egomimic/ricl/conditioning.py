"""RICL prefix-conditioning helpers (import-light: torch + numpy only).

These functions implement the *only* real surgery RICL needs on pi0.5. Per P0,
``PI0Pytorch.embed_prefix`` iterates over **all** images in the observation and
embeds the full ``tokenized_prompt``, with the entire prefix attended
bidirectionally. So injecting k retrieved demonstrations is purely a matter of:

1. ``augment_images_with_retrieved`` - append the k retrieved ``base_0_rgb``
   frames as extra entries in the observation's ``images``/``image_masks`` dicts
   (the model embeds them into the prefix automatically), and
2. ``build_retrieved_prompt_block`` - turn each retrieved demo's (state, action)
   into discretized **text tokens** spliced into the prompt, reusing the same
   binning pi0.5 already uses for the State block (``PI._discretize_state_for_sample``).

Kept free of any ``openpi`` / ``egomimic.algo`` import so the logic is unit-testable
on a CPU-only box (see ``egomimic/ricl/tests/conditioning_test.py``). The pi0.5 image
resize (``openpi.shared.image_tools.resize_with_pad_torch``) is imported lazily and
falls back to ``F.interpolate`` when openpi is unavailable.
"""

from __future__ import annotations

import numpy as np
import torch

# The pi0.5 generation anchor appended by ``PI._build_prompts`` (keep in sync).
PI05_ANCHOR = ";\nAction: "


# ---------------------------------------------------------------------------
# Discretization (mirrors PI._discretize_state_for_sample binning exactly)
# ---------------------------------------------------------------------------


def _bin_edges(num_bins: int, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    # Matches pi.py: np.linspace(lo, hi, num_bins + 1)[:-1]
    return np.linspace(lo, hi, num_bins + 1)[:-1]


def discretize_vector(
    v, num_bins: int = 256, lo: float = -1.0, hi: float = 1.0
) -> list[int]:
    """Clip a vector to [lo, hi] and digitize into ``num_bins`` bins.

    Accepts a torch.Tensor or array-like; returns a flat list[int] of bin ids,
    identical to the State-block discretization used in ``PI._build_prompts``.
    """
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    v = np.clip(v, lo, hi)
    bins = np.digitize(v, bins=_bin_edges(num_bins, lo, hi)) - 1
    return bins.astype(int).tolist()


def build_retrieved_prompt_block(
    states,
    actions,
    valid=None,
    *,
    prompt: str = "",
    num_bins: int = 256,
    action_steps: int = 1,
) -> str:
    """Build in-context demo *exemplars* for one sample's k retrieved demos.

    Each demo is rendered as a self-contained ``State -> Action`` exemplar that
    mirrors the pi0.5 query block format, so the model sees a sequence of
    demonstrations followed by the query (prompt-structure RICL, after the
    original ``ricl_openpi`` per-observation blocks — but flattened into one text
    prefix, *no* attention-mask surgery)::

        'Task: {prompt}, State: {s};\\nAction: {a}|'

    The action carries the FULL retrieved chunk (``action_steps`` leading steps,
    flattened over (step, dim)) so the discriminative *trajectory* — not just one
    near-zero step — reaches the model. That is what lets a kNN-retrieved demo
    differ from a random one; encoding a single step throws the signal away.

    Args:
        states:  ``(k, Ds)`` retrieved proprio (already in the query's normalized
                 convention so bins are comparable).
        actions: ``(k, Ha, Da)`` or ``(k, Da)`` retrieved actions (same norm).
        valid:   ``(k,)`` bool/0-1 mask; falsy entries are skipped (padding).
        prompt:  the demos' task text (within-group retrieval -> the query's task).
        action_steps: how many leading action steps to encode per demo.

    Returns:
        e.g. ``"Task: idli, State: 12 200 ...;\\nAction: 130 90 ...| Task: idli, ..."``
        or ``""`` when no demo is valid.
    """
    states = _to_numpy(states)
    actions = _to_numpy(actions)
    k = states.shape[0]
    if valid is None:
        valid = np.ones(k, dtype=bool)
    else:
        valid = _to_numpy(valid).reshape(-1).astype(bool)

    if actions.ndim == 3:  # (k, Ha, Da) -> keep leading `action_steps`, flatten
        actions = actions[:, :action_steps, :].reshape(k, -1)
    elif actions.ndim == 2:  # (k, Da)
        pass
    else:
        raise ValueError(f"actions must be (k,Ha,Da) or (k,Da), got {actions.shape}")

    head = f"Task: {prompt}" if prompt else "Demo"
    pieces = []
    for i in range(k):
        if not valid[i]:
            continue
        s = " ".join(map(str, discretize_vector(states[i], num_bins)))
        a = " ".join(map(str, discretize_vector(actions[i], num_bins)))
        pieces.append(f"{head}, State: {s}{PI05_ANCHOR}{a}|")
    return " ".join(pieces)


def splice_retrieved_into_prompt(prompt: str, retrieved_block: str) -> str:
    """Prepend the retrieved demo exemplars in front of the query block.

    In-context learning order is *demonstrations then query*: the demos go first,
    then the full query block (which already ends with the pi0.5 anchor
    ``;\\nAction: ``). The anchor must stay terminal to cue generation, so we never
    touch the query's tail — we only prepend. Empty block -> passthrough.
    """
    if not retrieved_block:
        return prompt
    return f"{retrieved_block} {prompt}"


# ---------------------------------------------------------------------------
# Image-dict augmentation
# ---------------------------------------------------------------------------


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_bchw(t: torch.Tensor) -> torch.Tensor:
    if t.ndim != 4:
        raise ValueError(f"expected 4D image, got {tuple(t.shape)}")
    if t.shape[1] in (1, 3):
        return t
    if t.shape[-1] in (1, 3):
        return t.permute(0, 3, 1, 2).contiguous()
    return t


def _to_minus1_1(img: torch.Tensor) -> torch.Tensor:
    if img.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        img = img.to(torch.float32) / 255.0
    else:
        img = torch.clamp(img.to(torch.float32), 0.0, 1.0)
    return img * 2.0 - 1.0


def _resize_to(img: torch.Tensor, hw) -> torch.Tensor:
    if tuple(img.shape[-2:]) == tuple(hw):
        return img
    try:  # match the base path exactly when openpi is present
        from openpi.shared.image_tools import resize_with_pad_torch

        return resize_with_pad_torch(img, *hw)
    except Exception:
        return torch.nn.functional.interpolate(
            img, size=tuple(hw), mode="bilinear", align_corners=False
        )


def augment_images_with_retrieved(
    images: dict,
    image_masks: dict,
    retrieved_images: torch.Tensor,
    retrieved_mask: torch.Tensor | None = None,
    *,
    image_resolution=(224, 224),
    base_key: str = "base_0_rgb",
    key_prefix: str = "retrieved",
) -> tuple[dict, dict]:
    """Append k retrieved ``base_0_rgb`` frames to the observation image dicts.

    Args:
        images / image_masks: the query observation dicts (mutated in place).
        retrieved_images: ``(B, k, C, H, W)`` or ``(B, k, H, W, C)``.
        retrieved_mask:   ``(B, k)`` bool (valid neighbor). Defaults to all-True.

    Returns the (images, image_masks) dicts with ``k`` new keys
    ``{key_prefix}_{i}_{base_key}`` appended (so the model embeds them into the
    bidirectional prefix). Query camera entries are left untouched / first, which
    preserves the pretrained model's camera-slot priors.
    """
    if retrieved_images.ndim != 5:
        raise ValueError(
            f"retrieved_images must be (B,k,C,H,W) or (B,k,H,W,C), got {tuple(retrieved_images.shape)}"
        )
    B, k = retrieved_images.shape[0], retrieved_images.shape[1]
    device = next(iter(images.values())).device if images else retrieved_images.device
    for i in range(k):
        img_i = _ensure_bchw(retrieved_images[:, i].to(device))
        img_i = _to_minus1_1(img_i)
        img_i = _resize_to(img_i, image_resolution)
        key = f"{key_prefix}_{i}_{base_key}"
        images[key] = img_i
        if retrieved_mask is not None:
            mask_i = retrieved_mask[:, i].to(device=device, dtype=torch.bool)
        else:
            mask_i = torch.ones(B, dtype=torch.bool, device=device)
        image_masks[key] = mask_i
    return images, image_masks


def estimate_prompt_tokens(
    num_retrieved: int,
    action_steps: int,
    state_dim: int = 32,
    action_dim: int = 32,
    base_prompt_tokens: int = 60,
) -> int:
    """Rough ballpark of prompt tokens so configs can size ``max_token_len``.

    NOTE: a loose under-estimate, not an upper bound — it assumes ~1.3 tokens per
    discretized number, but 3-digit bin ids cost ~3-4 PaliGemma tokens each, so the
    real worst case runs ~2.5x higher (k=4 @ 32/32 dims: this returns ~416 vs ~1147
    measured). Used only for a soft warning in the algo; for the authoritative
    check tokenize the real worst-case prompt with
    ``egomimic/ricl/scripts/check_prompt_budget.py``.
    """
    per_demo_numbers = state_dim + action_steps * action_dim
    per_demo_tokens = int(1.3 * per_demo_numbers) + 6  # + label/separators
    return base_prompt_tokens + num_retrieved * per_demo_tokens
