import torch


class _SimpleObservation:
    """Minimal container matching the structure expected by preprocess_observation_pytorch."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _ensure_bchw(t: torch.Tensor) -> torch.Tensor:
    """Accept [B, C, H, W] or [B, H, W, C]; return [B, C, H, W]."""
    if t.ndim != 4:
        raise ValueError(f"Expected 4D tensor for image, got {t.shape}")
    if t.shape[1] in (1, 3):
        return t
    elif t.shape[-1] in (1, 3):
        return t.permute(0, 3, 1, 2).contiguous()
    return t


def _bhwc(t_bchw: torch.Tensor) -> torch.Tensor:
    """Convert [B, C, H, W] -> [B, H, W, C]."""
    return t_bchw.permute(0, 2, 3, 1).contiguous()


def _to_minus1_1(img: torch.Tensor) -> torch.Tensor:
    """Convert uint8 [0,255] or float [0,1] → float [-1,1]."""
    if img.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        img = img.to(torch.float32) / 255.0
    else:
        img = img.to(torch.float32)
        img = torch.clamp(img, 0.0, 1.0)
    return img * 2.0 - 1.0


def _mask_from_batch(B: int, device) -> torch.Tensor:
    """Default per-image mask (all True)."""
    return torch.ones(B, dtype=torch.bool, device=device)


def _concat_proprio(
    batch: dict, proprio_keys: list[str], device: torch.device
) -> torch.Tensor:
    """Concat all proprio tensors along last dim → [B, D] (D can be 0)."""
    parts = []
    for k in proprio_keys:
        if k in batch:
            parts.append(batch[k].to(device))
    if not parts:
        # If no proprio, infer B from any tensor in batch (best-effort), else 0
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.ndim >= 1:
                B = v.shape[0]
                return torch.zeros(B, 0, device=device)
        return torch.zeros(0, 0, device=device)
    return torch.cat(parts, dim=-1)


def _empty_lang_placeholders(B: int, device: torch.device):
    """Empty language tensors with correct batch dim."""
    L = 0
    tok = torch.zeros(B, L, dtype=torch.long, device=device)
    mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    return tok, mask, mask.clone(), mask.clone()


def _mask_from_batch(B: int, device: torch.device) -> torch.Tensor:
    """Per-image mask [B] set to True."""
    return torch.ones(B, dtype=torch.bool, device=device)


# openpi's fixed camera tuple -> the dataset key every embodiment's keymap
# emits (Embodiment.VIZ_IMAGE_KEY for the front camera). Datasets use one
# naming for every algo; the Pi wrapper does the renaming here.
PI_CAMERA_SLOTS: dict[str, str] = {
    "base_0_rgb": "observations.images.front_img_1",
    "left_wrist_0_rgb": "observations.images.left_wrist_img",
    "right_wrist_0_rgb": "observations.images.right_wrist_img",
}


def _image_or_none(batch: dict, key: str) -> torch.Tensor | None:
    v = batch.get(key)
    if isinstance(v, torch.Tensor) and v.ndim == 4:
        return v
    return None


def gather_pi_images(
    batch: dict, slot_map: dict[str, str], device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, bool]]:
    """Pull openpi's camera slots out of a dataset batch.

    For each slot the dataset key from ``slot_map`` is used; the slot name
    itself is accepted as a fallback so robot rollout (which emits both names)
    and checkpoints trained before the remap keep working. Slots with no source
    image are filled with a copy of the first present one and reported absent in
    the returned flags so the caller can mask them out.

    Returns (images BCHW keyed by slot, present-flag keyed by slot).
    """
    images: dict[str, torch.Tensor] = {}
    present: dict[str, bool] = {}
    for slot, dataset_key in slot_map.items():
        img = _image_or_none(batch, dataset_key)
        if img is None:
            img = _image_or_none(batch, slot)
        present[slot] = img is not None
        if img is not None:
            images[slot] = _ensure_bchw(img.to(device))
    if not images:
        raise ValueError(
            "No camera image in batch for any Pi slot; looked for "
            + ", ".join(f"{s} <- {k}" for s, k in slot_map.items())
        )
    seed = next(iter(images.values()))
    return {
        slot: images[slot] if slot in images else seed.clone() for slot in slot_map
    }, present
