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

def _concat_proprio(batch: dict, proprio_keys: list[str], device: torch.device) -> torch.Tensor:
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

def _fill_missing_images(inputs: dict, required_keys: list[str], device: torch.device, default_hw=(224, 224)):
    """
    Ensures all required image keys exist in `inputs`.
    - Duplicates an existing image tensor if a required key is missing.
    - Infers shape (B, C, H, W) from the first present image; if none found, uses default_hw.
    
    Args:
        inputs (dict): Dictionary of available tensors (e.g., batch from dataloader).
        required_keys (list[str]): Keys that must exist in the returned dict.
        device (torch.device): Target device for all tensors.
        default_hw (tuple[int, int], optional): (H, W) to use if no valid image found. Default = (224, 224).

    Returns:
        dict[str, torch.Tensor]: Dictionary containing all required keys,
                                 duplicating existing images where inputs were missing.
    """
    images = {}
    B = None
    C = 3
    H, W = default_hw

    # Find the first valid image among the required keys to duplicate
    seed_img = None
    for k in required_keys:
        if k in inputs:
            img = inputs[k].to(device)
            if img.ndim == 4:
                seed_img = _ensure_bchw(img)
                B, C, H, W = seed_img.shape
                break

    if seed_img is None:
        raise ValueError("Cannot duplicate images; no valid image tensor found among required keys.")

    # Fill or copy each key
    for k in required_keys:
        if k in inputs:
            images[k] = _ensure_bchw(inputs[k].to(device))
        else:
            # Duplicate the seed image for missing keys
            images[k] = seed_img.clone()

    return images