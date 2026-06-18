"""
DINOv2 image-embedding ZarrKeyTransform.

Each input key must be a JPEG-encoded image array (as produced by ZarrWriter).
For each input key, produces a per-frame patch-token embedding of shape
``(T, 256, 768)`` float32 under the correspondingly-positioned output key
(16x16 patch grid for a 224x224 input at patch size 14).

Replicates the RICL reference encoder
(``ricl_openpi/src/openpi/policies/utils.py``: ``load_dinov2`` /
``process_dinov2`` / ``embed``): torch.hub ``dinov2_vitb14``, resize-with-pad
to 224 (PIL bilinear, black padding), ImageNet mean/std normalization, then
``forward_features()['x_norm_patchtokens']``. Pooling into the retrieval
descriptor happens downstream (``egomimic.ricl.retrieval.pool_patch_embeddings``),
so the stored tokens stay pooling-agnostic.
"""

from __future__ import annotations

import logging

import numpy as np
import simplejpeg
import torch
import zarr

from egomimic.scripts.embedding_process.zarr_key_transform import ZarrKeyTransform

logger = logging.getLogger(__name__)

# ricl_openpi process_dinov2 constants
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
DINOV2_IMAGE_SIZE = 224


def preprocess_dinov2(images: np.ndarray) -> torch.Tensor:
    """uint8 ``(B, H, W, 3)`` RGB -> float32 ``(B, 3, 224, 224)``, replicating
    ricl_openpi's ``process_dinov2``: resize-with-pad to 224 (PIL bilinear,
    black padding), scale to [0, 1], ImageNet mean/std normalize.
    """
    # the exact PIL-based function ricl_openpi uses (openpi_client is shipped
    # with the sibling openpi install)
    from openpi_client.image_tools import resize_with_pad

    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(
            f"expected uint8 (B, H, W, 3) RGB images, got {images.dtype} {images.shape}"
        )
    images = resize_with_pad(images, DINOV2_IMAGE_SIZE, DINOV2_IMAGE_SIZE)
    x = torch.from_numpy(images.transpose(0, 3, 1, 2)).float() / 255.0
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


class DINOv2ImageEmbedding(ZarrKeyTransform):
    """
    Per-frame DINOv2 patch-token embeddings (RICL reference encoder).

    Default output naming swaps the leading dotted segment of the input key
    for ``"dinov2"`` (e.g. ``images.front_1`` -> ``dinov2.front_1``). Pass
    ``output_keys`` explicitly to override.
    """

    DEFAULT_MODEL = "dinov2_vitb14"  # torch.hub name under facebookresearch/dinov2
    OUTPUT_PREFIX = "dinov2"

    def __init__(
        self,
        input_keys: list[str],
        output_keys: list[str] | None = None,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 64,
        overwrite: bool = False,
        device: str | torch.device = "cuda",
        chunk_timesteps: int = 100,
        dtype: torch.dtype = torch.float32,
    ):
        if output_keys is not None and len(input_keys) != len(output_keys):
            raise ValueError(
                "DINOv2ImageEmbedding requires len(input_keys) == len(output_keys); "
                f"got {len(input_keys)} vs {len(output_keys)}"
            )
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            batch_size=batch_size,
            overwrite=overwrite,
            device=device,
            chunk_timesteps=chunk_timesteps,
        )
        self.model_name = model_name
        self.dtype = dtype
        self.model = None

    def setup(self) -> None:
        logger.info(
            "Loading DINOv2 via torch.hub: facebookresearch/dinov2 %s", self.model_name
        )
        self.model = torch.hub.load("facebookresearch/dinov2", self.model_name)
        self.model.to(self.device, dtype=self.dtype).eval()

    @torch.no_grad()
    def _embed_batch(self, images: np.ndarray) -> np.ndarray:
        pixel_values = preprocess_dinov2(images).to(self.device, dtype=self.dtype)
        feats = self.model.forward_features(pixel_values)
        patches = feats["x_norm_patchtokens"]  # (B, 256, 768)
        return patches.float().cpu().numpy()

    def compute(self, store: zarr.Group) -> dict[str, np.ndarray]:
        outputs: dict[str, np.ndarray] = {}
        for in_key, out_key in zip(self.input_keys, self.output_keys):
            arr = store[in_key]
            n = arr.shape[0]
            total_frames = int(store.attrs.get("total_frames", n))
            n = min(n, total_frames)

            chunks: list[np.ndarray] = []
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                jpeg_batch = arr[start:end]
                images = np.stack(
                    [
                        simplejpeg.decode_jpeg(bytes(b), colorspace="RGB")
                        for b in jpeg_batch
                    ]
                )
                chunks.append(self._embed_batch(images))
            outputs[out_key] = np.concatenate(chunks, axis=0).astype(np.float32)
        return outputs
