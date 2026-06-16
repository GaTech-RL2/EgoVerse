"""
DINOv3 image-embedding ZarrKeyTransform.

Each input key must be a JPEG-encoded image array (as produced by ZarrWriter).
For each input key, produces a per-frame patch-token embedding of shape
``(T, N, D)`` float32 under the correspondingly-positioned output key,
where ``N`` is the number of patch tokens (CLS and any register tokens are
dropped) and ``D`` is the model hidden dim.
"""

from __future__ import annotations

import logging

import numpy as np
import simplejpeg
import torch
import zarr

from egomimic.scripts.embedding_process.zarr_key_transform import ZarrKeyTransform

logger = logging.getLogger(__name__)


class DINOv3ImageEmbedding(ZarrKeyTransform):
    """
    Per-frame DINOv3 image embeddings.

    Default output naming swaps the leading dotted segment of the input key
    for ``"dino"`` (e.g. ``images.front_1`` -> ``dino.front_1``). Pass
    ``output_keys`` explicitly to override.
    """

    DEFAULT_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    OUTPUT_PREFIX = "dino"

    def __init__(
        self,
        input_keys: list[str],
        output_keys: list[str] | None = None,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 64,
        overwrite: bool = False,
        device: str | torch.device = "cuda",
        chunk_timesteps: int = 100,
        dtype: torch.dtype = torch.float16,
    ):
        if output_keys is not None and len(input_keys) != len(output_keys):
            raise ValueError(
                "DINOv3ImageEmbedding requires len(input_keys) == len(output_keys); "
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
        self.processor = None
        self.model = None

    def setup(self) -> None:
        from transformers import AutoImageProcessor, AutoModel

        logger.info("Loading DINOv3 model: %s", self.model_name)
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.model.to(self.device).eval()

        cfg = self.model.config
        side = int(cfg.image_size) // int(cfg.patch_size)
        self.num_patches = side * side
        logger.info(
            "DINOv3 patch grid: %d x %d = %d tokens (hidden_dim=%d)",
            side,
            side,
            self.num_patches,
            int(cfg.hidden_size),
        )

    @torch.no_grad()
    def _embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        outputs = self.model(pixel_values=pixel_values)
        # last_hidden_state ends with the patch tokens; CLS + any register
        # tokens sit at the start. Slice the trailing num_patches tokens to
        # get just the patch grid in row-major order.
        patches = outputs.last_hidden_state[:, -self.num_patches :]
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
                images = [
                    simplejpeg.decode_jpeg(bytes(b), colorspace="RGB")
                    for b in jpeg_batch
                ]
                chunks.append(self._embed_batch(images))
            outputs[out_key] = np.concatenate(chunks, axis=0).astype(np.float32)
        return outputs
