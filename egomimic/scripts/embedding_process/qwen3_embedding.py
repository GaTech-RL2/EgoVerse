"""
Qwen3-Embedding-0.6B text-embedding ZarrKeyTransform.

Each input key is a JSON-encoded annotations array of records
{"text": str, "start_idx": int, "end_idx": int}. The corresponding output key
is a (T, D) float32 array where every frame in [start_idx, end_idx) receives
the embedding of that span's text. Frames not covered by any annotation are
zero-filled. Overlaps: later-listed spans win.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import torch
import zarr

from egomimic.scripts.embedding_process.zarr_key_transform import ZarrKeyTransform

logger = logging.getLogger(__name__)


def _decode_json_entry(value):
    if isinstance(value, np.void):
        value = value.item()
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if isinstance(value, bytes):
        return json.loads(value.decode("utf-8"))
    if isinstance(value, str):
        return json.loads(value)
    return value


def _last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Last-token pooling (Qwen3 official recipe), robust to left/right padding."""
    left_padded = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padded:
        return last_hidden_states[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(
        last_hidden_states.size(0), device=last_hidden_states.device
    )
    return last_hidden_states[batch_idx, seq_lens]


class Qwen3TextEmbedding(ZarrKeyTransform):
    """
    Per-frame Qwen3-Embedding-0.6B text embeddings, expanded from JSON
    annotation spans.

    Default output naming swaps the leading dotted segment of the input key
    for ``"qwen"`` (e.g. ``annotations`` -> ``qwen.annotations``). Pass
    ``output_keys`` explicitly to override.
    """

    DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    OUTPUT_PREFIX = "qwen"

    def __init__(
        self,
        input_keys: list[str],
        output_keys: list[str] | None = None,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 16,
        max_length: int = 512,
        overwrite: bool = False,
        device: str | torch.device = "cuda",
        chunk_timesteps: int = 100,
        dtype: torch.dtype = torch.float16,
    ):
        if output_keys is not None and len(input_keys) != len(output_keys):
            raise ValueError(
                "Qwen3TextEmbedding requires len(input_keys) == len(output_keys); "
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
        self.max_length = max_length
        self.dtype = dtype
        self.tokenizer = None
        self.model = None
        self._embed_dim: int | None = None

    def setup(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading Qwen3 embedding model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left"
        )
        self.model = AutoModel.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.model.to(self.device).eval()
        self._embed_dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._embed_dim or 0), dtype=np.float32)
        all_embs: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**tokens)
            pooled = _last_token_pool(
                outputs.last_hidden_state, tokens["attention_mask"]
            )
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            all_embs.append(pooled.cpu().numpy())
        return np.concatenate(all_embs, axis=0).astype(np.float32)

    def compute(self, store: zarr.Group) -> dict[str, np.ndarray]:
        total_frames = int(store.attrs["total_frames"])
        outputs: dict[str, np.ndarray] = {}
        for in_key, out_key in zip(self.input_keys, self.output_keys):
            raw = store[in_key][:]
            entries = [_decode_json_entry(x) for x in raw]
            entries = [e for e in entries if isinstance(e, dict)]
            texts = [e.get("text", "") for e in entries]
            embs = self._embed_texts(texts)

            assert self._embed_dim is not None
            per_frame = np.zeros((total_frames, self._embed_dim), dtype=np.float32)
            for emb, entry in zip(embs, entries):
                start = max(0, int(entry.get("start_idx", 0)))
                end = min(total_frames, int(entry.get("end_idx", 0)))
                if end > start:
                    per_frame[start:end] = emb
            outputs[out_key] = per_frame
        return outputs
