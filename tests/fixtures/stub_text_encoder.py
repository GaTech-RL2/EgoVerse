"""A ``QwenPerTokenEncoder`` whose tokenizer + HF model are replaced by a
deterministic per-prompt table, so tests never load Qwen3-Embedding-0.6B.

Mirrors ``CountingStubEncoder`` in tests/unit/test_text_feature_cache.py, but
accepts any prompt string (the train-step tier does not know the fixtures'
annotations up front) and swallows the real encoder's kwargs so a config can
swap it in by ``_target_`` alone.
"""

from __future__ import annotations

import zlib

import torch
import torch.nn as nn

from egomimic.models.hpt_nets import PolicyStem, QwenPerTokenEncoder


class StubTextEncoder(QwenPerTokenEncoder):
    def __init__(
        self,
        output_dim: int = 32,
        hidden_size: int = 16,
        tokens_per_prompt: int = 2,
        **kwargs,  # model_name / dtype / freeze / cache_* are accepted and ignored
    ) -> None:
        PolicyStem.__init__(self, specs=None)
        self.hidden_size = int(hidden_size)
        self.output_dim = int(output_dim)
        self.tokens_per_prompt = int(tokens_per_prompt)
        self.lengths: dict[str, int] = {}
        self.freeze_encoder = True
        self._snapshot_dir = ""
        self._load_dtype = torch.float32
        self.proj = nn.Linear(self.hidden_size, self.output_dim)
        self.encoder = nn.Identity()  # never called; train() keeps it in eval
        self._init_text_feature_cache(True, 128)

    def _row(self, prompt: str) -> torch.Tensor:
        # crc32, not hash(): str hashing is salted per process.
        generator = torch.Generator().manual_seed(zlib.crc32(prompt.encode()))
        length = self.lengths.get(prompt, self.tokens_per_prompt)
        return torch.randn(length, self.hidden_size, generator=generator)

    def _encode_online(self, prompts):
        rows = [self._row(prompt) for prompt in prompts]
        longest = max(row.shape[0] for row in rows)
        hidden = torch.zeros(len(rows), longest, self.hidden_size)
        mask = torch.zeros(len(rows), longest, dtype=torch.long)
        for i, row in enumerate(rows):  # left padding, as padding_side="left"
            hidden[i, longest - row.shape[0] :] = row
            mask[i, longest - row.shape[0] :] = 1
        return hidden, mask
