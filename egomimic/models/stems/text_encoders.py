"""Frozen text-encoder STEM modules for HPT language conditioning.

Role home for the Qwen3-Embedding stems that turn a batch of raw prompt
strings into cross-attention tokens. Ported here from the pre-batchflow
``models/hpt_nets.py`` (class bodies behaviour-identical); the
:class:`PolicyStem` lives in ``hpt_stems`` after the hydra refactor and in
``hpt_nets`` on older branches, so the import below accepts both.

Unlike every other stem, these consume ``list[str]`` rather than a tensor —
the stem owns tokenization, so ``HPTModel.stem_process`` routes list-valued
modalities straight to ``compute_latent`` with no positional embedding or
horizon masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from egomimic.models.stems.hpt_stems import PolicyStem
except ImportError:  # pre-refactor layout (hpt_nets.PolicyStem)
    from egomimic.models.hpt_nets import PolicyStem


def _qwen_last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Last-token pooling per the official Qwen3-Embedding recipe.

    Handles both left- and right-padding. For each row we read the hidden state
    at the position of the final non-padded token.
    """
    left_padded = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padded:
        return last_hidden_states[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(
        last_hidden_states.size(0), device=last_hidden_states.device
    )
    return last_hidden_states[batch_idx, seq_lens]


class _Qwen3BaseEncoder(PolicyStem):
    """Shared base for Qwen3-Embedding stems used by HPT.

    Owns the tokenizer + transformer encoder so HPT.process_batch_for_training
    only needs to pass a list of raw prompt strings. Subclasses pick whether
    the feature passed to cross-attention is a pooled (B, 1, D) summary or the
    full per-token (B, L, D) sequence.

    Args:
        model_name: HF identifier for the Qwen3-Embedding checkpoint.
        max_length: tokenizer truncation length.
        freeze: if True (default), freeze the encoder weights and run in
            eval mode under no_grad. If False, the encoder is trainable.
        dtype: weight dtype for the HF model (fp16 by default to keep VRAM
            cost down when frozen).
        output_dim: projects Qwen's hidden_size down to the cross-attn
            modality_embed_dim; must match the trunk's embed_dim.
    """

    DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = 128,
        freeze: bool = True,
        dtype: str = "float16",
        output_dim: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        self.freeze_encoder = freeze
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
        self.hidden_size = int(self.encoder.config.hidden_size)
        # Project Qwen's hidden_size (typ. 1024) down to the cross-attn
        # modality_embed_dim. Must match the trunk's embed_dim so the post-stem
        # token tensors concat with other modalities' tokens in
        # ``HPTModel.preprocess_tokens``.
        self.output_dim = output_dim if output_dim is not None else self.hidden_size
        self.proj = (
            nn.Linear(self.hidden_size, self.output_dim)
            if self.output_dim != self.hidden_size
            else nn.Identity()
        )

    def _encode(self, prompts):
        tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        if self.freeze_encoder:
            with torch.no_grad():
                out = self.encoder(**tokens)
        else:
            out = self.encoder(**tokens)
        return out.last_hidden_state, tokens["attention_mask"]

    def train(self, mode: bool = True):
        """Keep the frozen encoder in eval mode regardless of outer train flag."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self


class QwenPooledEncoder(_Qwen3BaseEncoder):
    """Qwen3-Embedding stem with last-token pooling -> (B, 1, output_dim)."""

    def forward(self, prompts):
        hidden, mask = self._encode(prompts)
        pooled = _qwen_last_token_pool(hidden, mask)
        pooled = F.normalize(pooled.float(), p=2, dim=1)
        pooled = self.proj(pooled)
        return pooled.unsqueeze(1)  # (B, 1, output_dim)

    def compute_latent(self, prompts):
        feat = self(prompts)  # (B, 1, output_dim)
        stem_tokens = self.tokens.repeat(feat.shape[0], 1, 1)
        return self.cross_attention(stem_tokens, feat)


class QwenPerTokenEncoder(_Qwen3BaseEncoder):
    """Qwen3-Embedding stem returning per-token hidden states (B, L, output_dim).

    Padding positions are zeroed before cross-attention so they don't pull
    information from the learnable stem tokens.
    """

    def forward(self, prompts):
        hidden, mask = self._encode(prompts)
        feat = hidden.float() * mask.unsqueeze(-1).float()
        return self.proj(feat)  # (B, L, output_dim)

    def compute_latent(self, prompts):
        feat = self(prompts)  # (B, L, output_dim)
        stem_tokens = self.tokens.repeat(feat.shape[0], 1, 1)
        return self.cross_attention(stem_tokens, feat)
