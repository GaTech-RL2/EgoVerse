"""Text / language-prompt encoders for HPT conditioning.

Ported from EgoVerse2 ``models/hpt_nets.py`` into gmm's ``models/stems``
package. These stems consume a **list of raw prompt strings** (one per batch
item) and own their own tokenizer, so ``HPT.process_batch_for_training`` only
needs to hand over the sampled annotation strings. They subclass the gmm
:class:`PolicyStem` (``egomimic.models.stems.hpt_stems.PolicyStem``) so the
learnable-token cross-attention pooling and the ``init_cross_attn`` /
``compute_latent`` contract are shared with the image/proprio stems.

Two Qwen3-Embedding variants:
  * :class:`QwenPooledEncoder`  -> one pooled sentence token, ``(B, 1, D)``.
  * :class:`QwenPerTokenEncoder` -> per-token hidden states, ``(B, L, D)``.
Plus a pre-tokenized :class:`T5Encoder` (+ :class:`T5TokenizerWrapper`).

``transformers`` is imported lazily inside each class so importing the stems
package stays cheap and does not hard-require HuggingFace for non-text models.
"""

from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.stems.hpt_stems import PolicyStem


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
    batch_idx = torch.arange(last_hidden_states.size(0), device=last_hidden_states.device)
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
        output_dim: project Qwen's hidden_size down to the cross-attn
            modality_embed_dim; None keeps the native hidden_size.
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
    """Qwen3-Embedding stem with last-token pooling -> (B, 1, hidden_size)."""

    def forward(self, prompts):
        hidden, mask = self._encode(prompts)
        pooled = _qwen_last_token_pool(hidden, mask)
        pooled = F.normalize(pooled.float(), p=2, dim=1)
        pooled = self.proj(pooled)
        return pooled.unsqueeze(1)  # (B, 1, output_dim)

    def compute_latent(self, prompts):
        feat = self(prompts)  # (B, 1, hidden_size)
        stem_tokens = self.tokens.repeat(feat.shape[0], 1, 1)
        return self.cross_attention(stem_tokens, feat)


class QwenPerTokenEncoder(_Qwen3BaseEncoder):
    """Qwen3-Embedding stem returning per-token hidden states (B, L, hidden_size).

    Padding positions are zeroed before cross-attention so they don't pull
    information from the learnable stem tokens.
    """

    def forward(self, prompts):
        hidden, mask = self._encode(prompts)
        feat = hidden.float() * mask.unsqueeze(-1).float()
        return self.proj(feat)  # (B, L, output_dim)

    def compute_latent(self, prompts):
        feat = self(prompts)  # (B, L, hidden_size)
        stem_tokens = self.tokens.repeat(feat.shape[0], 1, 1)
        return self.cross_attention(stem_tokens, feat)


class T5Encoder(PolicyStem):
    def __init__(self, per_token=True, **kwargs) -> None:
        """T5 Encoder that expects pre-tokenized inputs

        Args:
            per_token (bool): If True, return per-token embeddings. If False, return mean-pooled embeddings
        """
        super().__init__(**kwargs)
        from transformers import T5Model

        self.per_token = per_token
        self.encoder = T5Model.from_pretrained("t5-base").encoder

    def forward(self, tokenized_input: dict) -> torch.Tensor:
        """
        Args:
            tokenized_input: Dictionary containing:
                - input_ids: torch.Tensor [B, 1, L]
                - attention_mask: torch.Tensor [B, 1, L]
                (other fields will be ignored)
        Returns:
            torch.Tensor: Encoded representations
                if per_token=True: [B, L, hidden_size]
                if per_token=False: [B, hidden_size]
        """
        tokenized_input = {k: v.squeeze(1).long() for k, v in tokenized_input.items()}
        output = self.encoder(
            input_ids=tokenized_input["lang_input_ids"].to(self.device),
            attention_mask=tokenized_input["lang_attention_mask"].to(self.device),
            return_dict=True,
        )

        torch.cuda.empty_cache()  # empty cache to save memory
        if self.per_token:
            return output.last_hidden_state[:, 0].detach().unsqueeze(1)
        else:
            emb = output.last_hidden_state.mean(dim=1).detach().unsqueeze(1)
            return emb


class T5TokenizerWrapper:
    """Wrapper class for T5Tokenizer to prepare inputs for T5Encoder"""

    def __init__(self, model_name: str = "t5-base", max_length: int = 512):
        """
        Initialize T5 tokenizer wrapper.

        Args:
            model_name (str): Name of the T5 model to use for tokenization
            max_length (int): Maximum sequence length for tokenization
        """
        from transformers import T5Tokenizer

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __call__(self, text: Union[str, List[str]]) -> dict:
        """
        Tokenize input text(s) and prepare for T5Encoder.

        Args:
            text: Either a single string or list of strings to tokenize

        Returns:
            dict: Dictionary containing:
                - input_ids: torch.Tensor [B, L]
                - attention_mask: torch.Tensor [B, L]
        """
        # Handle single string input
        if isinstance(text, str):
            text = [text]

        # Tokenize with padding and truncation
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }


class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=-1)
