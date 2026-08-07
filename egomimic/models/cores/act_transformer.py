"""ACT transformer CORE + style readout relocated from ``models/act_nets.py``.

Role home for the ACT trunk: the sinusoidal :class:`PositionalEncoding`, the
encoder/decoder :class:`Transformer`, and the CVAE :class:`StyleEncoder` latent
readout used by the ACT algo. Moved here verbatim (class bodies byte-identical)
in the models/ hierarchy pass; only the import header changed.
"""

import math
from typing import Any, Callable, Optional

import torch
from torch import nn
from torch.distributions import Normal


class PositionalEncoding(nn.Module):
    """
    Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
    """

    def __init__(self, embed_dim):
        """
        Standard sinusoidal positional encoding scheme in transformers.

        Positional encoding of the k'th position in the sequence is given by:
            p(k, 2i) = sin(k/n^(i/d))
            p(k, 2i+1) = sin(k/n^(i/d))

        n: set to 10K in original Transformer paper
        d: the embedding dimension
        i: positions along the projected embedding space (ranges from 0 to d/2)

        Args:
            embed_dim: The number of dimensions to project the timesteps into.
        """
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        Input timestep of shape BxT
        """
        position = x

        # computing 1/n^(i/d) in log space and then exponentiating and fixing the shape
        div_term = (
            torch.exp(
                torch.arange(0, self.embed_dim, 2, device=x.device)
                * (-math.log(10000.0) / self.embed_dim)
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(x.shape[0], x.shape[1], 1)
        )
        pe = torch.zeros((x.shape[0], x.shape[1], self.embed_dim), device=x.device)
        pe[:, :, 0::2] = torch.sin(position.unsqueeze(-1) * div_term)
        pe[:, :, 1::2] = torch.cos(position.unsqueeze(-1) * div_term)
        return pe.detach()


class Transformer(nn.Module):
    """
    Basic transformer implementation using torch.nn. Also added option for custom pos embeddings.
    Made to be as basic as possible but also flexible to be put into ACT.

        d: hidden dimension
        h: number of heads
        d_ff: feed forward dimension
        num_layers: number of layers for encoder and decoder
        L: sequence length
        dropout: dropout rate
        src_vocab_size: size of source vocabulary
        tgt_vocab_size: size of target vocabulary
        pos_encoding_class : nn.Module class defining custom pos encoding

    """

    def __init__(
        self,
        d: int,
        h: int,
        d_ff: int,
        num_layers: int,
        dropout: float = 0.1,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        pos_encoding_class: Optional[Callable[..., nn.Module]] = None,
        **pos_encoding_kwargs: Any,  # Additional arguments for the custom encoding class
    ):
        super(Transformer, self).__init__()

        self.d = d
        self.h = h
        self.src_embed = nn.Embedding(src_vocab_size, d) if src_vocab_size else None
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d) if tgt_vocab_size else None

        if pos_encoding_class is not None:
            self.src_pos_encoding = pos_encoding_class(**pos_encoding_kwargs)
            self.tgt_pos_encoding = pos_encoding_class(**pos_encoding_kwargs)
        else:
            self.src_pos_encoding = PositionalEncoding(d)
            self.tgt_pos_encoding = PositionalEncoding(d)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=h, dim_feedforward=d_ff, dropout=dropout, batch_first=True
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=h, dim_feedforward=d_ff, dropout=dropout, batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d, tgt_vocab_size) if tgt_vocab_size else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, auto_masks=False):
        assert not auto_masks, "Auto mask not supported"
        src_mask = tgt_mask = None

        if self.src_embed:
            src = self.src_embed(src)
        if self.tgt_embed:
            tgt = self.tgt_embed(tgt)

        src = src.transpose(0, 1)  # [sequence_length, batch_size, hidden_dim]
        src_position_indices = (
            torch.arange(src.size(0), device=src.device)
            .unsqueeze(1)
            .expand(-1, src.size(1))
        )  # [sequence_length, batch_size]
        src_pos = self.src_pos_encoding(
            src_position_indices
        )  # [sequence_length, batch_size, hidden_dim]
        src = src + src_pos
        src = src.transpose(0, 1)

        tgt = tgt.transpose(0, 1)  # [T, B, hidden_dim]
        tgt_position_indices = (
            torch.arange(tgt.size(0), device=tgt.device)
            .unsqueeze(1)
            .expand(-1, tgt.size(1))
        )  # [sequence_length, batch_size]
        tgt_pos = self.tgt_pos_encoding(tgt_position_indices)
        tgt = tgt + tgt_pos
        tgt = tgt.transpose(0, 1)  # [B, T, hidden_dim]

        src = self.dropout(src)
        tgt = self.dropout(tgt)

        enc = self.encoder(src, src_key_padding_mask=src_mask)
        dec = self.decoder(
            tgt, enc, tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=src_mask
        )

        if self.fc:
            dec = self.fc(dec)

        return dec


class StyleEncoder(nn.Module):
    def __init__(
        self,
        act_len: int,
        hidden_dim: int,
        latent_dim: int,
        h: int,
        d_ff: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super(StyleEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.act_len = act_len
        self.hidden_dim = hidden_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=h, dim_feedforward=d_ff, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_embedding = nn.Parameter(torch.rand(1, hidden_dim))
        self.latent_projection = nn.Linear(hidden_dim, latent_dim * 2)

        self.pos_encoding = PositionalEncoding(hidden_dim)

        # self.encoder_norm = nn.LayerNorm(hidden_dim)

    def forward(self, qpos, actions):
        """
        qpos: linear projection of qpos
        actions: linear projection of actions
        """
        bsz = qpos.shape[0]
        qpos = qpos.unsqueeze(1)  # [bsz, 1, hidden_dim]

        cls = self.cls_embedding.unsqueeze(0).expand(
            bsz, -1, -1
        )  # [bsz, 1, hidden_dim]

        x = torch.cat([cls, qpos, actions], dim=1)  # [bsz, act_len + 2, hidden_dim]
        assert x.shape == (bsz, self.act_len + 2, self.hidden_dim)

        pos_indices = (
            torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(bsz, -1)
        )  # [bsz, act_len + 2]
        pos_embedded = self.pos_encoding(pos_indices)  # [bsz, act_len + 2, hidden_dim]

        x = x + pos_embedded

        x = x.transpose(0, 1)  # [act_len + 2, bsz, hidden_dim]

        x = self.encoder(x)  # [act_len + 2, bsz, hidden_dim]

        # x = self.encoder_norm(x) # cuz act has weird pre and post norms

        x = x[0]  # [bsz, hidden_dim]

        x = self.latent_projection(x)  # [bsz, latent_dim * 2]
        mu = x[:, self.latent_dim :]
        logvar = x[:, : self.latent_dim]
        dist = Normal(mu, (logvar / 2).exp())  # Create Normal distribution

        return dist
