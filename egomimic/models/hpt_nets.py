import hashlib
import json
import os
from functools import partial
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torchvision
from einops import rearrange, repeat
from termcolor import cprint
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import VisionTransformer
from torch import einsum
from torchvision import transforms
from transformers import T5Model, T5Tokenizer

from egomimic.utils.tensor_utils import get_sinusoid_encoding_table

# --------------------------------------------------------------------------
# Pretrained-weight protection
#
# ``HPTModel.finalize_modules`` runs a xavier pass (``_init_weights``) over the
# whole model once the stems / heads are ModuleDicts. That pass used to visit
# every ``nn.Linear`` in every stem, including the HF text encoders loaded from
# a pretrained snapshot, silently randomising them. Modules that own pretrained
# weights declare them here; ``apply_skipping_pretrained`` then walks the tree
# without ever entering such a subtree, so any future pretrained stem (e.g. a
# Qwen VLM stem) is protected without touching ``finalize_modules`` again.
# --------------------------------------------------------------------------


class PretrainedWeights:
    """Mixin marking a module whose weights come from a pretrained checkpoint.

    Subclasses list the attribute names holding the pretrained submodules in
    ``_hpt_pretrained_attrs``. Two things follow:

    * ``finalize_modules``'s init pass skips those subtrees entirely.
    * if the subclass can produce a reference state dict for the checkpoint
      (``pretrained_reference_state_dict``), the weights are hashed and
      compared with that reference once the model is fully built.
    """

    _hpt_pretrained = True
    _hpt_pretrained_attrs: tuple = ()

    def pretrained_submodules(self) -> List[nn.Module]:
        """The submodules whose weights must never be re-initialised."""
        mods = []
        for name in self._hpt_pretrained_attrs:
            mod = getattr(self, name, None)
            if isinstance(mod, nn.Module):
                mods.append(mod)
        return mods

    def pretrained_state_dict(self) -> dict:
        """Flat ``{"<attr>.<param>": tensor}`` view of the pretrained weights."""
        state = {}
        for name in self._hpt_pretrained_attrs:
            mod = getattr(self, name, None)
            if isinstance(mod, nn.Module):
                for key, value in mod.state_dict().items():
                    state[f"{name}.{key}"] = value
        return state

    def pretrained_reference_state_dict(self) -> Optional[dict]:
        """The checkpoint's own weights, read back independently of this model.

        ``None`` (the default) means "cannot be verified offline"; the checker
        then skips this module instead of failing.
        """
        return None

    def pretrained_hash_dtype(self) -> Optional[torch.dtype]:
        """dtype both sides are cast to before hashing (``None`` = as stored)."""
        return None


def pretrained_module_ids(root: nn.Module) -> set:
    """ids of every module living under a declared pretrained submodule."""
    skip = set()
    for module in root.modules():
        get = getattr(module, "pretrained_submodules", None)
        if not callable(get):
            continue
        for sub in get():
            skip.update(id(m) for m in sub.modules())
    return skip


def apply_skipping_pretrained(root: nn.Module, fn: Callable) -> None:
    """``root.apply(fn)`` that never enters a pretrained submodule.

    Same post-order traversal (and therefore the same RNG consumption) as
    ``nn.Module.apply`` for a tree with no pretrained submodules.
    """
    skip = pretrained_module_ids(root)

    def _walk(module: nn.Module) -> None:
        if id(module) in skip:
            return
        for child in module.children():
            _walk(child)
        fn(module)

    _walk(root)


def hash_state_dict(state: dict, dtype: Optional[torch.dtype] = None) -> str:
    """Deterministic sha256 over parameter names + raw bytes, in name order."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().to("cpu")
        if dtype is not None:
            tensor = tensor.to(dtype)
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.flatten().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_pretrained_weights(root: nn.Module, verbose: bool = True) -> dict:
    """Hash every declared pretrained submodule and compare with its checkpoint.

    Raises ``RuntimeError`` naming the module and both hashes on a mismatch.
    Returns ``{module_name: sha256}`` for the modules that could be checked.
    """
    checked = {}
    for name, module in root.named_modules():
        get_ref = getattr(module, "pretrained_reference_state_dict", None)
        if not callable(get_ref):
            continue
        reference = get_ref()
        if reference is None:
            continue
        dtype = module.pretrained_hash_dtype()
        current = module.pretrained_state_dict()
        if set(current) != set(reference):
            raise RuntimeError(
                f"pretrained weight check failed for '{name or root.__class__.__name__}'"
                f" ({module.__class__.__name__}): the checkpoint and the live module"
                f" disagree on which tensors exist; only in model:"
                f" {sorted(set(current) - set(reference))[:5]}, only in checkpoint:"
                f" {sorted(set(reference) - set(current))[:5]}"
            )
        live_hash = hash_state_dict(current, dtype)
        ref_hash = hash_state_dict(reference, dtype)
        label = name or root.__class__.__name__
        if live_hash != ref_hash:
            raise RuntimeError(
                f"pretrained weights of '{label}' ({module.__class__.__name__}) do not"
                f" match their checkpoint: model sha256={live_hash},"
                f" checkpoint sha256={ref_hash}. The weights were modified after"
                " loading (e.g. re-initialised or overwritten by a checkpoint)."
            )
        checked[label] = live_hash
        if verbose:
            cprint(
                f"[hpt] pretrained weights of '{label}' verified: sha256={live_hash}",
                color="green",
            )
    return checked


def _snapshot_state_dict(
    snapshot_dir: str, prefix: str = "", dtype: Optional[torch.dtype] = None
) -> Optional[dict]:
    """Read a local HF snapshot's safetensors shards into a flat state dict.

    ``None`` if the snapshot has no safetensors (the caller then skips the
    check rather than hitting the network).
    """
    from safetensors.torch import load_file

    index = os.path.join(snapshot_dir, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, encoding="utf-8") as handle:
            shards = sorted(set(json.load(handle)["weight_map"].values()))
        files = [os.path.join(snapshot_dir, shard) for shard in shards]
    else:
        single = os.path.join(snapshot_dir, "model.safetensors")
        if not os.path.isfile(single):
            return None
        files = [single]

    state = {}
    for path in files:
        for key, tensor in load_file(path, device="cpu").items():
            state[f"{prefix}{key}"] = tensor if dtype is None else tensor.to(dtype)
    return state


## Taken directly from hpt/models/transformer with no modifications
class CrossAttention(nn.Module):
    """
    CrossAttention module used in the Perceiver IO model.

    Args:
        query_dim (int): The dimension of the query input.
        heads (int, optional): The number of attention heads. Defaults to 8.
        dim_head (int, optional): The dimension of each attention head. Defaults to 64.
        dropout (float, optional): The dropout probability. Defaults to 0.0.
    """

    def __init__(
        self, query_dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the CrossAttention module.

        Args:
            x (torch.Tensor): The query input tensor.
            context (torch.Tensor): The context input tensor.
            mask (torch.Tensor, optional): The attention mask tensor. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.
        """
        h = self.heads
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if mask is not None:
            # fill in the masks with negative values
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        # dropout
        attn = self.dropout(attn)
        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """
        Initialize the Transformer model.

        Args:
            dim (int): The input dimension of the model.
            num_heads (int, optional): The number of attention heads. Defaults to 8.
            qkv_bias (bool, optional): Whether to include bias in the query, key, and value linear layers. Defaults to False.
            qk_scale (float, optional): Scale factor for query and key. Defaults to None.
            attn_drop (float, optional): Dropout rate for attention weights. Defaults to 0.0.
            proj_drop (float, optional): Dropout rate for the output of the projection layer. Defaults to 0.0.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable = nn.GELU,
        drop: float = 0.0,
    ):
        """
        Initialize the Transformer model.

        Args:
            in_features (int): Number of input features.
            hidden_features (int, optional): Number of hidden features. Defaults to None.
            out_features (int, optional): Number of output features. Defaults to None.
            act_layer (torch.nn.Module, optional): Activation layer. Defaults to nn.GELU.
            drop (float, optional): Dropout rate. Defaults to 0.0.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BlockWithMasking(nn.Module):
    def __init__(
        self,
        dim: int,
        attn_target: Callable,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        ffn_dropout_rate: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_type: Optional[str] = None,
        layer_scale_init_value: float = 1e-4,
    ):
        super().__init__()

        assert not isinstance(
            attn_target, nn.Module
        ), "attn_target should be a Callable. Otherwise attn_target is shared across blocks!"
        self.attn = attn_target()
        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.norm_1 = norm_layer(dim)
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=ffn_dropout_rate,
        )
        self.norm_2 = norm_layer(dim)
        self.layer_scale_type = layer_scale_type
        if self.layer_scale_type is not None:
            assert self.layer_scale_type in [
                "per_channel",
                "scalar",
            ], f"Found Layer scale type {self.layer_scale_type}"
            if self.layer_scale_type == "per_channel":
                # one gamma value per channel
                gamma_shape = [1, 1, dim]
            elif self.layer_scale_type == "scalar":
                # single gamma value for all channels
                gamma_shape = [1, 1, 1]
            # two gammas: for each part of the fwd in the encoder
            self.layer_scale_gamma1 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )
            self.layer_scale_gamma2 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        if self.layer_scale_type is None:
            x = x + self.drop_path(self.attn(self.norm_1(x), attn_mask))
            x = x + self.drop_path(self.mlp(self.norm_2(x)))
        else:
            x = (
                x
                + self.drop_path(self.attn(self.norm_1(x), attn_mask))
                * self.layer_scale_gamma1
            )
            x = x + self.drop_path(self.mlp(self.norm_2(x))) * self.layer_scale_gamma2
        return x


_LAYER_NORM = partial(nn.LayerNorm, eps=1e-6)


class MultiheadAttention(nn.MultiheadAttention):
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return super().forward(x, x, x, need_weights=False, attn_mask=attn_mask)[0]


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        attn_target: Callable,
        embed_dim: int,
        num_blocks: int,
        block: Callable = BlockWithMasking,
        pre_transformer_layer: Optional[Callable] = None,
        post_transformer_layer: Optional[Callable] = None,
        drop_path_rate: float = 0.0,
        drop_path_type: str = "progressive",
        norm_layer: Callable = _LAYER_NORM,
        mlp_ratio: int = 4,
        ffn_dropout_rate: float = 0.0,
        layer_scale_type: Optional[
            str
        ] = None,  # from cait; possible values are None, "per_channel", "scalar"
        layer_scale_init_value: float = 1e-4,  # from cait; float
        weight_init_style: str = "pytorch",  # possible values jax or pytorch
    ):
        """
        Simple Transformer with the following features
        1. Supports masked attention
        2. Supports DropPath
        3. Supports LayerScale
        4. Supports Dropout in Attention and FFN
        5. Makes few assumptions about the input except that it is a Tensor
        """
        super().__init__()
        self.pre_transformer_layer = pre_transformer_layer
        if drop_path_type == "progressive":
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        elif drop_path_type == "uniform":
            dpr = [drop_path_rate for i in range(num_blocks)]
        else:
            raise ValueError(f"Unknown drop_path_type: {drop_path_type}")

        self.blocks = nn.Sequential(
            *[
                block(
                    dim=embed_dim,
                    attn_target=attn_target,
                    mlp_ratio=mlp_ratio,
                    ffn_dropout_rate=ffn_dropout_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    layer_scale_type=layer_scale_type,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for i in range(num_blocks)
            ]
        )
        self.post_transformer_layer = post_transformer_layer
        self.weight_init_style = weight_init_style
        self.apply(self._init_weights)

    def forward(
        self,
        tokens: torch.Tensor,
        attn_mask: torch.Tensor = None,
        use_checkpoint: bool = False,
        checkpoint_every_n: int = 1,
        checkpoint_blk_ids: Optional[List[int]] = None,
    ):
        """
        Inputs
        - tokens: data of shape N x L x D (or L x N x D depending on the attention implementation)
        - attn: mask of shape L x L

        Output
        - x: data of shape N x L x D (or L x N x D depending on the attention implementation)
        """
        block_outputs = []
        if self.pre_transformer_layer:
            tokens = self.pre_transformer_layer(tokens)
        if use_checkpoint and checkpoint_blk_ids is None:
            checkpoint_blk_ids = [
                blk_id
                for blk_id in range(len(self.blocks))
                if blk_id % checkpoint_every_n == 0
            ]
        if checkpoint_blk_ids:
            checkpoint_blk_ids = set(checkpoint_blk_ids)
        for blk_id, blk in enumerate(self.blocks):
            if use_checkpoint and blk_id in checkpoint_blk_ids:
                tokens = checkpoint.checkpoint(
                    blk, tokens, attn_mask, use_reentrant=False
                )
            else:
                tokens = blk(tokens, attn_mask=attn_mask)
            block_outputs.append(tokens)
        if self.post_transformer_layer:
            tokens = self.post_transformer_layer(tokens)
        return tokens, block_outputs

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.weight_init_style == "jax":
                # Based on MAE and official Jax ViT implementation
                torch.nn.init.xavier_uniform_(m.weight)

            elif self.weight_init_style == "pytorch":
                # PyTorch ViT uses trunc_normal_
                trunc_normal_(m.weight, std=0.02)

            elif self.weight_init_style == "allzero":
                # PyTorch ViT uses trunc_normal_
                torch.nn.init.constant_(m.weight, 0)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


# --------------------------------------------------------
# Policy stem from hpt/models/policy_stem.py
# Changelog:
#
# --------------------------------------------------------
INIT_CONST = 0.02


class PolicyStem(nn.Module):
    """policy stem"""

    def __init__(self, **kwargs):
        super().__init__()
        self.specs = kwargs.get("specs")

    def init_cross_attn(self, stem_spec):
        """initialize cross attention module and the learnable tokens"""
        token_num = stem_spec.crossattn_latent
        self.tokens = nn.Parameter(
            torch.randn(1, token_num, stem_spec.modality_embed_dim) * INIT_CONST
        )

        self.cross_attention = CrossAttention(
            stem_spec.modality_embed_dim,
            heads=stem_spec.crossattn_heads,
            dim_head=stem_spec.crossattn_dim_head,
            dropout=stem_spec.crossattn_modality_dropout,
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @property
    def device(self):
        return next(self.parameters()).device

    def compute_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the latent representations of input data by attention.

        Args:
            Input tensor with shape [32, 3, 1, 49, 512] representing the batch size,
            horizon, instance (e.g. num of views), number of features, and feature dimensions respectively.

        Returns:
            Output tensor with latent tokens, shape [32, 16, 128], where 16 is the number
            of tokens and 128 is the dimensionality of each token.

        Examples for vision features from ResNet:
        >>> x = np.random.randn(32, 3, 1, 49, 512)
        >>> latent_tokens = model.compute_latent(x)
        >>> print(latent_tokens.shape)
        (32, 16, 128)

        Examples for proprioceptive features:
        >>> x = np.random.randn(32, 3, 1, 7)
        >>> latent_tokens = model.compute_latent(x)
        >>> print(latent_tokens.shape)
        (32, 16, 128)
        """
        # Initial reshape to adapt to token dimensions
        # (32, 3, 1, 49, 128)
        stem_feat = self(x)
        stem_feat = stem_feat.reshape(
            stem_feat.shape[0], -1, stem_feat.shape[-1]
        )  # (32, 147, 128)
        # Replicating tokens for each item in the batch and computing cross-attention
        stem_tokens = self.tokens.repeat(len(stem_feat), 1, 1)  # (32, 16, 128)
        stem_tokens = self.cross_attention(stem_tokens, stem_feat)  # (32, 16, 128)
        return stem_tokens


class STPolicyStem(nn.Module):
    """Policy Stem that tokenizes different modalities into the same latent space.
    This version implements the spatial temporal version.
    It uses conv2D to handle 1-dimension features [B, T, L, D]
    It uses conv3D to handle 1-dimension features [B, T, H, W, D]
    # https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/main/videollama2/model/projector.py
    """

    def __init__(self, dimension=2, **kwargs):
        super().__init__(**kwargs)

    def init(self, stem_spec, modality):
        """initialize cross attention module and the learnable tokens"""
        downsample_tokens = getattr(stem_spec.crossattn_latent, modality)
        stem_modality_spec = getattr(stem_spec, modality)

        if stem_modality_spec.conv_dimension == 2:
            self.conv_dim = stem_modality_spec.conv_dimension
            dim_token = downsample_tokens
            self.conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=stem_modality_spec.input_dim,
                    out_channels=stem_modality_spec.output_dim,
                    kernel_size=stem_modality_spec.filter_size,
                    stride=1,
                    padding=stem_modality_spec.hidden_dim_tokens,
                )
                ** (1.0 / 3)
            )
            self.conv = nn.Sequential(
                nn.Conv3d(
                    in_channels=stem_modality_spec.input_dim,
                    out_channels=stem_modality_spec.output_dim,
                    kernel_size=stem_modality_spec.filter_size,
                    stride=1,
                    padding=stem_modality_spec.filter_size // 2,
                    bias=True,
                ),
                nn.SiLU(),
            )
            self.pool = nn.AdaptiveAvgPool3d((dim_token, dim_token, dim_token))

    def compute_latent(self, x):
        """
        Args:
            example x: Input tensor with shape [32, 3, 1, 49, 512] representing the batch size,
            horizon, instance (e.g. num of views), number of features, and feature dimensions respectively.
            Average over the number of instances.
        """
        B, T, num_instances, *_ = x.shape
        x = rearrange(x, "B T I ... D -> (B I) D T ...")

        if self.conv_dim == 3:
            # assume fixed width and height
            x = rearrange(
                x,
                "B D T (W1 W2) -> B D T W1 W2",
                W1=int(x.shape[-1] ** (1 / 2)),
                W2=int(x.shape[-1] ** (1 / 2)),
            )
        out = self.conv(x)
        out = self.pool(out)
        out = rearrange(out, "(B I) D ... -> B I (...) D", B=B, I=num_instances).mean(
            dim=1
        )
        return out


class AttentivePooling(nn.Module):
    """attentive pooling with cross attention"""

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, embed_dim) * INIT_CONST)
        self.cross_attention = CrossAttention(embed_dim, heads=8, dim_head=64)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, D]
        tokens = self.token.repeat(len(x), 1, 1)
        x = self.cross_attention(tokens, x)
        return x


class MLPPolicyStem(PolicyStem):
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        widths: List[int] = [512],
        tanh_end: bool = False,
        ln: bool = True,
        num_of_copy: int = 1,
        **kwargs,
    ) -> None:
        """vanilla MLP class"""
        super().__init__(**kwargs)
        modules = [nn.Linear(input_dim, widths[0]), nn.SiLU()]

        for i in range(len(widths) - 1):
            modules.extend([nn.Linear(widths[i], widths[i + 1])])
            if ln:
                modules.append(nn.LayerNorm(widths[i + 1]))
            modules.append(nn.SiLU())

        modules.append(nn.Linear(widths[-1], output_dim))
        if tanh_end:
            modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
        self.num_of_copy = num_of_copy
        if self.num_of_copy > 1:
            self.net = nn.ModuleList(
                [nn.Sequential(*modules) for _ in range(num_of_copy)]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the model.
        Args:
            x: Image tensor with shape [B, T, N, 3, H, W] representing the batch size,
            horizon, instance (e.g. num of views)
        Returns:
            Flatten tensor with shape [B, M, 512]
        """
        if self.num_of_copy > 1:
            out = []
            iter_num = min(self.num_of_copy, x.shape[1])
            for idx in range(iter_num):
                input = x[:, idx]
                net = self.net[idx]
                out.append(net(input))
            y = torch.stack(out, dim=1)
        else:
            y = self.net(x)
        return y


class ResNet(PretrainedWeights, PolicyStem):
    # ``self.net`` is the ImageNet-pretrained torchvision backbone; ``self.proj``
    # is built by HPT and must keep its xavier init.
    _hpt_pretrained_attrs = ("net",)

    def __init__(
        self,
        output_dim: int = 10,
        weights: str = "DEFAULT",
        resnet_model: str = "resnet18",
        num_of_copy: int = 1,
        freeze_backbone: bool = False,
        **kwargs,
    ) -> None:
        """ResNet Encoder for Images"""
        super().__init__(**kwargs)
        pretrained_model = getattr(torchvision.models, resnet_model)(weights=weights)

        # by default we use a separate image encoder for each view in downstream evaluation
        self.num_of_copy = num_of_copy
        self.net = nn.Sequential(*list(pretrained_model.children())[:-2])

        if num_of_copy > 1:
            self.net = nn.ModuleList(
                [
                    nn.Sequential(*list(pretrained_model.children())[:-2])
                    for _ in range(num_of_copy)
                ]
            )
        self.input = input
        self.out_dim = output_dim
        self.to_tensor = transforms.ToTensor()
        self.proj = nn.Linear(512, output_dim)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # Freeze the backbone if specified
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze all parameters in the ResNet backbone"""
        if isinstance(self.net, nn.ModuleList):
            for net in self.net:
                for param in net.parameters():
                    param.requires_grad = False
        else:
            for param in self.net.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the model.
        Args:
            x: Image tensor with shape [B, T, N, 3, H, W] representing the batch size,
            horizon, instance (e.g. num of views)
        Returns:
            Flatten tensor with shape [B, M, 512]
        """
        B, *_, H, W = x.shape
        x = x.view(len(x), -1, 3, H, W)
        if self.num_of_copy > 1:
            # separate encoding for each view
            out = []
            iter_num = min(self.num_of_copy, x.shape[1])
            for idx in range(iter_num):
                input = x[:, idx]
                net = self.net[idx]
                out.append(net(input))
            feat = torch.stack(out, dim=1)
        else:
            x = x.view(-1, 3, H, W)
            feat = self.net(x)
        # concat along time
        feat = feat.view(B, feat.shape[1], -1).transpose(1, 2)
        feat = self.proj(feat)
        return feat


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


def _local_snapshot_if_offline(model_name: str) -> str:
    """Resolve a HF repo id to its cached snapshot directory when the process
    runs offline (``HF_HUB_OFFLINE=1`` / ``TRANSFORMERS_OFFLINE=1``); a local
    path or an online process is returned unchanged."""
    if os.path.isdir(model_name):
        return model_name
    offline = os.environ.get("HF_HUB_OFFLINE") == "1" or (
        os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    )
    if not offline:
        return model_name
    from huggingface_hub import snapshot_download

    return snapshot_download(model_name, local_files_only=True)


class _Qwen3BaseEncoder(PretrainedWeights, PolicyStem):
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
        normalize_pooled: only used by the pooled subclass; L2-normalizes the
            sentence embedding (Qwen3 official recipe).
    """

    DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    _hpt_pretrained_attrs = ("encoder",)

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
        # transformers >= 4.57 probes the Hub while loading a tokenizer by repo
        # id even under HF_HUB_OFFLINE=1 (tokenization_utils_base
        # ._patch_mistral_regex -> model_info), which raises on compute nodes
        # with no internet. Loading from the cached snapshot DIRECTORY skips
        # that probe, so resolve the repo id to its local snapshot when offline.
        load_path = _local_snapshot_if_offline(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path, padding_side="left")
        self.encoder = AutoModel.from_pretrained(load_path, dtype=torch_dtype)
        # Remembered so the weight-hash check can read the same snapshot back
        # and compare in the dtype the encoder was loaded in.
        self._snapshot_dir = load_path
        self._load_dtype = torch_dtype
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

    def pretrained_reference_state_dict(self) -> Optional[dict]:
        """The snapshot's own safetensors, cast exactly the way
        ``from_pretrained(dtype=...)`` casts them."""
        if not os.path.isdir(self._snapshot_dir):
            return None
        return _snapshot_state_dict(
            self._snapshot_dir, prefix="encoder.", dtype=self._load_dtype
        )

    def pretrained_hash_dtype(self) -> Optional[torch.dtype]:
        """Hash in the load dtype. ``HPT.__init__`` upcasts the whole model with
        ``nets.float()``; fp16 -> fp32 -> fp16 is exact, so casting the live
        weights back down compares like for like without a second model load."""
        return self._load_dtype

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

    def forward_with_mask(self, prompts):
        """Returns ``(features, attention_mask)``; the mask is what padded
        positions must be excluded by, since zeroing them before the biased
        ``proj`` turns them into the learned constant ``proj.bias``."""
        hidden, mask = self._encode(prompts)
        feat = hidden.float() * mask.unsqueeze(-1).float()
        return self.proj(feat), mask.bool()  # (B, L, output_dim), (B, L)

    def forward(self, prompts):
        return self.forward_with_mask(prompts)[0]

    def compute_latent(self, prompts):
        feat, mask = self.forward_with_mask(prompts)  # (B, L, output_dim), (B, L)
        stem_tokens = self.tokens.repeat(feat.shape[0], 1, 1)
        return self.cross_attention(stem_tokens, feat, mask=mask)


# --------------------------------------------------------------------------
# Qwen 3.5 VLM stem pair
#
# ``Qwen35VLMEncoder`` replaces the ResNet as ``encoder_specs.front_img_1``.
# It runs ONE joint image+text forward of Qwen 3.5 per batch and splits the
# resulting hidden states by token type: the visual tokens go back to the
# image stem (the existing ``MLPPolicyStem`` + Perceiver, ``input_dim`` 1024),
# and the text tokens are parked on the encoder for ``Qwen35TextStem``
# (``shared_stem_specs.annotation``) to pick up in the same ``stem_process``
# pass. Because the LM is causal and the image precedes the text in the Qwen
# chat layout, the text tokens are image-conditioned and the visual tokens are
# not text-conditioned.
# --------------------------------------------------------------------------


class Qwen35VLMEncoder(PretrainedWeights, PolicyStem):
    """Qwen 3.5 VLM as the HPT image encoder.

    Args:
        model_name: HF identifier (or local snapshot dir) for the VLM.
        dtype: weight dtype the VLM is loaded and RUN in. ``HPT.__init__``
            upcasts the whole policy with ``nets.float()``; ``_apply`` below
            puts the VLM back, so an 853M VLM does not silently cost 3.4 GB and
            an fp32 forward.
        freeze: freeze the VLM (default). Combined with ``trainable_layers=0``
            the whole forward runs under ``torch.no_grad()`` in eval mode.
        trainable_layers: 0 = fully frozen; N > 0 unfreezes the LAST N text
            layers (and drops the ``no_grad``), for the follow-up run.
        feature_layer: index into ``hidden_states``; -1 (default) is the final
            post-norm ``last_hidden_state``, taken without materialising the
            248k-wide ``lm_head`` logits.
        image_size: (H, W) the frames are resized to before the processor. The
            processor's own resize is disabled, so the visual token count is
            pinned at ``(H / (patch * merge)) * (W / (patch * merge))``
            (640x352 -> 220). Asserted once at init against a dummy image.
        max_text_tokens: prompts are truncated to this many tokens.
    """

    DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
    _hpt_pretrained_attrs = ("model",)

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        dtype: str = "bfloat16",
        freeze: bool = True,
        trainable_layers: int = 0,
        feature_layer: int = -1,
        image_size: tuple = (352, 640),
        max_text_tokens: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_name = model_name
        self.feature_layer = int(feature_layer)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.max_text_tokens = int(max_text_tokens)
        self.freeze_encoder = bool(freeze)
        self.trainable_layers = int(trainable_layers)
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        load_path = _local_snapshot_if_offline(model_name)
        self.processor = AutoProcessor.from_pretrained(load_path)
        # Right padding: a prompt's real tokens always start at position 0, so
        # its RoPE phase does not depend on how long its batch-mates are.
        self.processor.tokenizer.padding_side = "right"
        self.model = AutoModelForImageTextToText.from_pretrained(
            load_path, dtype=torch_dtype
        )
        # Remembered so the weight-hash check can read the same snapshot back.
        self._snapshot_dir = load_path
        self._load_dtype = torch_dtype

        config = self.model.config
        self.image_token_id = int(config.image_token_id)
        self.hidden_size = int(config.text_config.hidden_size)
        self.output_dim = self.hidden_size
        stride = int(config.vision_config.patch_size) * int(
            config.vision_config.spatial_merge_size
        )
        self._token_stride = stride
        if self.image_size[0] % stride or self.image_size[1] % stride:
            raise ValueError(
                f"image_size {self.image_size} must be divisible by {stride}"
                " (vision patch_size x spatial_merge_size)"
            )

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            if self.trainable_layers > 0:
                for layer in self._text_layers()[-self.trainable_layers :]:
                    for param in layer.parameters():
                        param.requires_grad = True
        elif self.trainable_layers:
            raise ValueError("trainable_layers only applies when freeze=True")
        self._no_grad = bool(freeze) and self.trainable_layers == 0

        # Per-batch state, set by ``set_prompts`` / filled by ``forward``.
        # Plain attributes, so nothing lands in the state dict.
        self._prompts = None
        self.last_text_features = None
        self.last_text_mask = None
        self.num_visual_tokens = self._assert_visual_token_count()

    # -- setup helpers ----------------------------------------------------

    def _text_layers(self) -> nn.ModuleList:
        """The text stack's decoder layers, wherever transformers puts them."""
        depth = int(self.model.config.text_config.num_hidden_layers)
        for name, module in self.model.named_modules():
            if (
                name.endswith("layers")
                and isinstance(module, nn.ModuleList)
                and len(module) == depth
            ):
                return module
        raise RuntimeError(
            f"could not locate the {depth} text layers of {self.model_name}"
        )

    def _assert_visual_token_count(self) -> int:
        """Run the processor once on a dummy frame and pin the token count."""
        dummy = torch.zeros(1, 3, *self.image_size)
        inputs = self._processor_inputs(dummy, [""])
        counts = (inputs["input_ids"] == self.image_token_id).sum(dim=1)
        found = int(counts[0])
        expected = (self.image_size[0] // self._token_stride) * (
            self.image_size[1] // self._token_stride
        )
        if found != expected:
            raise RuntimeError(
                f"{self.model_name} produced {found} visual tokens for an"
                f" {self.image_size} frame, expected {expected}"
            )
        return found

    # -- per-batch plumbing ------------------------------------------------

    def set_prompts(self, prompts) -> None:
        """Hand the encoder the batch's prompts, just before ``forward``.

        ``HPTModel.stem_process`` calls this; ``None`` (no annotation modality
        in the batch) degrades to empty prompts, i.e. image tokens only.
        """
        self._prompts = None if prompts is None else list(prompts)

    def _processor_inputs(self, images: torch.Tensor, prompts: List[str]) -> dict:
        """Chat layout (image first, then the prompt) for every sample."""
        tokenizer = self.processor.tokenizer
        texts = []
        for prompt in prompts:
            ids = tokenizer(str(prompt), add_special_tokens=False)["input_ids"]
            text = tokenizer.decode(ids[: self.max_text_tokens])
            texts.append(
                self.processor.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": text},
                            ],
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        # do_resize / do_rescale off: the frames arrive as float tensors already
        # in [0, 1] at exactly ``image_size``, so only the processor's
        # normalization and patchification run (this is what replaces the
        # ResNet path's ImageNet ``Normalize``).
        return self.processor(
            text=texts,
            images=list(images),
            padding=True,
            return_tensors="pt",
            do_rescale=False,
            do_resize=False,
        )

    @staticmethod
    def _pack_tokens(hidden: torch.Tensor, keep: torch.Tensor):
        """Gather the ``keep`` positions of each row, right-padded to the max."""
        batch, _, dim = hidden.shape
        width = max(int(keep.sum(dim=1).max().item()), 1)
        feats = hidden.new_zeros(batch, width, dim)
        mask = torch.zeros(batch, width, dtype=torch.bool, device=hidden.device)
        slots = keep.long().cumsum(dim=1) - 1
        rows, cols = keep.nonzero(as_tuple=True)
        feats[rows, slots[rows, cols]] = hidden[rows, cols]
        mask[rows, slots[rows, cols]] = True
        return feats, mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One joint VLM forward; returns the visual tokens ``(B, N_vis, D)``.

        The text tokens of the SAME forward are parked on
        ``last_text_features`` / ``last_text_mask`` for ``Qwen35TextStem``.
        """
        batch, *_, height, width = x.shape
        x = x.reshape(batch, -1, 3, height, width)
        if x.shape[1] != 1:
            raise ValueError(
                "Qwen35VLMEncoder handles exactly one view per sample,"
                f" got {x.shape[1]}"
            )
        images = x[:, 0].float()
        if (height, width) != self.image_size:
            images = F.interpolate(
                images,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        images = images.clamp(0.0, 1.0)

        prompts = self._prompts if self._prompts is not None else [""] * batch
        if len(prompts) != batch:
            raise RuntimeError(
                f"Qwen35VLMEncoder got {len(prompts)} prompts for {batch} images"
            )

        inputs = self._processor_inputs(images, prompts)
        device = next(self.model.parameters()).device
        inputs = {
            key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in inputs.items()
        }
        if torch.is_tensor(inputs.get("pixel_values")):
            inputs["pixel_values"] = inputs["pixel_values"].to(self._load_dtype)

        want_all = self.feature_layer != -1
        if self._no_grad:
            with torch.no_grad():
                out = self.model.model(**inputs, output_hidden_states=want_all)
        else:
            out = self.model.model(**inputs, output_hidden_states=want_all)
        # ``self.model.model`` is the VLM without the lm_head: taking
        # ``last_hidden_state`` here avoids materialising a
        # (B, L, 248320) logit tensor that nothing uses.
        hidden = (
            out.hidden_states[self.feature_layer]
            if want_all
            else (out.last_hidden_state)
        )

        image_positions = inputs["input_ids"] == self.image_token_id
        counts = image_positions.sum(dim=1)
        if not bool((counts == self.num_visual_tokens).all()):
            raise RuntimeError(
                f"expected {self.num_visual_tokens} visual tokens per sample,"
                f" got {counts.tolist()}"
            )
        visual = hidden[image_positions].view(batch, self.num_visual_tokens, -1)

        # Every real (non-padding) token that is not an image token, i.e. the
        # prompt tokens AND the fixed chat-template tokens around them. The
        # template is identical for every sample, so it is a constant context
        # the cross-attention can learn to ignore; separating it out would cost
        # a second tokenizer pass per batch for no information.
        text_positions = inputs["attention_mask"].bool() & ~image_positions
        feats, mask = self._pack_tokens(hidden, text_positions)
        self.last_text_features = feats.float()
        self.last_text_mask = mask
        self._prompts = None
        return visual.float()

    # -- pretrained-weight bookkeeping -------------------------------------

    def _checkpoint_keys(self) -> Optional[set]:
        """The snapshot's tensor names, prefixed like ``pretrained_state_dict``."""
        index = os.path.join(self._snapshot_dir, "model.safetensors.index.json")
        if not os.path.isfile(index):
            return None
        with open(index, encoding="utf-8") as handle:
            return {f"model.{key}" for key in json.load(handle)["weight_map"]}

    def pretrained_state_dict(self) -> dict:
        """Live weights, restricted to the tensors the checkpoint actually has.

        Qwen 3.5 ties ``lm_head.weight`` to the input embedding, so the live
        module has a name the snapshot does not; the embedding itself is
        checked, so nothing is lost.
        """
        state = super().pretrained_state_dict()
        keys = self._checkpoint_keys()
        if keys is None:
            return state
        return {name: value for name, value in state.items() if name in keys}

    def pretrained_reference_state_dict(self) -> Optional[dict]:
        """The snapshot's own safetensors, restricted to what the model built.

        The snapshot also ships the multi-token-prediction head (``mtp.*``),
        which ``Qwen3_5ForConditionalGeneration`` does not instantiate.
        """
        if not os.path.isdir(self._snapshot_dir):
            return None
        reference = _snapshot_state_dict(
            self._snapshot_dir, prefix="model.", dtype=self._load_dtype
        )
        if reference is None:
            return None
        live = set(super().pretrained_state_dict())
        return {name: value for name, value in reference.items() if name in live}

    def pretrained_hash_dtype(self) -> Optional[torch.dtype]:
        return self._load_dtype

    def _apply(self, fn, recurse: bool = True):
        """Keep the VLM in its load dtype through ``nets.float()``."""
        out = super()._apply(fn, recurse=recurse)
        load_dtype = getattr(self, "_load_dtype", None)
        model = getattr(self, "model", None)
        if model is not None and load_dtype not in (None, torch.float32):
            model.to(dtype=load_dtype)
        return out

    def train(self, mode: bool = True):
        """Keep the frozen VLM in eval mode regardless of the outer train flag."""
        super().train(mode)
        if self.freeze_encoder:
            self.model.eval()
        return self


class Qwen35TextStem(PolicyStem):
    """Text half of the joint Qwen 3.5 VLM forward.

    Registered as ``shared_stem_specs.annotation``. ``compute_latent`` ignores
    the prompt strings - the ``Qwen35VLMEncoder`` already consumed them - and
    reads the text-token hidden states that encoder parked for this batch. The
    reference to the encoder is a PLAIN attribute, not a registered submodule,
    so the VLM weights exist exactly once in the policy; ``HPT.__init__`` wires
    it once both objects are built.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 840,
        vlm_modality: str = "front_img_1",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.vlm_modality = vlm_modality
        self.proj = (
            nn.Linear(self.input_dim, self.output_dim)
            if self.input_dim != self.output_dim
            else nn.Identity()
        )
        object.__setattr__(self, "_vlm", None)

    def attach_vlm(self, encoder: nn.Module) -> None:
        """Plain reference (``object.__setattr__``): NOT a registered child."""
        object.__setattr__(self, "_vlm", encoder)

    def forward_with_mask(self, prompts):
        """``(features, bool mask)``; same contract as ``QwenPerTokenEncoder``."""
        encoder = self._vlm
        if encoder is None:
            raise RuntimeError(
                "Qwen35TextStem has no VLM encoder attached; HPT.__init__ wires"
                " it from encoder_specs[vlm_modality]"
            )
        feats, mask = encoder.last_text_features, encoder.last_text_mask
        if feats is None or mask is None:
            raise RuntimeError(
                "the VLM encoder holds no text features for this batch; the"
                f" '{self.vlm_modality}' encoder must run before this stem"
            )
        if prompts is not None and len(prompts) != feats.shape[0]:
            raise RuntimeError(
                f"the VLM encoder holds {feats.shape[0]} rows of text features"
                f" but this batch has {len(prompts)} prompts"
            )
        # Zero the padding before the biased ``proj``, and keep the mask: the
        # zeros become ``proj.bias``, which the cross-attention must not see.
        feats = feats.float() * mask.unsqueeze(-1).float()
        return self.proj(feats), mask

    def forward(self, prompts):
        return self.forward_with_mask(prompts)[0]

    def compute_latent(self, prompts):
        feats, mask = self.forward_with_mask(prompts)
        stem_tokens = self.tokens.repeat(feats.shape[0], 1, 1)
        return self.cross_attention(stem_tokens, feats, mask=mask)


class T5Encoder(PretrainedWeights, PolicyStem):
    _hpt_pretrained_attrs = ("encoder",)

    def __init__(self, per_token=True, **kwargs) -> None:
        """T5 Encoder that expects pre-tokenized inputs

        Args:
            per_token (bool): If True, return per-token embeddings. If False, return mean-pooled embeddings
        """
        super().__init__(**kwargs)
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


def vit_base_patch16(checkpoint_path="output/mae_pretrain_vit_base.pth", **kwargs):
    # load pretrained weights to initialize vit model
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    print("load pretrained model:", checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path)["model"], strict=False)
    return model


# --------------------------------------------------------
# Standard Policy head from hpt/models/policy_head.py
# Changelog:
#
# --------------------------------------------------------

LOSS = partial(F.smooth_l1_loss, beta=0.05)
LOSS_MSE = partial(F.mse_loss)


class PolicyHead(nn.Module):
    """Abstract class for policy head."""

    def __init__(self, **kwargs):
        super().__init__()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @property
    def device(self):
        return next(self.parameters()).device

    def compute_loss(self, x: torch.Tensor, data: dict):
        """
        Compute smooth L1 loss between predicted and target actions,
        slicing as needed if their dimensions differ.

        Args:
            x (torch.Tensor): Transformer outputs used to predict actions.
            data (dict): Contains:
                - 'action': ground-truth action tensor of shape (B, T, D_target)

        Returns:
            torch.Tensor: Scalar loss
        """
        target_action = data["action"]
        B, T = target_action.shape[:2]

        pred_action = self(x).view(B, T, -1)

        D_pred = pred_action.shape[-1]
        D_target = target_action.shape[-1]

        D_common = min(D_pred, D_target)
        pred_action = pred_action[..., :D_common]
        target_action = target_action[..., :D_common]

        return LOSS(pred_action, target_action)


class MLPPolicyHead(PolicyHead):
    """Simple MLP based policy head"""

    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        widths: List[int] = [512],
        dropout: bool = False,
        tanh_end: bool = False,
        ln: bool = True,
        **kwargs,
    ) -> None:
        """vanilla MLP head on the pooled feature"""
        super().__init__()
        self.input = input
        modules = [nn.Linear(input_dim, widths[0]), nn.SiLU()]

        for i in range(len(widths) - 1):
            modules.extend([nn.Linear(widths[i], widths[i + 1])])
            if dropout:
                modules.append(nn.Dropout(p=0.1))
            if ln:
                modules.append(nn.LayerNorm(widths[i + 1]))
            modules.append(nn.SiLU())

        modules.append(nn.Linear(widths[-1], output_dim))
        if tanh_end:
            modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        """
        Forward pass of the policy head module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        y = self.net(x)
        return y


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.self_attention = Attention(
            dim=input_dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout,
        )

        self.cross_attention = CrossAttention(
            input_dim,
            heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.SiLU(), nn.Linear(input_dim, input_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)

    def forward(self, tokens, context):
        query = self.self_attention(self.norm1(tokens))
        query = tokens + query

        out = self.cross_attention(self.norm2(query), context)
        out = query + out

        mlp_out = self.mlp(self.norm3(out))
        tokens = mlp_out + out
        return tokens


class MultiBlockTransformerDecoder(PolicyHead):
    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 10,
        action_horizon: int = 16,
        latent_token_len: int = 8,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        num_layers: int = 4,
        final_norm: bool = False,
    ):
        super().__init__()
        self.tokens = nn.Parameter(
            torch.randn(1, action_horizon, input_dim) * INIT_CONST
        )
        self.pos_token = nn.Parameter(
            get_sinusoid_encoding_table(0, action_horizon, input_dim)
        )
        self.pos_context = nn.Parameter(
            get_sinusoid_encoding_table(0, latent_token_len, input_dim)
        )

        self.context_norm = nn.LayerNorm(input_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, output_dim),
        )

        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    input_dim=input_dim,
                    num_heads=num_heads,
                    dim_head=dim_head,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = final_norm
        if self.final_norm:
            self.last_layer_norm = nn.LayerNorm(input_dim)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[MultiBlockTransformerDecoder] Total trainable parameters: {total_params / 1e6:.2f}M"
        )

    def forward(self, x):
        B = x.shape[0]
        tokens = self.tokens.expand(B, -1, -1) + self.pos_token.expand(B, -1, -1)
        context = self.context_norm(x + self.pos_context.expand(B, -1, -1))

        for block in self.blocks:
            tokens = block(tokens, context)

        if self.final_norm:
            tokens = self.last_layer_norm(tokens)

        return self.out_proj(tokens)


class T5TokenizerWrapper:
    """Wrapper class for T5Tokenizer to prepare inputs for T5Encoder"""

    def __init__(self, model_name: str = "t5-base", max_length: int = 512):
        """
        Initialize T5 tokenizer wrapper.

        Args:
            model_name (str): Name of the T5 model to use for tokenization
            max_length (int): Maximum sequence length for tokenization
        """
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
