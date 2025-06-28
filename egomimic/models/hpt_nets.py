from einops import rearrange, repeat, reduce
from functools import partial
from typing import Callable, List, Optional, Union

import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from torch import nn, einsum
import torch.nn.functional as F

from timm.models.vision_transformer import VisionTransformer

from functools import partial

from transformers import T5Tokenizer, T5Model, AutoTokenizer

from einops import rearrange, repeat, reduce
from functools import partial
from typing import Callable, List, Optional, Union

import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from torch import nn, einsum
import torch.nn.functional as F

from timm.models.vision_transformer import VisionTransformer

from functools import partial

from transformers import T5Tokenizer, T5Model, AutoTokenizer
from transformers import CLIPTextModel, CLIPVisionModel # TODO: add CLIP

from egomimic.utils.egomimicUtils import get_sinusoid_encoding_table

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

    def __init__(self, query_dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
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
            x = x + self.drop_path(self.attn(self.norm_1(x), attn_mask)) * self.layer_scale_gamma1
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
        layer_scale_type: Optional[str] = None,  # from cait; possible values are None, "per_channel", "scalar"
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
            checkpoint_blk_ids = [blk_id for blk_id in range(len(self.blocks)) if blk_id % checkpoint_every_n == 0]
        if checkpoint_blk_ids:
            checkpoint_blk_ids = set(checkpoint_blk_ids)
        for blk_id, blk in enumerate(self.blocks):
            if use_checkpoint and blk_id in checkpoint_blk_ids:
                tokens = checkpoint.checkpoint(blk, tokens, attn_mask, use_reentrant=False)
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
        """ initialize cross attention module and the learnable tokens """
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

    def save(self, path : str):
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
        stem_feat = stem_feat.reshape(stem_feat.shape[0], -1, stem_feat.shape[-1])  # (32, 147, 128)
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
        """ initialize cross attention module and the learnable tokens """
        downsample_tokens = getattr(stem_spec.crossattn_latent, modality)
        stem_modality_spec = getattr(stem_spec, modality)

        if stem_modality_spec.conv_dimension == 2:
            self.conv_dim = stem_modality_spec.conv_dimension
            dim_token = downsample_tokens
            self.conv = nn.Sequential(nn.Conv1d(
                in_channels=stem_modality_spec.input_dim,
                out_channels=stem_modality_spec.output_dim,
                kernel_size=stem_modality_spec.filter_size,
                stride=1,
                padding=stem_modality_spec.hidden_dim_tokens) ** (1./3))
            self.conv = nn.Sequential(nn.Conv3d(
                in_channels=stem_modality_spec.input_dim,
                out_channels=stem_modality_spec.output_dim,
                kernel_size=stem_modality_spec.filter_size,
                stride=1,
                padding=stem_modality_spec.filter_size // 2,
                bias=True
            ), nn.SiLU())
            self.pool = nn.AdaptiveAvgPool3d((dim_token, dim_token, dim_token))

    def compute_latent(self, x):
        """
        Args:
            example x: Input tensor with shape [32, 3, 1, 49, 512] representing the batch size,
            horizon, instance (e.g. num of views), number of features, and feature dimensions respectively.
            Average over the number of instances.
        """
        B, T, I, *_ = x.shape
        x = rearrange(x, 'B T I ... D -> (B I) D T ...')

        if self.conv_dim == 3:
            # assume fixed width and height
            x = rearrange(x, 'B D T (W1 W2) -> B D T W1 W2', 
                          W1=int(x.shape[-1] ** (1/2)),  W2=int(x.shape[-1] ** (1/2)))
        out = self.conv(x)
        out = self.pool(out)
        out = rearrange(out, '(B I) D ... -> B I (...) D', B=B, I=I).mean(dim=1)
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

    def forward(self, x : torch.Tensor) -> torch.Tensor:
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
            self.net = nn.ModuleList([nn.Sequential(*modules) for _ in range(num_of_copy)])

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
    

class ResNet(PolicyStem):
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
                [nn.Sequential(*list(pretrained_model.children())[:-2]) for _ in range(num_of_copy)]
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


class T5Encoder(PolicyStem):
    def __init__(
        self,
        per_token=True,
        **kwargs
    ) -> None:
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
            return_dict=True
        )
        
        torch.cuda.empty_cache() # empty cache to save memory
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
    """ Abstract class for policy head."""

    def __init__(self, **kwargs):
        super().__init__()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def save(self, path : str):
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
    def __init__(self,
        input_dim: int = 10,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        self.self_attention = Attention(
            dim=input_dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout
        )
        
        self.cross_attention = CrossAttention(
            input_dim,
            heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        
        self.mlp = nn.Sequential(nn.Linear(input_dim, input_dim), nn.SiLU(), nn.Linear(input_dim, input_dim))
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
        self.tokens = nn.Parameter(torch.randn(1, action_horizon, input_dim) * INIT_CONST)
        self.pos_token = nn.Parameter(get_sinusoid_encoding_table(0, action_horizon, input_dim))
        self.pos_context = nn.Parameter(get_sinusoid_encoding_table(0, latent_token_len, input_dim))
    
        self.context_norm = nn.LayerNorm(input_dim)
        
        self.out_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, output_dim),
        )
        
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(
                input_dim=input_dim,
                num_heads=num_heads,
                dim_head=dim_head,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = final_norm
        if self.final_norm:
            self.last_layer_norm = nn.LayerNorm(input_dim)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MultiBlockTransformerDecoder] Total trainable parameters: {total_params / 1e6:.2f}M")

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
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }

class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=-1)
