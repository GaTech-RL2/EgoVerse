import logging
import math
from typing import Union

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

logger = logging.getLogger(__name__)


def posemb_sincos(
    pos: torch.Tensor,  # shape: (B,)
    embedding_dim: int,
    min_period: float,
    max_period: float,
) -> torch.Tensor:  # returns (B, embedding_dim)
    """
    Sine–cosine positional embedding (PyTorch).

    Args:
      pos           : 1-D tensor of positions, length B
      embedding_dim : must be even
      min_period    : smallest wavelength
      max_period    : largest wavelength

    Returns:
      Tensor of shape (B, embedding_dim) on the same device/dtype as `pos`.
    """
    if embedding_dim % 2:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    # (embedding_dim // 2,)
    fraction = torch.linspace(
        0.0, 1.0, embedding_dim // 2, device=pos.device, dtype=pos.dtype
    )
    period = min_period * (max_period / min_period) ** fraction

    # (B, embedding_dim // 2)
    sinusoid_input = torch.einsum("i,j->ij", pos, (1.0 / period) * 2 * torch.pi)

    return torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(
        self, inp_channels, out_channels, kernel_size, n_groups=8, conv_groups=1
    ):
        super().__init__()

        # conv_groups splits the CHANNEL axis into independent paths (this is
        # nn.Conv1d's `groups`, unrelated to n_groups, which is GroupNorm's).
        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                groups=conv_groups,
            ),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, *args, **kwargs):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        first_conv_groups=1,
    ):
        super().__init__()

        # Only the FIRST conv is grouped. Grouping deeper layers would keep the
        # two streams apart for the whole trunk, which is a different (much
        # stronger) architecture; the point here is only to stop layer one from
        # blending translation and rotation channels.
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    n_groups=n_groups,
                    conv_groups=first_conv_groups,
                ),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )

        # make sure dimensions compatible
        # The residual path would re-mix the streams even if blocks[0] does not,
        # so it has to carry the same grouping.
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1, groups=first_conv_groups)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond, *args, **kwargs):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        cond_dim=None,
        diffusion_step_embed_dim=256,
        ac_latent_seq=8,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        input_groups=1,
    ):
        """Build a one-dimensional U-Net with global conditioning.

        ``input_groups`` splits the INPUT channel axis into that many
        independent paths through the first residual block, so semantically
        distinct channel blocks are not blended by layer one. For the stacked
        ARC token ``[x, y, v_xy, cos, sin, omega]``, ``input_groups=2`` keeps
        translation and rotation apart until the second convolution. It relies
        on that channel ORDER, so it is only meaningful for a token whose
        channels are grouped contiguously by stream.
        """
        super().__init__()
        input_groups = int(input_groups)
        if input_groups < 1:
            raise ValueError("input_groups must be at least one")
        if input_groups > 1 and (
            input_dim % input_groups or down_dims[0] % input_groups
        ):
            raise ValueError(
                f"input_groups={input_groups} must divide both input_dim="
                f"{input_dim} and down_dims[0]={down_dims[0]}"
            )
        self.input_groups = input_groups
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed // 2),
            nn.Linear(dsed // 2, dsed * 2),
            nn.Mish(),
            nn.Linear(dsed * 2, dsed // 2),
        )
        self.proj_cond = nn.Linear(ac_latent_seq * cond_dim, dsed // 2)

        cond_dim = dsed

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        layer_func = ConditionalResidualBlock1D

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                layer_func(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                layer_func(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        layer_func(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            # Only the first down module touches raw token
                            # channels; everything deeper is already mixed.
                            first_conv_groups=input_groups if ind == 0 else 1,
                        ),
                        layer_func(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        layer_func(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        layer_func(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond,
    ):
        """
        sample: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, "b h t -> b t h")

        # NB the rearrange above turns (B, T, D) into (B, D, T): channels are
        # dim 1 and TIME is dim 2. Each down module halves the time axis, so an
        # input length not divisible by 2**(len(down_modules)-1) leaves the skip
        # connection and the upsampled feature map at different lengths
        # ("Sizes of tensors must match except in dimension 1"). Pad the time
        # axis by REPEATING the last step, run the net, then trim back. When the
        # length already divides evenly _pad is 0 and this is a no-op, so
        # existing configurations stay bit-identical.
        _T = sample.shape[2]
        _factor = 2 ** max(len(self.down_modules) - 1, 0)
        _pad = (-_T) % _factor
        if _pad:
            sample = torch.cat(
                [sample, sample[:, :, -1:].expand(-1, -1, _pad)], dim=2
            )

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )

        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        global_feature = self.diffusion_step_encoder(timesteps)

        cond = self.proj_cond(global_cond)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, cond], axis=-1)

        x = sample
        h = []
        # print("sample shape:", sample.shape)

        # downsample upsample in which dimensions?
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            # print("global_feature 1:", global_feature.shape, x.shape)
            x = resnet2(x, global_feature)
            # print("global_feature 2:", global_feature.shape, x.shape)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            # print("x shape", x.shape, h[0].shape, h[-1].shape)
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, "b t h -> b h t")
        if _pad:
            x = x[:, :_T, :]
        return x


class PaperConditionalUnet1D(nn.Module):
    """Diffusion Policy U-Net with global observation conditioning."""

    def __init__(
        self,
        input_dim,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.global_cond_dim = None if global_cond_dim is None else int(global_cond_dim)
        all_dims = [self.input_dim] + [int(value) for value in down_dims]
        start_dim = all_dims[1]
        embed_dim = int(diffusion_step_embed_dim)
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Mish(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        condition_dim = embed_dim + (self.global_cond_dim or 0)
        pairs = list(zip(all_dims[:-1], all_dims[1:]))

        def residual(input_channels, output_channels):
            return ConditionalResidualBlock1D(
                input_channels,
                output_channels,
                cond_dim=condition_dim,
                kernel_size=int(kernel_size),
                n_groups=int(n_groups),
                cond_predict_scale=bool(cond_predict_scale),
            )

        middle = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [residual(middle, middle), residual(middle, middle)]
        )
        self.down_modules = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        residual(input_channels, output_channels),
                        residual(output_channels, output_channels),
                        Downsample1d(output_channels)
                        if index < len(pairs) - 1
                        else nn.Identity(),
                    ]
                )
                for index, (input_channels, output_channels) in enumerate(pairs)
            ]
        )
        reverse_pairs = list(reversed(pairs[1:]))
        self.up_modules = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        residual(output_channels * 2, input_channels),
                        residual(input_channels, input_channels),
                        Upsample1d(input_channels),
                    ]
                )
                for input_channels, output_channels in reverse_pairs
            ]
        )
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=int(kernel_size)),
            nn.Conv1d(start_dim, self.input_dim, 1),
        )
        logger.info(
            "PaperConditionalUnet1D parameters: %d",
            sum(parameter.numel() for parameter in self.parameters()),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond=None,
    ):
        sample = einops.rearrange(sample, "b h t -> b t h")
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        condition = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            if (
                self.global_cond_dim is not None
                and global_cond.shape[-1] != self.global_cond_dim
            ):
                raise ValueError(
                    f"expected condition width {self.global_cond_dim}, "
                    f"got {global_cond.shape[-1]}"
                )
            condition = torch.cat((condition, global_cond), dim=-1)

        x = sample
        residuals = []
        for first, second, downsample in self.down_modules:
            x = first(x, condition)
            x = second(x, condition)
            residuals.append(x)
            x = downsample(x)
        for block in self.mid_modules:
            x = block(x, condition)
        for first, second, upsample in self.up_modules:
            x = first(torch.cat((x, residuals.pop()), dim=1), condition)
            x = upsample(second(x, condition))
        return einops.rearrange(self.final_conv(x), "b t h -> b h t")


class CrossBlock(nn.Module):
    def __init__(
        self,
        cond_dim,
        hidden_dim,
        n_heads,
        dropout,
        mlp_layers,
        mlp_ratio,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.cmha = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ln3 = nn.LayerNorm(hidden_dim)
        mlp_dim_size = int(hidden_dim * mlp_ratio)
        layers = [nn.Linear(hidden_dim, mlp_dim_size), nn.GELU()]
        for i in range(mlp_layers - 1):
            layers.append(nn.Linear(mlp_dim_size, mlp_dim_size))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(mlp_dim_size, hidden_dim))
        self.mlp = nn.Sequential(*layers)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

    def forward_cross(self, x, cond):
        res = x
        x = self.ln1(x)
        x, _ = self.mha(x, x, x)
        x = x + res
        res = x
        x = self.ln2(x)
        x, _ = self.cmha(x, cond, cond)
        x = x + res
        res = x
        x = self.ln3(x)
        x = self.mlp(x)
        x = x + res
        return x

    def forward(self, x, cond):
        cond = self.cond_proj(cond)
        return self.forward_cross(x, cond)


class CrossTransformer(nn.Module):
    def __init__(
        self,
        nblocks,
        cond_dim,
        hidden_dim,
        act_dim,
        act_seq,
        n_heads,
        dropout,
        mlp_layers,
        mlp_ratio,
        time_conditioning="concat",
        **kwargs,
    ):
        super().__init__()
        if time_conditioning not in {"concat", "additive"}:
            raise ValueError("time_conditioning must be either 'concat' or 'additive'")
        if time_conditioning == "concat" and hidden_dim % 2:
            raise ValueError("concat time conditioning requires an even hidden_dim")
        self.time_conditioning = time_conditioning
        self.layers = nn.ModuleList(
            [
                CrossBlock(
                    cond_dim,
                    hidden_dim,
                    n_heads,
                    dropout,
                    mlp_layers,
                    mlp_ratio,
                    **kwargs,
                )
                for i in range(nblocks)
            ]
        )
        action_embedding_dim = (
            hidden_dim // 2 if time_conditioning == "concat" else hidden_dim
        )
        if action_embedding_dim < act_dim:
            raise ValueError(
                "Rank-deficient denoiser input: action embedding width "
                f"{action_embedding_dim} is smaller than noisy action dimension "
                f"{act_dim}. Increase hidden_dim or use additive time conditioning."
            )
        self.proj_u = nn.Linear(act_dim, action_embedding_dim)
        self.proj_d = nn.Linear(hidden_dim, act_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, act_seq, action_embedding_dim))

    def forward(self, x, timesteps, cond, *args, **kwargs):
        hid_tkns = self.proj_u(x)
        hid_tkns = hid_tkns + self.pos_emb
        time_embed = (
            posemb_sincos(
                timesteps, self.proj_u.out_features, min_period=4e-3, max_period=4.0
            )
            .unsqueeze(1)
            .expand(hid_tkns.shape[0], hid_tkns.shape[1], -1)
            .to(hid_tkns.device)
        )
        if self.time_conditioning == "concat":
            hid_tkns = torch.cat([hid_tkns, time_embed], dim=-1)
        else:
            hid_tkns = hid_tkns + time_embed

        for layer in self.layers:
            hid_tkns = layer(hid_tkns, cond)
        x = self.proj_d(hid_tkns)
        return x
