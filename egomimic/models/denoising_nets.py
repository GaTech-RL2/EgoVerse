import math

from typing import Union, Tuple
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import math

logger = logging.getLogger(__name__)

def posemb_sincos(
  pos: torch.Tensor,               # shape: (B,)
  embedding_dim: int,
  min_period: float,
  max_period: float
) -> torch.Tensor:                 # returns (B, embedding_dim)
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
  fraction = torch.linspace(0.0, 1.0, embedding_dim // 2,
                            device=pos.device, dtype=pos.dtype)
  period = min_period * (max_period / min_period) ** fraction

  # (B, embedding_dim // 2)
  sinusoid_input = torch.einsum(
    "i,j->ij",
    pos,
    (1.0 / period) * 2 * torch.pi
  )

  return torch.cat([torch.sin(sinusoid_input),
                    torch.cos(sinusoid_input)], dim=-1)

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

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

def test():
    cb = Conv1dBlock(256, 128, kernel_size=3)
    x = torch.zeros((1, 256, 16))
    o = cb(x)

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
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8, cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
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
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

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


class ConditionalConcatResidualBlock1D(nn.Module):
    """very naive feature concatenations instead of a film layer with residual connections"""

    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8, cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels + cond_dim, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        self.out_channels = out_channels

    def forward(self, x, cond, *args, **kwargs):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        cond = cond[:, :, None].repeat(1, 1, x.shape[-1])
        x = torch.cat((x, cond), dim=1)
        out = self.blocks[0](x)
        out = self.blocks[1](out)
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
        feature_concatenate=False,
        **kwargs,
    ):
        """
        local conditioning and global conditioning scheme
        """
        super().__init__()
        
        self.mean_flow = kwargs.get("mean_flow", False)
        
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim

        if self.mean_flow:
            self.t_emb = SinusoidalPosEmb(dsed // 2)
            self.dt_emb = SinusoidalPosEmb(dsed // 2)
            diffusion_step_encoder = nn.Sequential(
                nn.Linear(dsed, dsed * 2),  # t_emb || dt_emb → dsed
                nn.Mish(),
                nn.Linear(dsed * 2, dsed // 2),
            )
        else:
            diffusion_step_encoder = nn.Sequential(
                SinusoidalPosEmb(dsed // 2),
                nn.Linear(dsed // 2, dsed * 2),
                nn.Mish(),
                nn.Linear(dsed * 2, dsed // 2),
            )
            
        self.proj_cond = nn.Linear(ac_latent_seq * cond_dim, dsed // 2)

        cond_dim = dsed

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        layer_func = ConditionalResidualBlock1D if not feature_concatenate else ConditionalConcatResidualBlock1D

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
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int, Tuple[torch.Tensor, torch.Tensor]],
        global_cond,
        *args,
        **kwargs
    ):
        """
        sample: (B,T,input_dim)
        timestep: (B,) or int, diffusion step or (r, t) tuple, each of shape (B, )
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, "b h t -> b t h")

        # 1. time
        if self.mean_flow:
            r, t = timestep   # (B,), (B,)
            dt = t - r
            t_emb = self.t_emb(t)
            dt_emb = self.dt_emb(dt)
            time_embed = torch.cat([t_emb, dt_emb], dim=-1)  # (B, D)
            global_feature = self.diffusion_step_encoder(time_embed)
            
        else:
            timesteps = timestep
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)

            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)

            timesteps = timesteps.expand(sample.shape[0])
            global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            cond = self.proj_cond(global_cond)
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
        return x



class ConditionalClassifier1D(nn.Module):
    def __init__(
        self,
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        ac_latent_seq=8,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ):
        """
        local conditioning and global conditioning scheme
        """
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed * 2
        self.proj_cond = nn.Linear(cond_dim * ac_latent_seq, dsed)
        
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        self.resnet = ConditionalResidualBlock1D(
            input_dim,
            start_dim,
            cond_dim=cond_dim,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, 1, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.final_conv = final_conv

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        *args,
        local_cond=None,
        global_cond=None,
        **kwargs
    ):
        """
        sample: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, "b h t -> b t h")

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            # 512 + 39
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = self.resnet(sample, global_feature)
        x = self.final_conv(x)

        x = einops.rearrange(x, "b t h -> b h t")
        return x


class ConditionalClassifier1D(nn.Module):
    def __init__(
        self,
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ):
        """
        local conditioning and global conditioning scheme
        """
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        self.resnet = ConditionalResidualBlock1D(
            input_dim,
            start_dim,
            cond_dim=cond_dim,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, 1, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.final_conv = final_conv

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        local_cond=None,
        global_cond=None,
        **kwargs
    ):
        """
        sample: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """

        sample = einops.rearrange(sample, "b h t -> b t h")

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = self.resnet(sample, global_feature)
        x = self.final_conv(x)
        x = einops.rearrange(x, "b t h -> b h t")
        return x

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, cond_dim, hidden_dim, n_layers, shift=True):
        """
        Args:
            cond_dim (int): Flattened conditioning dim (D1)
            hidden_dim (int): Token dim (D)
            n_layers (int): Number of transformer layers
            shift (bool): Whether to output shift along with scale
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.shift = shift

        self.cond_proj = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
        )

        out_dim = hidden_dim * n_layers * (2 if shift else 1)

        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, out_dim)
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, timestep, cond, layer_idx):
        """
        Args:
            x        : (B, T, D)
            timestep : (B, T, D)
            cond     : (B, cond_dim)
            layer_idx : int in [0, n_layers)

        Returns:
            x_modulated: (B, T, D)
        """
        B, T, D = x.shape
        assert timestep.shape == x.shape, "timestep and x must have same shape"
        assert cond.shape[0] == B

        cond_proj = self.cond_proj(cond)  # (B, D)
        cond_proj = cond_proj[:, None, :].expand(B, T, D)  # (B, T, D)

        h = timestep + cond_proj  # (B, T, D)

        mod = self.mlp(h)  # (B, T, n_layers * D * [1 or 2])

        if self.shift:
            mod = mod.view(B, T, self.n_layers, 2, D)
            shift = mod[:, :, layer_idx, 0, :]  # (B, T, D)
            scale = mod[:, :, layer_idx, 1, :]  # (B, T, D)
        else:
            mod = mod.view(B, T, self.n_layers, D)
            shift = None
            scale = mod[:, :, layer_idx, :]     # (B, T, D)

        x_norm = F.layer_norm(x, (D,), eps=1e-5)
        x_mod = x_norm * (1 + scale)
        if self.shift:
            x_mod = x_mod + shift
        return x_mod

class DiTBlock(nn.Module):
    def __init__(
        self,
        cond_dim,
        hidden_dim,
        n_heads,
        dropout,
        mlp_layers,
        mlp_ratio,
        ad_nlayers,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.aln1 = AdaptiveLayerNorm(cond_dim, hidden_dim, ad_nlayers)

        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.alns1 = AdaptiveLayerNorm(cond_dim, hidden_dim, ad_nlayers, shift=False)

        self.ln2 = nn.LayerNorm(hidden_dim)
        self.aln2 = AdaptiveLayerNorm(cond_dim, hidden_dim, ad_nlayers)

        mlp_dim_size = int(hidden_dim * mlp_ratio)
        mlp_layers_list = [nn.Linear(hidden_dim, mlp_dim_size), nn.GELU()]
        for _ in range(mlp_layers - 1):
            mlp_layers_list.append(nn.Linear(mlp_dim_size, mlp_dim_size))
            mlp_layers_list.append(nn.GELU())
        mlp_layers_list.append(nn.Linear(mlp_dim_size, hidden_dim))
        self.mlp = nn.Sequential(*mlp_layers_list)

        self.alns2 = AdaptiveLayerNorm(cond_dim, hidden_dim, ad_nlayers)

    def forward(self, x, cond, timestep, layer_idx):
        """
        Args:
            x         : (B, T, D)
            cond      : (B, cond_dim)
            timestep  : (B, T, D)
            layer_idx : int, required
        """
        B, T, D = x.shape
        assert timestep.shape == x.shape
        assert cond.shape[0] == B
        assert layer_idx is not None

        res = x
        x = self.ln1(x)
        x = self.aln1(x, timestep, cond, layer_idx)
        x, _ = self.mha(x, x, x)
        x = self.alns1(x, timestep, cond, layer_idx)
        x = x + res

        res = x
        x = self.ln2(x)
        x = self.aln2(x, timestep, cond, layer_idx)
        x = self.mlp(x)
        x = self.alns2(x, timestep, cond, layer_idx) + res

        return x

class CrossBlock(nn.Module):
    def __init__(
        self,
        cond_dim,
        hidden_dim,
        n_heads,
        dropout,
        mlp_layers,
        mlp_ratio,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.cmha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
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

class CrossBlockCfg2(CrossBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
    def forward(self, x, cond, time):
        time = self.time_proj(time)
        cond = self.cond_proj(cond)
        cond_time = torch.cat([cond, time], dim=-2)
        return super().forward_cross(x, cond_time)
    
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
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([CrossBlock(cond_dim, hidden_dim, n_heads, dropout, mlp_layers, mlp_ratio, **kwargs) for i in range(nblocks)])
        self.proj_u = nn.Linear(act_dim, hidden_dim//2)
        self.proj_d = nn.Linear(hidden_dim, act_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, act_seq, hidden_dim// 2))
        self.mean_flow = kwargs.get("mean_flow", False)            


    def forward(self, x, timesteps, cond):
        hid_tkns = self.proj_u(x)
        hid_tkns = hid_tkns + self.pos_emb
        if self.mean_flow:
            r, t = timesteps
            dt = t - r
            t_embed = posemb_sincos(t, self.proj_u.out_features // 2, min_period=4e-3, max_period=4.0)
            dt_embed = posemb_sincos(dt, self.proj_u.out_features // 2, min_period=4e-3, max_period=4.0)
            time_embed = torch.cat([t_embed, dt_embed], dim=-1)
        else:
            time_embed = posemb_sincos(timesteps, self.proj_u.out_features, min_period=4e-3, max_period=4.0)

        time_embed = time_embed.unsqueeze(1).expand(hid_tkns.shape[0], hid_tkns.shape[1], -1).to(hid_tkns.device)
        hid_tkns = torch.cat([hid_tkns, time_embed], dim=-1)

        for layer in self.layers:
            hid_tkns = layer(hid_tkns, cond)

        x = self.proj_d(hid_tkns)
        return x

class CrossTransformerCfg2(nn.Module):
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
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([CrossBlockCfg2(cond_dim, hidden_dim, n_heads, dropout, mlp_layers, mlp_ratio, **kwargs) for i in range(nblocks)])
        self.proj_u = nn.Linear(act_dim, hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, act_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, act_seq, hidden_dim))


    def forward(self, x, timesteps, cond, *args, **kwargs):
        hid_tkns = self.proj_u(x)
        hid_tkns = hid_tkns + self.pos_emb
        time_embed = posemb_sincos(timesteps, self.proj_u.out_features, min_period=4e-3, max_period=4.0).unsqueeze(1).expand(hid_tkns.shape[0], hid_tkns.shape[1], -1).to(hid_tkns.device) # (1, S, D)

        for layer in self.layers:
            hid_tkns = layer(hid_tkns, cond, time_embed)
        x = self.proj_d(hid_tkns)
        return x

class DiffusionTransformer(nn.Module):
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
        **kwargs
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.nblocks = nblocks
        self.n_heads = n_heads
        self.dropout = dropout
        self.mlp_layers = mlp_layers
        self.mlp_ratio = mlp_ratio
        self.pos_emb = nn.Parameter(torch.zeros(1, act_seq, hidden_dim))
        self.mean_flow = kwargs.get("mean_flow", False)            

        self.proj_u = nn.Linear(act_dim, hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, act_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.layers = nn.ModuleList([
            DiTBlock(
                cond_dim=cond_dim,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                dropout=dropout,
                mlp_layers=mlp_layers,
                mlp_ratio=mlp_ratio,
                ad_nlayers=kwargs.get("ad_nlayers", nblocks),
            )
            for _ in range(nblocks)
        ])

    def forward(self, x, timesteps, cond):
        hid_tokens = self.proj_u(x) + self.pos_emb  # (B, T, hidden_dim)

        if self.mean_flow:
            r, t = timesteps
            r_emb = posemb_sincos(r, self.hidden_dim, 4e-3, 4.0)
            t_emb = posemb_sincos(t, self.hidden_dim, 4e-3, 4.0)
        else:
            t = timesteps
            t_emb = posemb_sincos(t, self.hidden_dim, 4e-3, 4.0)
            r_emb = torch.zeros_like(t_emb)

        time_feat = self.time_mlp(r_emb) + self.time_mlp(t_emb)
        time_embed = time_feat.unsqueeze(1).expand(hid_tokens.shape[0], hid_tokens.shape[1], -1)

        for i, layer in enumerate(self.layers):
            hid_tokens = layer(hid_tokens, cond, time_embed, layer_idx=i)

        return self.proj_d(hid_tokens)


class CrossTransformerProj(nn.Module):
    def __init__(
        self,
        nblocks,
        cond_dim,
        hidden_dim,
        act_dim_robot,
        act_dim_human,
        act_seq,
        n_mlp_proj,
        n_heads,
        dropout,
        mlp_layers,
        mlp_ratio,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([CrossBlock(cond_dim, hidden_dim, n_heads, dropout, mlp_layers, mlp_ratio, **kwargs) for i in range(nblocks)])
        self.proj_u_robot = nn.Linear(act_dim_robot, hidden_dim//2)
        self.proj_d_robot = nn.Linear(hidden_dim, act_dim_robot)
        self.proj_u_human = nn.Linear(act_dim_human, hidden_dim//2)
        self.proj_d_human = nn.Linear(hidden_dim, act_dim_human)
        self.n_mlp_proj = n_mlp_proj
        self.hidden_dim = hidden_dim
        if self.n_mlp_proj > 0:
            h_mlp_u = [nn.GELU(), nn.Linear(hidden_dim//2, hidden_dim//2)]
            r_mlp_u = [nn.GELU(), nn.Linear(hidden_dim//2, hidden_dim//2)]
            h_mlp_d = [nn.GELU(), nn.Linear(hidden_dim, hidden_dim)]
            r_mlp_d = [nn.GELU(), nn.Linear(hidden_dim, hidden_dim)]
            for i in range(n_mlp_proj - 1):
                h_mlp_u.append(nn.GELU())
                h_mlp_u.append(nn.Linear(hidden_dim//2, hidden_dim//2))
                r_mlp_u.append(nn.GELU())
                r_mlp_u.append(nn.Linear(hidden_dim//2, hidden_dim//2))

                h_mlp_d.append(nn.GELU())
                h_mlp_d.append(nn.Linear(hidden_dim, hidden_dim))
                r_mlp_d.append(nn.GELU())
                r_mlp_d.append(nn.Linear(hidden_dim, hidden_dim))
            
            self.h_mlp_u = nn.Sequential(*h_mlp_u)
            self.r_mlp_u = nn.Sequential(*r_mlp_u)
            self.h_mlp_d = nn.Sequential(*h_mlp_d)
            self.r_mlp_d = nn.Sequential(*r_mlp_d)

        self.pos_emb = nn.Parameter(torch.zeros(1, act_seq, hidden_dim// 2))

    def forward(self, x, timesteps, cond, robot, *args, **kwargs):
        if robot:
            hid_tkns = self.proj_u_robot(x)
            if self.n_mlp_proj > 0:
                hid_tkns = self.r_mlp_u(hid_tkns)
        else:
            hid_tkns = self.proj_u_human(x)
            if self.n_mlp_proj > 0:
                hid_tkns = self.h_mlp_u(hid_tkns)
        hid_tkns = hid_tkns + self.pos_emb
        time_embed = posemb_sincos(timesteps, self.hidden_dim//2, min_period=4e-3, max_period=4.0).unsqueeze(1).expand(hid_tkns.shape[0], hid_tkns.shape[1], -1).to(hid_tkns.device) # (1, S, D)
        hid_tkns = torch.cat([hid_tkns, time_embed], dim=-1)
        
        # if hasattr(timesteps, "shape") and len(timesteps.shape) > 0 and timesteps.shape[0] == 1:
        #     breakpoint()
        for layer in self.layers:
            hid_tkns = layer(hid_tkns, cond)
        if robot:
            if self.n_mlp_proj > 0:
                hid_tkns = self.r_mlp_d(hid_tkns)
            x = self.proj_d_robot(hid_tkns)
        else:
            if self.n_mlp_proj > 0:
                hid_tkns = self.h_mlp_d(hid_tkns)
            x = self.proj_d_human(hid_tkns)
        return x
    
