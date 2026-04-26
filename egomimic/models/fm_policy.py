from typing import Tuple
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.denoising_nets import ConditionalUnet1D, CrossTransformer
from egomimic.models.denoising_policy import DenoisingPolicy

from overrides import override

logger = logging.getLogger(__name__)


class FMPolicy(DenoisingPolicy):
    """
    A diffusion-based policy head.

    Args:
        model (ConditionalUnet1D): The model used for prediction.
        noise_scheduler: The noise scheduler used for the diffusion process.
        action_horizon (int): The number of time steps in the action horizon.
        output_dim (int): The dimension of the output.
        num_inference_steps (int, optional): The number of inference steps.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        model: ConditionalUnet1D,
        action_horizon,
        infer_ac_dims,
        num_inference_steps=None,
        **kwargs,
    ):
        super().__init__(
            model, action_horizon, infer_ac_dims, num_inference_steps, **kwargs
        )
        self.time_dist = kwargs.get("time_dist", "beta")

    def step(self, x_t, t, global_cond):
        if len(t.shape) != 1:
            t = torch.tensor([t], device=global_cond.device)
        v_t = self.model(x_t, t, global_cond)
        return x_t + self.dt * v_t, t + self.dt

    @override
    def inference(self, noise, global_cond, generator=None) -> torch.Tensor:
        self.dt = -1.0 / self.num_inference_steps
        x_t = noise
        time = torch.ones((len(global_cond)), device=global_cond.device)
        while time[0] >= -self.dt / 2:
            x_t, time = self.step(x_t, time, global_cond)
        return x_t

    @override
    def predict(self, actions, global_cond) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn(actions.shape, device=actions.device)
        batch_shape = (actions.shape[0],)
        if self.time_dist == "beta":
            a, b = 1.5, 1.0
            time = torch.distributions.Beta(a, b).sample(batch_shape).to(actions.device)
        elif self.time_dist == "uniform":
            time = (
                torch.distributions.Uniform(0, 1).sample(batch_shape).to(actions.device)
            )
        time = time * 0.999 + 0.001

        time_expanded = time.unsqueeze(-1).unsqueeze(-1)
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        v_t = self.model(x_t, time, global_cond)

        target = u_t
        pred = v_t
        return pred, target


class HierarchicalFMPolicy(DenoisingPolicy):
    """
    Block-head flow matching policy.

    Separate CrossTransformer per action block; each block conditions on detached x0
    predictions from its parent blocks (specified by block_parents).

    block_dims: list[int], must sum to infer_ac_dims[embodiment]
    block_parents: list[list[int]], DAG parents for each block (indices into block_dims)
    cond_dim: int, trunk conditioning dim (e.g. 256)
    block_model_cfg: dict of shared CrossTransformer hyperparams (except act_dim/act_seq)
    model: ignored dummy (nn.Identity); required by DenoisingPolicy base API

    Inference is n_blocks x more expensive than vanilla (sequential per-block decoding at each step).
    """

    def __init__(
        self,
        block_dims,
        block_parents,
        cond_dim,
        block_model_cfg,
        model,
        action_horizon,
        infer_ac_dims,
        num_inference_steps=None,
        **kwargs,
    ):
        super().__init__(model, action_horizon, infer_ac_dims, num_inference_steps, **kwargs)

        assert len(block_parents) == len(block_dims)
        self.block_dims = block_dims
        self.block_parents = block_parents
        self.n_blocks = len(block_dims)
        self.cond_dim = cond_dim
        self.time_dist = kwargs.get("time_dist", "beta")

        self.block_slices = []
        offset = 0
        for d in block_dims:
            self.block_slices.append(slice(offset, offset + d))
            offset += d

        cfg = dict(block_model_cfg)
        self.block_models = nn.ModuleList([
            CrossTransformer(
                act_dim=d,
                act_seq=cfg["act_seq"],
                nblocks=cfg["nblocks"],
                cond_dim=cond_dim,
                hidden_dim=cfg["hidden_dim"],
                n_heads=cfg["n_heads"],
                dropout=cfg["dropout"],
                mlp_layers=cfg["mlp_layers"],
                mlp_ratio=cfg["mlp_ratio"],
            )
            for d in block_dims
        ])

        # x0_proj[i][j]: projects x0 of parent block j into cond_dim for block i
        # stored flat: index = sum(range(i)) + j = i*(i-1)//2 + j  (only for j in parents[i])
        # For simplicity, store as ModuleList of ModuleLists via a flat list with lookup helper.
        self._proj_flat = nn.ModuleList()
        self._proj_index = {}  # (i, j) -> flat index
        for i, parents in enumerate(block_parents):
            for j in parents:
                idx = len(self._proj_flat)
                self._proj_index[(i, j)] = idx
                self._proj_flat.append(nn.Linear(block_dims[j], cond_dim))

        logger.warning(
            f"[HierarchicalFMPolicy] block_heads mode: inference is {self.n_blocks}x more "
            f"expensive per ODE step. block_dims={block_dims}, block_parents={block_parents}"
        )

    def _build_cond(self, trunk_cond, x0_preds, block_idx):
        """Concatenate trunk_cond with projected x0 from parent blocks along seq dim."""
        parts = [trunk_cond]
        for j in self.block_parents[block_idx]:
            proj = self._proj_flat[self._proj_index[(block_idx, j)]]
            parts.append(proj(x0_preds[j]))  # [B, S, cond_dim]
        return torch.cat(parts, dim=1)  # [B, S*(1+len(parents)), cond_dim]

    def _sample_time(self, B, device):
        if self.time_dist == "beta":
            time = torch.distributions.Beta(1.5, 1.0).sample((B,)).to(device)
        else:
            time = torch.distributions.Uniform(0, 1).sample((B,)).to(device)
        return time * 0.999 + 0.001

    def compute_loss(self, global_cond, data):
        actions, global_cond = self.preprocess_compute_loss(global_cond, data)
        B = actions.shape[0]
        noise = torch.randn_like(actions)
        time = self._sample_time(B, actions.device)
        t_exp = time[:, None, None]
        x_t = t_exp * noise + (1 - t_exp) * actions

        x0_preds = [None] * self.n_blocks
        total_loss = actions.new_zeros(())

        for i in range(self.n_blocks):
            sl = self.block_slices[i]
            x_t_i = x_t[..., sl]
            cond_i = self._build_cond(global_cond, x0_preds, i)
            v_i = self.block_models[i](x_t_i, time, cond_i)
            x0_preds[i] = (x_t_i - t_exp * v_i).detach()
            u_i = noise[..., sl] - actions[..., sl]
            total_loss = total_loss + F.mse_loss(v_i, u_i)

        return total_loss

    def inference(self, noise, global_cond, generator=None) -> torch.Tensor:
        dt = -1.0 / self.num_inference_steps
        x_t = noise  # [B, S, act_dim]
        time = torch.ones(len(global_cond), device=global_cond.device)

        while time[0] >= -dt / 2:
            x0_preds = [None] * self.n_blocks
            v_all = []
            t_exp = time[:, None, None]

            for i in range(self.n_blocks):
                sl = self.block_slices[i]
                x_t_i = x_t[..., sl]
                cond_i = self._build_cond(global_cond, x0_preds, i)
                v_i = self.block_models[i](x_t_i, time, cond_i)
                x0_preds[i] = x_t_i - t_exp * v_i
                v_all.append(v_i)

            x_t = x_t + dt * torch.cat(v_all, dim=-1)
            time = time + dt

        return x_t
