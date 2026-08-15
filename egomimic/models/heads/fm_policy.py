from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import override

from egomimic.models.diffusion.denoising_nets import ConditionalUnet1D
from egomimic.models.heads.denoising_policy import DenoisingPolicy


class FMPolicyWithVelDecoder(nn.Module):
    """FMPolicy wrapper with a separate MLPVelocityDecoder for the trailing
    velocity token of the arc-tok sequence.

    Waypoint shape (M rows) is denoised by the flow-matching head;
    velocity token (row M) is predicted by a dedicated MLP fed by pooled
    stem/trunk tokens. Both losses (FM MSE on waypoints, direct MSE on vel)
    propagate through the shared trunk+stems.
    """

    def __init__(
        self,
        fm_policy: "FMPolicy",
        vel_decoder: nn.Module,
        num_waypoints: int,
        vel_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.fm_policy = fm_policy
        self.vel_decoder = vel_decoder
        self.num_waypoints = int(num_waypoints)
        self.vel_loss_weight = float(vel_loss_weight)
        self.infer_ac_dims = getattr(fm_policy, "infer_ac_dims", {})
        self.action_horizon = self.num_waypoints + 1

    def _split_domain(self, global_cond):
        if isinstance(global_cond, tuple):
            return global_cond[0], global_cond[1]
        return global_cond, None

    def forward(self, global_cond):
        cond, domain = self._split_domain(global_cond)
        return self.sample_action(cond, domain)

    def sample_action(self, global_cond, embodiment_name, generator=None):
        waypoints = self.fm_policy.sample_action(
            global_cond, embodiment_name, generator=generator
        )  # (B, M, act_dim)
        vel = self.vel_decoder(global_cond, domain=embodiment_name)  # (B, 1, act_dim)
        return torch.cat([waypoints, vel], dim=1)

    def compute_loss(self, global_cond, data):
        cond, domain = self._split_domain(global_cond)
        actions = data["action"]
        B = cond.shape[0]
        if actions.dim() == 2:
            actions = actions.reshape(B, self.num_waypoints + 1, -1)
        wp_target = actions[:, : self.num_waypoints]
        vel_target = actions[:, self.num_waypoints : self.num_waypoints + 1]

        data_wp = dict(data)
        data_wp["action"] = wp_target
        fm_loss = self.fm_policy.compute_loss(cond, data_wp)

        pred_vel = self.vel_decoder(cond, domain=domain)
        vel_loss = F.mse_loss(pred_vel, vel_target)
        return fm_loss + self.vel_loss_weight * vel_loss


class FMPolicyWithVelReadout(nn.Module):
    """Split-decoder arc-tok head that reads the velocity token off a
    dedicated trunk output slot instead of pooling stem tokens through an
    MLP.

    Convention: the HPT trunk is configured with ``action_horizon = M + 1``.
    ``postprocess_tokens`` therefore emits ``(B, M+1, embed_dim)``. This
    wrapper slices:
        shape_ctx = global_cond[:, :-1]      -> FM head cross-attn context
        vel_slot  = global_cond[:, -1]       -> Linear -> (B, 1, act_dim)
    The FM head is configured to denoise M waypoints only.
    """

    def __init__(
        self,
        fm_policy: "FMPolicy",
        num_waypoints: int,
        vel_readout_dim: int,
        act_dim: int,
        vel_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.fm_policy = fm_policy
        self.num_waypoints = int(num_waypoints)
        self.vel_loss_weight = float(vel_loss_weight)
        self.vel_head = nn.Linear(vel_readout_dim, act_dim)
        self.infer_ac_dims = getattr(fm_policy, "infer_ac_dims", {})
        self.action_horizon = self.num_waypoints + 1

    def _split_domain(self, global_cond):
        if isinstance(global_cond, tuple):
            return global_cond[0], global_cond[1]
        return global_cond, None

    def _split_cond(self, cond):
        return cond[:, :-1], cond[:, -1]

    def forward(self, global_cond):
        cond, domain = self._split_domain(global_cond)
        return self.sample_action(cond, domain)

    def sample_action(self, global_cond, embodiment_name, generator=None):
        shape_ctx, vel_slot = self._split_cond(global_cond)
        waypoints = self.fm_policy.sample_action(
            shape_ctx, embodiment_name, generator=generator
        )
        vel = self.vel_head(vel_slot).unsqueeze(1)
        return torch.cat([waypoints, vel], dim=1)

    def compute_loss(self, global_cond, data):
        cond, domain = self._split_domain(global_cond)
        shape_ctx, vel_slot = self._split_cond(cond)

        actions = data["action"]
        B = cond.shape[0]
        if actions.dim() == 2:
            actions = actions.reshape(B, self.num_waypoints + 1, -1)
        wp_target = actions[:, : self.num_waypoints]
        vel_target = actions[:, self.num_waypoints : self.num_waypoints + 1]

        data_wp = dict(data)
        data_wp["action"] = wp_target
        fm_loss = self.fm_policy.compute_loss(shape_ctx, data_wp)

        pred_vel = self.vel_head(vel_slot).unsqueeze(1)
        vel_loss = F.mse_loss(pred_vel, vel_target)
        return fm_loss + self.vel_loss_weight * vel_loss


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
