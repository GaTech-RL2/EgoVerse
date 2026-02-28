from typing import Tuple

import torch
from overrides import override

from egomimic.models.denoising_nets import ConditionalUnet1D
from egomimic.models.denoising_policy import DenoisingPolicy


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
        encoder_map=None,
        **kwargs,
    ):
        super().__init__(
            model, action_horizon, infer_ac_dims, num_inference_steps, **kwargs
        )
        self.time_dist = kwargs.get("time_dist", "beta")
        self.encoder_map = encoder_map

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
