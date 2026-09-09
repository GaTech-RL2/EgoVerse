"""Diffusion process used by factorized pipeline stages."""

from __future__ import annotations

import torch
import torch.nn as nn

from egomimic.models.denoising_nets import ConditionalUnet1D


class DiffusionPolicy(nn.Module):
    """A single-width diffusion process with no batch-routing concerns.

    Args:
        model (ConditionalUnet1D): The model used for prediction.
        noise_scheduler: The noise scheduler used for the diffusion process.
        action_horizon (int): The number of time steps in the action horizon.
        num_inference_steps (int, optional): The number of inference steps.
    """

    def __init__(
        self,
        model: ConditionalUnet1D,
        noise_scheduler,
        action_horizon: int,
        num_inference_steps: int,
    ):
        super().__init__()
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")

    def inference(self, noise, global_cond, generator=None) -> torch.Tensor:
        self.noise_scheduler.set_timesteps(
            self.num_inference_steps, device=global_cond.device
        )
        actions = noise
        for t in self.noise_scheduler.timesteps:
            t_model = (
                torch.tensor([t], device=global_cond.device) if len(t.shape) != 1 else t
            )
            model_output = self.model(actions, t_model, global_cond)
            actions = self.noise_scheduler.step(
                model_output, t, actions, generator=generator
            ).prev_sample
        return actions

    def sample_action(
        self,
        global_cond: torch.Tensor,
        *,
        action_dim: int,
        generator=None,
    ) -> torch.Tensor:
        """Sample one fixed-width sequence for the supplied conditioning."""
        action_dim = int(action_dim)
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        noise = torch.randn(
            len(global_cond),
            self.action_horizon,
            action_dim,
            dtype=global_cond.dtype,
            device=global_cond.device,
            generator=generator,
        )
        return self.inference(noise, global_cond, generator)
