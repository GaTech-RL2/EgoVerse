from typing import Tuple
import torch

from egomimic.models.denoising_nets import ConditionalUnet1D
from egomimic.models.denoising_policy import DenoisingPolicy

from overrides import override


class DiffusionPolicy(DenoisingPolicy):
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
        noise_scheduler,
        action_horizon,
        infer_ac_dims,
        num_inference_steps=None,
        **kwargs,
    ):
        super().__init__(
            model, action_horizon, infer_ac_dims, num_inference_steps, **kwargs
        )
        self.noise_scheduler = noise_scheduler

    @override
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

    @override
    def predict(self, actions, global_cond) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn(actions.shape, device=actions.device)
        bsz = actions.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (bsz,), device=actions.device
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        pred = self.model(noisy_actions, timesteps, global_cond)
        target = noise
        return pred, target
