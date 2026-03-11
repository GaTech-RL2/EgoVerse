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
        num_inference_steps=None,
        embodiment_specs=None,
        **kwargs,
    ):
        super().__init__(
            model, action_horizon, num_inference_steps, embodiment_specs, **kwargs
        )
        self.time_dist = kwargs.get("time_dist", "beta")
        self.dt = -1.0 / self.num_inference_steps

    def step(self, x_t, t, global_cond, embodiment_name):
        if len(t.shape) != 1:
            t = torch.tensor([t], device=global_cond.device)
        v_t = self.denoising_model(x_t, t, global_cond, embodiment_name)
        return x_t + self.dt * v_t, t + self.dt

    @override
    def inference(
        self, noise, global_cond, embodiment_name, generator=None
    ) -> torch.Tensor:
        self.dt = -1.0 / self.num_inference_steps
        x_t = noise
        time = torch.ones((len(global_cond)), device=global_cond.device)
        while time[0] >= -self.dt / 2:
            x_t, time = self.step(x_t, time, global_cond, embodiment_name)
        return x_t

    @override
    def predict(
        self, actions, global_cond, embodiment_name
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        v_t = self.denoising_model(x_t, time, global_cond, embodiment_name)

        target = u_t
        pred = v_t
        return pred, target

    def denoising_model(self, x_t, time, global_cond, embodiment_name):
        if self.codec_enabled:
            x_t = self.embodiment_specs[embodiment_name]["encoder"](x_t)
        else:
            x_t = x_t
        v_t = self.model(x_t, time, global_cond)
        if self.codec_enabled:
            v_t = self.embodiment_specs[embodiment_name]["decoder"](v_t)
        else:
            v_t = v_t
        return v_t


if __name__ == "__main__":
    import hydra
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(
        "/coc/flash7/paphiwetsa3/projects/EgoVerse/egomimic/hydra_configs/model/hpt_cotrain_flow_shared_head_latent.yaml"
    )
    model = hydra.utils.instantiate(cfg.robomimic_model.head_specs.shared)

    # test the model
    aria_input = torch.randn(8, 100, 140)
    global_cond = torch.randn(8, 64, 256)
    aria_output = model.step(
        aria_input, torch.tensor([0.0]), global_cond, "aria_bimanual"
    )
    aria_output_inference = model.inference(aria_input, global_cond, "aria_bimanual")
    aria_output_predict = model.predict(aria_input, global_cond, "aria_bimanual")

    eva_input = torch.randn(8, 100, 14)
    eva_output = model.step(eva_input, torch.tensor([0.0]), global_cond, "eva_bimanual")
    eva_output_inference = model.inference(eva_input, global_cond, "eva_bimanual")
    eva_output_predict = model.predict(eva_input, global_cond, "eva_bimanual")
    breakpoint()
