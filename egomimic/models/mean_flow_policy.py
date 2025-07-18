from typing import Tuple
import torch
from torch.autograd.functional import jvp

from egomimic.models.denoising_nets import ConditionalUnet1D, SinusoidalPosEmb
from egomimic.models.denoising_policy import DenoisingPolicy

from overrides import override
from termcolor import cprint

class MeanFlowPolicy(DenoisingPolicy):
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
        super().__init__(model, action_horizon, infer_ac_dims, num_inference_steps, **kwargs)
        self.time_dist = kwargs.get("time_dist", "lognorm")
        self.flow_ratio = kwargs.get("flow_ratio", 0.25)
        self.adaptive_loss = kwargs.get("adaptive_loss", False)
        if self.num_inference_steps != 1:
            cprint(
                "WARNING: MeanFlowPolicy is recommended with single step inference. Please use single step inference",
                color="yellow",
                attrs=["bold"]
            )

    def sample_t_r(self, batch_size, device):
        if self.time_dist == 'uniform':
            samples = torch.rand(batch_size, 2, device=device)

        elif self.time_dist == 'lognorm':
            mu, sigma = -0.4, 1.0
            dist = torch.distributions.Normal(mu, sigma)
            z = dist.sample((batch_size, 2)).to(device)
            samples = torch.sigmoid(z)
        else:
            raise ValueError(f"Unsupported time_dist: {self.time_dist}")

        # Guarantee t >= r
        t = torch.maximum(samples[:, 0], samples[:, 1])
        r = torch.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        if num_selected > 0:
            indices = torch.randperm(batch_size, device=device)[:num_selected]
            r[indices] = t[indices]

        return t, r

    def step(self, x_t, r, t, global_cond):
        delta_t = (t - r).view(-1, 1, 1)
        u = self.model(x_t, (r, t), global_cond)
        return x_t - delta_t * u

    @override
    def inference(self, noise, global_cond, generator=None) -> torch.Tensor:
        B = noise.shape[0]
        x_t = noise

        if self.num_inference_steps == 1:
            r = torch.zeros(B, device=noise.device)
            t = torch.ones(B, device=noise.device)
            return x_t - self.model(x_t, (r, t), global_cond)

        time_vals = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, device=noise.device)  # shape: (N+1,)
        
        for i in range(self.num_inference_steps):
            t_i = time_vals[i].expand(B)
            r_i = time_vals[i + 1].expand(B)
            x_t = self.step(x_t, r_i, t_i, global_cond)

        return x_t
                
    @override
    def predict(self, actions, global_cond) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn(actions.shape, device=actions.device)
        
        t, r = self.sample_t_r(actions.shape[0], actions.device)

        t_expanded = t.unsqueeze(-1).unsqueeze(-1)
        
        z = t_expanded * noise + (1 - t_expanded) * actions
        v = noise - actions
        
        model_partial = lambda z, r, t : self.model(z, (r, t), global_cond)
        u, dudt = jvp(
            model_partial,
            inputs=(z, r, t),
            v=(v, torch.zeros_like(r), torch.ones_like(t)),
            create_graph=True
        )
        
        u_t = v - (t - r).view(-1, 1, 1) * dudt
        
        target = u_t.detach()
        pred = u
        
        return pred, target

    @override
    def loss_fn(self, pred, target, p=1.0, c=1e-4):
        """
        Implements adaptive squared L2 loss:
        L = sg(w) * ||pred - target||^2
        where w = 1 / (||.||^2 + c)^p
        """
        if self.adaptive_loss:
            delta = pred - target
            l2sq = (delta ** 2).sum(dim=(1, 2))  # (B,)

            weights = 1.0 / (l2sq + c) ** p
            loss = weights.detach() * l2sq

            return loss.mean()
        else:
            return super().loss_fn(pred, target)
