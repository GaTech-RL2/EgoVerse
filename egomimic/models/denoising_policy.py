from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.denoising_nets import ConditionalUnet1D
from egomimic.rldb.embodiment.embodiment import get_embodiment


class DenoisingPolicy(nn.Module):
    """
    Template class for a diffusion-based policy head.

    Args:
        model (ConditionalUnet1D): The model used for prediction.
        action_horizon (int): The number of time steps in the action horizon.
        infer_ac_dims (dict): Dictionary mapping embodiment names to output dimensions.
        num_inference_steps (int, optional): The number of diffusion inference steps.
        **kwargs: Optional settings like padding, pooling, robot type, etc.
    """

    def __init__(
        self,
        model: ConditionalUnet1D,
        action_horizon: int,
        num_inference_steps: int = None,
        embodiment_specs: dict = None,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.action_horizon = action_horizon
        self.num_inference_steps = num_inference_steps
        self.embodiment_specs = embodiment_specs
        self.codec_enabled = False

        _codecs = {}
        if embodiment_specs is not None:
            for _emb_name, _spec in embodiment_specs.items():
                if _spec.get("encoder") is not None:
                    _codecs[f"{_emb_name}_encoder"] = _spec["encoder"]
                if _spec.get("decoder") is not None:
                    _codecs[f"{_emb_name}_decoder"] = _spec["decoder"]
        if _codecs:
            self.codecs = nn.ModuleDict(_codecs)

        self.padding = kwargs.get("padding", None)
        self.pooling = kwargs.get("pooling", None)

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                print(f"[warn] {name} has requires_grad=False")

        total_params = sum(p.numel() for p in self.model.parameters())
        if self.embodiment_specs is not None:
            for embodiment_name, spec in self.embodiment_specs.items():
                if spec.get("ac_dims") is None:
                    raise ValueError(f"ac_dims must be specified for {embodiment_name}")
            for embodiment_name, spec in self.embodiment_specs.items():
                if spec.get("encoder") is not None:
                    encoder_params = sum(
                        p.numel() for p in spec["encoder"].parameters()
                    )
                    self.codec_enabled = True
                if spec.get("decoder") is not None:
                    decoder_params = sum(
                        p.numel() for p in spec["decoder"].parameters()
                    )
                    self.codec_enabled = True
                print(
                    f"[{embodiment_name}] Encoder params: {encoder_params / 1e6:.2f}M"
                )
                print(
                    f"[{embodiment_name}] Decoder params: {decoder_params / 1e6:.2f}M"
                )
                total_params += encoder_params + decoder_params

        print(
            f"[{self.__class__.__name__}] Total trainable parameters: {total_params / 1e6:.2f}M"
        )

    def preprocess_sampling(self, global_cond, embodiment_name, generator=None):
        if self.pooling == "mean":
            global_cond = global_cond.mean(dim=1)
        elif self.pooling == "flatten":
            global_cond = global_cond.reshape(global_cond.shape[0], -1)

        noise = torch.randn(
            (
                len(global_cond),
                self.action_horizon,
                self.embodiment_specs[embodiment_name].get("ac_dims"),
            ),
            dtype=global_cond.dtype,
            device=global_cond.device,
            generator=generator,
        )
        return noise, global_cond

    def inference(
        self, noise, global_cond, embodiment_name, generator=None
    ) -> torch.Tensor:  # pyright: ignore[reportUnusedParameter]
        """
        To be implemented in subclass: predict actions from noise and conditioning.
        """
        raise NotImplementedError

    def sample_action(self, global_cond, embodiment_name, generator=None):
        noise, global_cond = self.preprocess_sampling(
            global_cond, embodiment_name, generator
        )
        return self.inference(noise, global_cond, embodiment_name, generator)

    def forward(self, global_cond):
        cond, embodiment_name = global_cond
        return self.sample_action(cond, embodiment_name)

    def predict(
        self, actions, global_cond, embodiment_name
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        To be implemented in subclass: returns (prediction, target) given action input and conditioning.
        """
        raise NotImplementedError

    def loss_fn(self, pred, target):
        """
        Computes loss, function to override for stuff like adaptive loss weighting
        """
        return F.mse_loss(pred, target)

    def preprocess_compute_loss(self, global_cond, data, embodiment_name):
        if self.pooling == "mean":
            global_cond = global_cond.mean(dim=1)
        elif self.pooling == "flatten":
            global_cond = global_cond.reshape(global_cond.shape[0], -1)

        actions = data["action"].reshape(len(global_cond), self.action_horizon, -1)

        if self.padding is not None:
            if actions.shape[-1] in {6, 12}:
                pad_shape = (*actions.shape[:-1], 1)
                padding = (
                    torch.randn(pad_shape, device=actions.device)
                    if self.padding == "gaussian"
                    else torch.zeros(pad_shape, device=actions.device)
                )
                if actions.shape[-1] == 6:
                    actions = torch.cat((actions, padding), dim=-1)
                else:  # 12
                    actions = torch.cat(
                        (actions[..., :6], padding, actions[..., 6:], padding), dim=-1
                    )

        return actions, global_cond

    def compute_loss(self, global_cond, data):
        embodiment_name = get_embodiment(data["embodiment"][0].item()).lower()
        actions, global_cond = self.preprocess_compute_loss(
            global_cond, data, embodiment_name
        )
        pred, target = self.predict(actions, global_cond, embodiment_name)
        return self.loss_fn(pred, target)
