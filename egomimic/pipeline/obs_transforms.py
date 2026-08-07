"""Train-time obs transforms for the batchflow pipeline.

Each transform is dict -> dict on the per-emb obs dict (applied in
``PipelineAlgo.process_batch_for_training`` AFTER normalization, train only).
"""
import torch


class GaussianImageNoise:
    """Additive Gaussian pixel noise on image keys (input-level augmentation).

    Images arrive as float tensors in [0, 1]; noise is clamped back to range.
    ``std`` is in pixel-intensity units (0.02 = ~5/255).
    """

    def __init__(self, std: float = 0.02, keys=("front_img_1", "front_img_2",
                                                "wrist_img_1")):
        self.std = float(std)
        self.keys = tuple(keys)

    def __call__(self, obs: dict) -> dict:
        for k in self.keys:
            v = obs.get(k)
            if v is not None and torch.is_tensor(v) and v.is_floating_point():
                obs[k] = (v + torch.randn_like(v) * self.std).clamp_(0.0, 1.0)
        return obs
