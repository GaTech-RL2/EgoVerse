import torch
from torchvision.transforms import Compose


class PerImageCompose:
    """Compose that draws augmentation params independently PER IMAGE when given
    a batched (B,C,H,W) tensor.

    Stock torchvision transforms sample ONE set of random params per call, so
    applying them to a batch shares a single crop/rotation across all images in
    that batch (how train_image_augs has always been applied here). This wrapper
    restores per-sample diversity at the cost of a python loop over the batch.
    """

    def __init__(self, transforms):
        self.transform = Compose(transforms)

    def __call__(self, x):
        if torch.is_tensor(x) and x.ndim == 4:
            return torch.stack([self.transform(img) for img in x])
        return self.transform(x)
