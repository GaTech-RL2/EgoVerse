import os

import torch
from lightning import Callback
from lightning.pytorch.utilities import rank_zero_only

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class AugImageLogger(Callback):
    """Periodically dumps what the image branch actually sees after train-time
    augmentation. One PNG per log point (top row: raw batch images, bottom row:
    the same images through model.train_image_augs, un-normalized for display),
    saved under <run_dir>/aug_debug/ and optionally mirrored to WandB.

    WandB-frugal by design: every_n_epochs=100 on a 2000-epoch run = 20 small
    images total. The augmentation is re-applied here with fresh random draws —
    statistically identical to what the forward pass consumes.
    """

    def __init__(
        self,
        every_n_epochs: int = 100,
        n_images: int = 8,
        image_key: str = "obs.aria_image",
        log_wandb: bool = True,
    ):
        self.every_n_epochs = every_n_epochs
        self.n_images = n_images
        self.image_key = image_key
        self.log_wandb = log_wandb

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx != 0 or trainer.current_epoch % self.every_n_epochs != 0:
            return
        augs = getattr(getattr(pl_module, "model", pl_module), "train_image_augs", None)
        for emb, sub in batch.items():
            if not isinstance(sub, dict) or self.image_key not in sub:
                continue
            try:
                self._dump(trainer, sub[self.image_key], augs, str(emb))
            except Exception as e:  # never break training over a debug image
                print(f"[AugImageLogger] skipped ({e})", flush=True)

    def _dump(self, trainer, imgs, augs, emb):
        from torchvision.utils import make_grid, save_image

        raw = imgs[: self.n_images].detach().float().cpu()
        if raw.max() > 1.5:  # uint8-style range
            raw = raw / 255.0
        rows = [raw]
        if augs is not None:
            with torch.no_grad():
                auged = augs(raw)
            # undo the ImageNet Normalize (last transform) for display
            auged = auged * IMAGENET_STD + IMAGENET_MEAN
            rows.append(auged.clamp(0, 1))
        grid = make_grid(torch.cat(rows, dim=0), nrow=self.n_images, padding=2)

        out_dir = os.path.join(trainer.default_root_dir, "aug_debug")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(
            out_dir, f"epoch_{trainer.current_epoch:04d}_{emb}.png"
        )
        save_image(grid, path)

        if self.log_wandb:
            try:
                import wandb

                for logger in trainer.loggers:
                    if hasattr(logger, "experiment") and hasattr(
                        logger.experiment, "log"
                    ):
                        logger.experiment.log(
                            {f"AugDebug/{emb}": wandb.Image(path)},
                            step=trainer.global_step,
                        )
                        break
            except Exception:
                pass
        print(f"[AugImageLogger] wrote {path}", flush=True)
