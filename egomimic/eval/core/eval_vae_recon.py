"""VAE reconstruction evaluator.

Per val pass: run a few episodes' worth of frames through the VAE
(``encode`` -> posterior mean -> ``decode``) and emit one mp4 per
embodiment showing GT vs reconstruction side-by-side. Tests whether
the frozen VAE we'd hand to the downstream diffusion model can
actually represent these images.

Output: ``<output_dir>/videos/epoch_N/<embodiment>/validation_video_N.mp4``
each frame is ``[GT | recon]`` width-concatenated.
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np
import torch

from egomimic.eval.core.eval_video import EvalVideo
from egomimic.eval.core.img_utils import img_chw_to_uint8
from egomimic.rldb.embodiment.embodiment import get_embodiment_id



class VAEReconEval(EvalVideo):
    """Side-by-side GT | reconstruction video.

    Args:
        image_key: which obs key carries the image stream. Default
            ``"front_img_1"``.
        max_frames: cap on number of frames rendered per val pass.
        upscale_to: per-panel side after upscaling. The output frame
            width is 2*upscale_to.
        limit_val_batches: per-pass cap on val batches.
        embodiment_name: which embodiment to expect in the batch.
    """

    def __init__(
        self,
        image_key: str = "front_img_1",
        max_frames: int = 256,
        upscale_to: int = 256,
        limit_val_batches: int = 1,
        embodiment_name: str = "pushshapes_sim",
        viz_func=None,
        transform_lists=None,
    ):
        super().__init__(
            limit_val_batches=limit_val_batches,
            viz_func=viz_func,
            transform_lists=transform_lists,
            max_videos=None,
        )
        self.image_key = str(image_key)
        self.max_frames = int(max_frames)
        self.upscale_to = int(upscale_to)
        self.embodiment_name = embodiment_name

    def compute_metrics_and_viz(self, batch):
        algo = self.model
        metrics: Dict[str, torch.Tensor] = {}
        images_dict: Dict[int, np.ndarray] = {}

        device = self.trainer.lightning_module.device
        emb_id = get_embodiment_id(self.embodiment_name)
        if emb_id not in batch:
            return metrics, images_dict
        _batch = batch[emb_id]
        if "images" not in _batch:
            return metrics, images_dict

        x = _batch["images"][: self.max_frames]
        with torch.no_grad():
            mu, _logvar = algo.nets["vae"].encode(x.to(device))
            x_rec = algo.nets["vae"].decode(mu)

        # Per-frame MSE in pixel space, reported as a scalar.
        recon_mse = float(((x_rec - x.to(device)) ** 2).mean().item())
        metrics[f"Valid/emb{emb_id}_recon_mse"] = torch.tensor(recon_mse, device=device)

        # Stack [GT | recon] frame-by-frame.
        frames = []
        for i in range(x.shape[0]):
            gt = img_chw_to_uint8(x[i])
            rc = img_chw_to_uint8(x_rec[i])
            gt = cv2.resize(
                gt,
                (self.upscale_to, self.upscale_to),
                interpolation=cv2.INTER_NEAREST,
            )
            rc = cv2.resize(
                rc,
                (self.upscale_to, self.upscale_to),
                interpolation=cv2.INTER_NEAREST,
            )
            frames.append(np.concatenate([gt, rc], axis=1))

        if frames:
            images_dict[emb_id] = np.stack(frames, axis=0)
        return metrics, images_dict
