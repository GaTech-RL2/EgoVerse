"""Anchored clean-history rollout eval for the 1D obs-action(+image) DFoT.

Flat bundle ``[state | vae_mu_flat | action]``. Seeds the first
``n_context_frames`` bundles from GT (state + frozen-VAE mu + executed action),
pins them clean, and predicts the rest with a single-tensor anchored DDIM
rollout. Decodes the latent slice -> ``[GT|pred]`` video; extracts the action
slice -> per-step action MSE on the predicted future (policy variant). For the
world-model variant (``action_in_bundle=False``) the action is fed as
``external_cond`` and only video MSE is reported.
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
import torch

from egomimic.algo.dfot.discrete_diffusion import DiscreteDiffusion
from egomimic.algo.dfot.sampling import sample as _sample, sample_step, vanilla_schedule
from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


def _u8(img_chw: torch.Tensor) -> np.ndarray:
    x = np.clip(img_chw.detach().cpu().float().numpy(), 0.0, 1.0)
    return np.transpose((x * 255.0).astype(np.uint8), (1, 2, 0))


class DFoTBundleAnchoredEval(EvalVideo):
    def __init__(
        self, n_context_frames: int = 4, rollout_steps: int = 32,
        n_chunk_steps: int = 50, embodiment_name: str = "pushshapes_sim",
        image_key: str = "front_img_1", recon_loss_n_frames: int = 20,
        upscale_to: int = 384, limit_val_batches: int = 4, max_videos: int = 2,
        video_subdir: str = "videos_bundle_anchored", viz_func=None,
        transform_lists=None,
    ):
        super().__init__(limit_val_batches=limit_val_batches, viz_func=viz_func,
                         transform_lists=transform_lists, max_videos=max_videos)
        self.n_context_frames = int(n_context_frames)
        self.rollout_steps = int(rollout_steps)
        self.n_chunk_steps = int(n_chunk_steps)
        self.embodiment_name = embodiment_name
        self.image_key = str(image_key)
        self.recon_loss_n_frames = int(recon_loss_n_frames)
        self.upscale_to = int(upscale_to)
        self._video_subdir = str(video_subdir)

    def video_dir(self):
        import os
        return os.path.join(self.root_dir(), self._video_subdir)

    @torch.no_grad()
    def _rollout(self, algo, ctx_bundle, cond_seq, T, device):
        outer, diff = algo.outer_stage, algo.diffusion
        n = ctx_bundle.shape[1]
        dts = int(diff.timesteps) if isinstance(diff, DiscreteDiffusion) else None
        sched = vanilla_schedule(self.n_chunk_steps, T, discrete_timesteps=dts).to(device)
        clean = -1 if dts is not None else 0.0
        sched = sched.clone(); sched[:, :n] = clean
        x = torch.randn(1, T, outer.bundle_dim, device=device)
        x[:, :n] = ctx_bundle
        for s in range(sched.shape[0] - 1):
            x = sample_step(diff, algo.backbone, x=x, current_levels=sched[s],
                            next_levels=sched[s + 1], external_cond=cond_seq, eta=0.0)
            x[:, :n] = ctx_bundle
        return x

    def compute_metrics_and_viz(self, batch):
        algo = self.model
        metrics: Dict[str, torch.Tensor] = {}
        images: Dict[int, np.ndarray] = {}
        emb_id = get_embodiment_id(self.embodiment_name)
        if emb_id not in batch:
            return metrics, images
        _batch = batch[emb_id]
        outer = algo.outer_stage
        if not hasattr(outer, "image_latent_slice") or self.image_key not in _batch:
            return metrics, images
        device = self.trainer.lightning_module.device
        ac_key = algo.resolved_ac_keys[emb_id]
        in_bundle = bool(getattr(outer, "action_in_bundle", True))
        imgs = _batch[self.image_key]
        is_packed = _batch.get("_packed", False)
        if is_packed:
            cu = _batch["cu_seqlens"].to(imgs.device, dtype=torch.long)
            n = min(int(cu.shape[0] - 1), self.max_videos or 99999)
            spans = [(int(cu[i].item()), int(cu[i + 1].item())) for i in range(n)]
        else:
            n = min(imgs.shape[0], self.max_videos or imgs.shape[0])
            spans = [(i, None) for i in range(n)]

        a_sse = np.zeros(self.recon_loss_n_frames); a_n = np.zeros(self.recon_loss_n_frames, dtype=np.int64)
        v_sse = np.zeros(self.recon_loss_n_frames); v_n = np.zeros(self.recon_loss_n_frames, dtype=np.int64)
        all_frames: List[np.ndarray] = []
        for (s0, s1) in spans:
            if is_packed:
                Lep = s1 - s0; T = min(self.rollout_steps, Lep); sl = slice(s0, s0 + T)
                pick = lambda k: _batch[k][sl]
            else:
                Lep = imgs.shape[1]; T = min(self.rollout_steps, Lep)
                pick = lambda k: _batch[k][s0, :T]
            ncx = min(self.n_context_frames, T - 1)
            img_seq = pick(self.image_key).to(device).float()
            if img_seq.max() > 1.5: img_seq = img_seq / 255.0
            act_seq = pick(ac_key).to(device).float()
            state = torch.cat([pick(k).to(device).float() for k in outer.real_bundle_obs_keys], -1)
            mu, _ = outer.vae.encode(img_seq)
            mu_flat = mu.flatten(1)

            # context bundle for first ncx steps
            pieces = [state[:ncx], mu_flat[:ncx]]
            if in_bundle: pieces.append(act_seq[:ncx])
            ctx_bundle = torch.cat(pieces, -1).unsqueeze(0)
            cond_seq = None if in_bundle else act_seq.unsqueeze(0)

            pred = self._rollout(algo, ctx_bundle, cond_seq, T, device).squeeze(0)  # (T, bundle_dim)
            lat = pred[:, outer.image_latent_slice].reshape(T, *outer.latent_shape)
            pred_frames = outer.vae.decode(lat)

            # video MSE
            m = min(self.recon_loss_n_frames, T)
            vmse = ((pred_frames[:m] - img_seq[:m]) ** 2).mean(dim=(1, 2, 3)).detach().cpu().numpy()
            for t in range(m): v_sse[t] += float(vmse[t]); v_n[t] += 1
            # action MSE on predicted future
            if in_bundle:
                pa = pred[:, outer.action_slice]
                gf = act_seq[ncx:]; pf = pa[ncx:]
                ma = min(self.recon_loss_n_frames, gf.shape[0])
                amse = ((pf[:ma] - gf[:ma]) ** 2).mean(dim=-1).detach().cpu().numpy()
                for t in range(ma): a_sse[t] += float(amse[t]); a_n[t] += 1

            for t in range(pred_frames.shape[0]):
                g = cv2.resize(_u8(img_seq[t]), (self.upscale_to,) * 2, interpolation=cv2.INTER_NEAREST)
                p = cv2.resize(_u8(pred_frames[t]), (self.upscale_to,) * 2, interpolation=cv2.INTER_NEAREST)
                all_frames.append(np.concatenate([g, p], axis=1))

        for t in range(self.recon_loss_n_frames):
            if v_n[t] > 0:
                metrics[f"Valid/emb{emb_id}_video_mse_step_{t:02d}"] = torch.tensor(v_sse[t] / v_n[t], device=device)
            if a_n[t] > 0:
                metrics[f"Valid/emb{emb_id}_action_mse_step_{t:02d}"] = torch.tensor(a_sse[t] / a_n[t], device=device)
        if v_n.sum() > 0:
            metrics[f"Valid/emb{emb_id}_video_mse_first{self.recon_loss_n_frames}"] = torch.tensor(v_sse.sum() / v_n.sum(), device=device)
        if a_n.sum() > 0:
            metrics[f"Valid/emb{emb_id}_action_mse_first{self.recon_loss_n_frames}"] = torch.tensor(a_sse.sum() / a_n.sum(), device=device)
        if all_frames:
            images[emb_id] = np.stack(all_frames, axis=0)
        return metrics, images
