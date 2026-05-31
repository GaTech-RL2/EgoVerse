"""Action-prediction eval for the 2D policy (``SpatialObsActionPolicyDFoTOuterStage``).

Implements the **clean-history -> predict-next-chunk** rollout (teacher-forced):
the first ``n_context_frames`` obs-latent + action tokens are pinned CLEAN from
GT (the observed history + executed actions), and the remaining obs-latent AND
action tokens are denoised jointly — the DiT3D backbone returns
``(v_image, v_action)`` and we run a structured DDIM step on each stream. The
predicted FUTURE actions (steps >= n_context) are compared to GT actions
(action MSE), and the predicted latents are VAE-decoded into a ``[GT|pred]``
video. Future obs are predicted (not GT), so the action chunk conditions only
on the clean past + its own predicted future, not on leaked future frames.

NOTE: state enters as external_cond and is currently supplied from GT for all
steps (teacher-forced); a fully causal closed-loop policy would predict/withhold
future state. This eval verifies the action-token mechanism end to end.
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
import torch

from egomimic.algo.dfot.discrete_diffusion import DiscreteDiffusion
from egomimic.algo.dfot.sampling import vanilla_schedule
from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


def _img_chw_to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    x = np.clip(img_chw.detach().cpu().float().numpy(), 0.0, 1.0)
    return np.transpose((x * 255.0).astype(np.uint8), (1, 2, 0))


class DFoTPolicyActionEval(EvalVideo):
    def __init__(
        self,
        n_context_frames: int = 4,
        rollout_steps: int = 32,
        n_chunk_steps: int = 50,
        embodiment_name: str = "pushshapes_sim",
        image_key: str = "front_img_1",
        recon_loss_n_frames: int = 20,
        upscale_to: int = 384,
        limit_val_batches: int = 4,
        max_videos: int = 2,
        video_subdir: str = "videos_policy_action",
        viz_func=None,
        transform_lists=None,
    ):
        super().__init__(
            limit_val_batches=limit_val_batches, viz_func=viz_func,
            transform_lists=transform_lists, max_videos=max_videos,
        )
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

    def _ddim_from_v(self, diff, x, v, cur, nxt):
        """One eta=0 DDIM step from a v-prediction. ``cur``/``nxt`` are (T,)
        per-token levels; ``x`` is (1, T, *trailing)."""
        T = x.shape[1]
        pad = x.dim() - 2
        kBT = cur.clamp_min(0).long().unsqueeze(0)            # (1, T)
        x0 = diff.predict_start_from_v(x, kBT, v)
        eps = diff.predict_noise_from_v(x, kBT, v)
        an = diff.alphas_cumprod[nxt.clamp_min(0).long()]      # (T,)
        an = torch.where(nxt < 0, torch.ones_like(an), an)
        an = an.reshape(1, T, *([1] * pad))
        c = (1.0 - an).clamp_min(0.0).sqrt()
        return x0 * an.sqrt() + eps * c

    @torch.no_grad()
    def _rollout(self, algo, latent_ctx, action_ctx, cond_seq, T, device):
        """Structured clean-history rollout. latent_ctx (1,n,C,H,W),
        action_ctx (1,n,A), cond_seq (1,T,proj). Returns (pred_latent
        (1,T,C,H,W), pred_action (1,T,A))."""
        outer, diff = algo.outer_stage, algo.diffusion
        n = latent_ctx.shape[1]
        C, H, W = outer.bundle_shape
        A = outer._action_dim
        dts = int(diff.timesteps) if isinstance(diff, DiscreteDiffusion) else None
        sched = vanilla_schedule(self.n_chunk_steps, T, discrete_timesteps=dts).to(device)
        clean = -1 if dts is not None else 0.0
        sched = sched.clone()
        sched[:, :n] = clean

        x_lat = torch.randn(1, T, C, H, W, device=device)
        x_lat[:, :n] = latent_ctx
        x_act = torch.randn(1, T, A, device=device)
        x_act[:, :n] = action_ctx
        for s in range(sched.shape[0] - 1):
            k = sched[s].clamp_min(0).long().unsqueeze(0)      # (1, T)
            v_lat, v_act = algo.backbone(
                x_lat, k, external_cond=cond_seq, action=x_act
            )
            x_lat = self._ddim_from_v(diff, x_lat, v_lat, sched[s], sched[s + 1])
            x_act = self._ddim_from_v(diff, x_act, v_act, sched[s], sched[s + 1])
            x_lat[:, :n] = latent_ctx
            x_act[:, :n] = action_ctx
        return x_lat, x_act

    def compute_metrics_and_viz(self, batch):
        algo = self.model
        metrics: Dict[str, torch.Tensor] = {}
        images: Dict[int, np.ndarray] = {}
        emb_id = get_embodiment_id(self.embodiment_name)
        if emb_id not in batch:
            return metrics, images
        _batch = batch[emb_id]
        if self.image_key not in _batch:
            return metrics, images
        outer = algo.outer_stage
        if not hasattr(outer, "_state_to_cond"):
            return metrics, images  # not the 2D policy
        device = self.trainer.lightning_module.device
        ac_key = algo.resolved_ac_keys[emb_id]
        imgs = _batch[self.image_key]
        is_packed = _batch.get("_packed", False)
        if is_packed:
            cu = _batch["cu_seqlens"].to(imgs.device, dtype=torch.long)
            n = min(int(cu.shape[0] - 1), self.max_videos or 99999)
            starts = [int(cu[i].item()) for i in range(n)]
            lens = [int(cu[i + 1].item() - cu[i].item()) for i in range(n)]
        else:
            n = min(imgs.shape[0], self.max_videos or imgs.shape[0])
            starts = [None] * n
            lens = [imgs.shape[1]] * n

        per_step_sse = np.zeros(self.recon_loss_n_frames)
        per_step_n = np.zeros(self.recon_loss_n_frames, dtype=np.int64)
        all_frames: List[np.ndarray] = []
        for ep in range(n):
            T = min(self.rollout_steps, lens[ep])
            ncx = min(self.n_context_frames, T - 1)
            if is_packed:
                sl = slice(starts[ep], starts[ep] + T)
                img_seq, act_seq = imgs[sl], _batch[ac_key][sl]
                state = torch.cat([_batch[k][sl] for k in outer.bundle_obs_keys], -1)
            else:
                img_seq, act_seq = imgs[ep, :T], _batch[ac_key][ep, :T]
                state = torch.cat([_batch[k][ep, :T] for k in outer.bundle_obs_keys], -1)
            img_seq = img_seq.to(device).float()
            if img_seq.max() > 1.5:
                img_seq = img_seq / 255.0
            act_seq = act_seq.to(device).float()
            cond = outer.state_only_proj(state.to(device).float()).unsqueeze(0)  # (1,T,proj)

            mu, _ = outer.vae.encode(img_seq)
            latent_all = outer.normalize_latent(mu)            # (T,C,H,W)
            latent_ctx = latent_all[:ncx].unsqueeze(0)
            action_ctx = act_seq[:ncx].unsqueeze(0)
            pred_lat, pred_act = self._rollout(algo, latent_ctx, action_ctx, cond, T, device)

            # ---- action MSE on the PREDICTED future (steps >= ncx) ----
            gt_fut = act_seq[ncx:]
            pred_fut = pred_act.squeeze(0)[ncx:]
            m = min(self.recon_loss_n_frames, gt_fut.shape[0])
            if m > 0:
                amse = ((pred_fut[:m] - gt_fut[:m]) ** 2).mean(dim=-1).detach().cpu().numpy()
                for t in range(m):
                    per_step_sse[t] += float(amse[t]); per_step_n[t] += 1

            # ---- decode predicted latents -> [GT|pred] frames ----
            pred_frames = outer.vae.decode(outer.denormalize_latent(pred_lat.squeeze(0)))
            for t in range(pred_frames.shape[0]):
                gt_t = cv2.resize(_img_chw_to_uint8(img_seq[t]), (self.upscale_to, self.upscale_to), interpolation=cv2.INTER_NEAREST)
                pr_t = cv2.resize(_img_chw_to_uint8(pred_frames[t]), (self.upscale_to, self.upscale_to), interpolation=cv2.INTER_NEAREST)
                all_frames.append(np.concatenate([gt_t, pr_t], axis=1))

        for t in range(self.recon_loss_n_frames):
            if per_step_n[t] > 0:
                metrics[f"Valid/emb{emb_id}_action_mse_step_{t:02d}"] = torch.tensor(
                    per_step_sse[t] / per_step_n[t], device=device)
        if per_step_n.sum() > 0:
            metrics[f"Valid/emb{emb_id}_action_mse_first{self.recon_loss_n_frames}"] = torch.tensor(
                per_step_sse.sum() / per_step_n.sum(), device=device)
        if all_frames:
            images[emb_id] = np.stack(all_frames, axis=0)
        return metrics, images
