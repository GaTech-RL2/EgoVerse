"""Receding-horizon action eval for the 2D policy.

The parent ``DFoTPolicyActionEval`` denoises the WHOLE T=32 future action+latent
chunk jointly in one pass — that is the policy's *worst case* (28 future action
tokens co-denoised under a shared per-step noise level), and its non-monotonic
per-step curve is an artifact of that joint long-chunk coupling plus n=2
sampling noise, NOT the policy's deployable accuracy.

This eval measures the DEPLOYMENT metric: slide an anchor across the whole
episode; at each anchor feed CLEAN GT obs-latent + action history of length
``n_context`` and predict ONLY the next ``k`` actions via a tiny
``T = n_context + k`` rollout (matching how a receding-horizon controller runs:
predict next chunk, execute, re-anchor on the observed history). Score those k
predicted actions against GT in normalized action space, pooled over ALL anchors
and ``max_episodes`` episodes — orders of magnitude more samples than the old
2-episode single-chunk read.

Headline: ``rh_k1_action_mse_overall``. If the short-horizon regime truly holds
(old step_00 = 0.046) it lands ~0.02-0.05 across the whole episode — directly
debunking the one-shot-long-chunk blow-up (old step_02 = 2.10).

Reuses ``_rollout`` and ``_ddim_from_v`` from the parent VERBATIM.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from egomimic.eval.dfot.eval_dfot_policy_action import DFoTPolicyActionEval
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


class DFoTPolicyRecedingHorizonEval(DFoTPolicyActionEval):
    def __init__(
        self,
        n_context_frames: int = 4,
        n_chunk_steps: int = 50,
        k_actions=(1, 2),
        max_episodes: int = 8,
        anchor_stride: int = 1,
        embodiment_name: str = "pushshapes_sim",
        image_key: str = "front_img_1",
        limit_val_batches: int = 4,
        max_videos: int = 2,
        rollout_steps: int = 6,          # accepted for ABI; T is computed as n_ctx+k
        recon_loss_n_frames: int = 20,   # accepted for ABI; unused in the RH metric
        video_subdir: str = "videos_policy_rh",
        viz_func=None,
        transform_lists=None,
    ):
        super().__init__(
            n_context_frames=n_context_frames, rollout_steps=rollout_steps,
            n_chunk_steps=n_chunk_steps, embodiment_name=embodiment_name,
            image_key=image_key, recon_loss_n_frames=recon_loss_n_frames,
            limit_val_batches=limit_val_batches, max_videos=max_videos,
            video_subdir=video_subdir, viz_func=viz_func, transform_lists=transform_lists,
        )
        self.k_actions = tuple(int(k) for k in k_actions)
        self.max_episodes = int(max_episodes)
        self.anchor_stride = max(1, int(anchor_stride))

    @torch.no_grad()
    def compute_metrics_and_viz(self, batch):
        algo = self.model
        metrics: Dict[str, torch.Tensor] = {}
        emb_id = get_embodiment_id(self.embodiment_name)
        if emb_id not in batch:
            return metrics, {}
        _batch = batch[emb_id]
        if self.image_key not in _batch:
            return metrics, {}
        outer = algo.outer_stage
        if not hasattr(outer, "_state_to_cond"):
            return metrics, {}  # not the 2D policy
        device = self.trainer.lightning_module.device
        ac_key = algo.resolved_ac_keys[emb_id]
        imgs = _batch[self.image_key]
        is_packed = _batch.get("_packed", False)

        # ---- per-episode spans, capped by max_episodes (NOT max_videos) ----
        if is_packed:
            cu = _batch["cu_seqlens"].to(imgs.device, dtype=torch.long)
            ne = min(int(cu.shape[0] - 1), self.max_episodes)
            spans = [(int(cu[i].item()), int(cu[i + 1].item() - cu[i].item())) for i in range(ne)]
        else:
            ne = min(imgs.shape[0], self.max_episodes)
            spans = [(ep, imgs.shape[1]) for ep in range(ne)]

        Ks = self.k_actions
        n_ctx = self.n_context_frames
        sse = {k: np.zeros(k) for k in Ks}
        cnt = {k: np.zeros(k, dtype=np.int64) for k in Ks}
        n_anchors_total = 0

        for (start, L) in spans:
            if L < n_ctx + max(Ks):
                continue
            # ---- encode the whole episode ONCE, reuse across all anchors ----
            if is_packed:
                sl = slice(start, start + L)
                img_seq = imgs[sl]
                act_seq = _batch[ac_key][sl]
                state = torch.cat([_batch[key][sl] for key in outer.bundle_obs_keys], -1)
            else:
                img_seq = imgs[start, :L]
                act_seq = _batch[ac_key][start, :L]
                state = torch.cat([_batch[key][start, :L] for key in outer.bundle_obs_keys], -1)
            img_seq = img_seq.to(device).float()
            if img_seq.max() > 1.5:
                img_seq = img_seq / 255.0
            act_seq = act_seq.to(device).float()                 # already normalized
            mu, _ = outer.vae.encode(img_seq)
            latent_all = outer.normalize_latent(mu)              # (L, C, H, W)
            cond_all = outer.state_only_proj(state.to(device).float())  # (L, proj)

            for k in Ks:
                T = n_ctx + k
                last_anchor = L - k  # inclusive
                for a in range(n_ctx, last_anchor + 1, self.anchor_stride):
                    lo = a - n_ctx
                    latent_ctx = latent_all[lo:a].unsqueeze(0)       # (1, n_ctx, C, H, W) clean
                    action_ctx = act_seq[lo:a].unsqueeze(0)          # (1, n_ctx, A) clean
                    cond = cond_all[lo:a + k].unsqueeze(0)           # (1, T, proj)
                    _, pred_act = self._rollout(algo, latent_ctx, action_ctx, cond, T, device)
                    pred_next = pred_act.squeeze(0)[n_ctx:n_ctx + k]  # (k, A) predicted future
                    gt_next = act_seq[a:a + k]                        # (k, A)
                    off = ((pred_next - gt_next) ** 2).mean(dim=-1).detach().cpu().numpy()
                    for o in range(k):
                        sse[k][o] += float(off[o])
                        cnt[k][o] += 1
                    if k == Ks[0]:
                        n_anchors_total += 1

        for k in Ks:
            for o in range(k):
                if cnt[k][o] > 0:
                    metrics[f"Valid/emb{emb_id}_rh_k{k}_action_mse_off{o:02d}"] = torch.tensor(
                        sse[k][o] / cnt[k][o], device=device)
            if cnt[k].sum() > 0:
                metrics[f"Valid/emb{emb_id}_rh_k{k}_action_mse_overall"] = torch.tensor(
                    sse[k].sum() / cnt[k].sum(), device=device)
        if n_anchors_total > 0:
            metrics[f"Valid/emb{emb_id}_rh_n_anchors"] = torch.tensor(float(n_anchors_total), device=device)
        return metrics, {}
