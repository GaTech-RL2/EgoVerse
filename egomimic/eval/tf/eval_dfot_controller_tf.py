"""Teacher-forced sanity check for the spatial_rh closed-loop controller.

Drives ``algo._inference_step_spatial_rh`` through GT val episodes one frame at
a time, but OVERWRITES the controller's recorded action with the GT action each
step (so its context is clean GT obs + GT actions — no own-action drift). Then
compares the controller's *predicted* action to GT in normalized space.

Decisive read:
  * overall_mse ~= offline RH eval's 0.017  -> controller is faithful given good
    context; the closed-loop sim failure is the offline->online DRIFT
    (exposure bias), not a controller bug.
  * overall_mse >> 0.017                     -> the controller code itself is the
    bug (context-building / cond / normalization), independent of drift.

Forces sp_commit=1 (re-plan every tick) so every step is testable.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from egomimic.eval.core.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


class DFoTControllerTFEval(EvalVideo):
    def __init__(
        self, n_steps_eval: int = 40, max_episodes: int = 4,
        embodiment_name: str = "pushshapes_sim", image_key: str = "front_img_1",
        limit_val_batches: int = 4, max_videos: int = 2,
        video_subdir: str = "videos_ctrl_tf", viz_func=None, transform_lists=None,
    ):
        super().__init__(limit_val_batches=limit_val_batches, viz_func=viz_func,
                         transform_lists=transform_lists, max_videos=max_videos)
        self.n_steps_eval = int(n_steps_eval)
        self.max_episodes = int(max_episodes)
        self.embodiment_name = embodiment_name
        self.image_key = str(image_key)
        self._video_subdir = str(video_subdir)

    def video_dir(self):
        import os
        return os.path.join(self.root_dir(), self._video_subdir)

    @torch.no_grad()
    def compute_metrics_and_viz(self, batch):
        algo = self.model
        metrics: Dict[str, torch.Tensor] = {}
        emb_id = get_embodiment_id(self.embodiment_name)
        if emb_id not in batch:
            return metrics, {}
        _batch = batch[emb_id]
        outer = algo.outer_stage
        if self.image_key not in _batch or not hasattr(outer, "_state_to_cond"):
            return metrics, {}
        device = self.trainer.lightning_module.device
        ac_key = algo.resolved_ac_keys[emb_id]
        obs_keys = list(outer.bundle_obs_keys)
        imgs = _batch[self.image_key]
        is_packed = _batch.get("_packed", False)
        if is_packed:
            cu = _batch["cu_seqlens"].to(imgs.device, dtype=torch.long)
            ne = min(int(cu.shape[0] - 1), self.max_episodes)
            spans = [(int(cu[i].item()), int(cu[i + 1].item() - cu[i].item())) for i in range(ne)]
        else:
            ne = min(imgs.shape[0], self.max_episodes)
            spans = [(ep, imgs.shape[1]) for ep in range(ne)]

        old_commit = getattr(algo, "sp_commit", 1)
        algo.sp_commit = 1
        Tcap = self.n_steps_eval
        sse = np.zeros(Tcap); cnt = np.zeros(Tcap, dtype=np.int64)
        dbg = []
        for (start, L) in spans:
            T = min(Tcap, L)
            if T < 2:
                continue
            for attr in ("_sp_lat", "_sp_state", "_sp_act", "_sp_queue"):
                if hasattr(algo, attr):
                    delattr(algo, attr)
            for t in range(T):
                if is_packed:
                    gi = start + t
                    sl = slice(gi, gi + 1)
                    img_t = imgs[sl].to(device).float()
                    gt_act = _batch[ac_key][gi].to(device).float()          # normalized (A,)
                    obs_zarr = {self.image_key: img_t}
                    for kk in obs_keys:
                        gv = _batch[kk][sl].to(device).float()              # normalized (1, d)
                        obs_zarr[kk] = algo.norm_stats.unnormalize({kk: gv}, emb_id)[kk]
                else:
                    img_t = imgs[start, t:t + 1].to(device).float()
                    gt_act = _batch[ac_key][start, t].to(device).float()
                    obs_zarr = {self.image_key: img_t}
                    for kk in obs_keys:
                        gv = _batch[kk][start, t:t + 1].to(device).float()
                        obs_zarr[kk] = algo.norm_stats.unnormalize({kk: gv}, emb_id)[kk]

                pred_unnorm = algo._inference_step_spatial_rh(obs_zarr, t, emb_id)  # (A,) world
                pu = torch.tensor(np.asarray(pred_unnorm), device=device).float().unsqueeze(0)
                pred_norm = algo.norm_stats.normalize({ac_key: pu}, emb_id)[ac_key].squeeze(0)
                e = float(((pred_norm - gt_act) ** 2).mean().item())
                sse[t] += e; cnt[t] += 1
                if len(dbg) < 6:
                    dbg.append((t, [round(x, 3) for x in pred_norm.detach().cpu().numpy().tolist()],
                                [round(x, 3) for x in gt_act.detach().cpu().numpy().tolist()]))
                # teacher-force: replace recorded predicted action with GT (normalized)
                algo._sp_act[-1] = gt_act.clone()
        algo.sp_commit = old_commit

        print("[TF_SANITY] first preds (t, pred_norm, gt_norm):")
        for d in dbg:
            print("   ", d)
        if cnt.sum() > 0:
            for t in range(Tcap):
                if cnt[t] > 0:
                    metrics[f"Valid/emb{emb_id}_tf_action_mse_step_{t:02d}"] = torch.tensor(sse[t] / cnt[t], device=device)
            metrics[f"Valid/emb{emb_id}_tf_action_mse_overall"] = torch.tensor(sse.sum() / cnt.sum(), device=device)
        return metrics, {}
