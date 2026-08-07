"""Decoupled CHUNK overlay: for each val frame, pin obs_t CLEAN, predict the
k-action chunk [a_t..a_{t+k-1}] (decoupled, n_samples averaged, future obs
denoised by the model's world-model), and overlay GT (green) vs pred (red)
markers on the obs -> visually confirm the model's obs->action is correct.
Step 0 (from the real clean obs) should be tightest; later steps use predicted
future obs so they fan out."""
from __future__ import annotations
import argparse, os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as tvio
from omegaconf import OmegaConf
from hydra.utils import instantiate

from egomimic.eval.core.ckpt_loading import load_algo_from_ckpt
from egomimic.models.diffusion.sampling import vanilla_schedule
from egomimic.algo.diffusion.discrete_diffusion import DiscreteDiffusion
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-path", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=25)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--max-eps", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--emb", default="pushshapes_sim")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo, _ = load_algo_from_ckpt(args.ckpt, args.config_path)
    algo.nets = algo.nets.to(device); algo.device = device; algo.nets.eval()

    full_cfg = OmegaConf.load(args.config_path)
    dm = instantiate(full_cfg.data); dm.setup(stage="validate")
    print("DATA_SETUP_DONE", flush=True)
    first = next(iter(dm.val_dataloader()))
    print("BATCH_LOADED", flush=True)
    batch = first[0] if isinstance(first, tuple) else first
    batch = algo.process_batch_for_training(batch)
    print("BATCH_PROCESSED", flush=True)
    emb_id = get_embodiment_id(args.emb)
    b = batch[emb_id]
    outer = algo.outer_stage; diff = algo.diffusion
    ac_key = algo.resolved_ac_keys[emb_id]
    Ci = int(outer._image_channels); H = W = int(outer._image_size)
    A = int(outer.action_dim)
    k = args.k; n_samp = args.n_samples

    imgs = b[outer.image_key]; acts = b[ac_key]
    packed = b.get("_packed", False)
    if packed:
        cu = b["cu_seqlens"].long()
        spans = [(int(cu[i]), int(cu[i + 1])) for i in range(min(len(cu) - 1, args.max_eps))]
    else:
        spans = [(ep, ep + 1) for ep in range(min(imgs.shape[0], args.max_eps))]
    dts = int(diff.timesteps) if isinstance(diff, DiscreteDiffusion) else None
    WSCALE = 256.0 / 512.0

    def _mark(im, act_norm, color, big=False):
        aw = algo.norm_stats.unnormalize({ac_key: act_norm.unsqueeze(0)}, emb_id)[ac_key][0]
        px = int(np.clip(float(aw[0]) * WSCALE, 3, 252))
        py = int(np.clip(float(aw[1]) * WSCALE, 3, 252))
        r = 3 if big else 1
        im[py - r:py + r + 1, px - r:px + r + 1] = color

    sse = np.zeros(k); cnt = 0; vid = []
    for (s, e) in spans:
        ep_img = (imgs[s:e] if packed else imgs[s]).to(device).float()
        ep_act = (acts[s:e] if packed else acts[s]).to(device).float()
        if ep_img.max() > 1.5:
            ep_img = ep_img / 255.0
        Tep = ep_img.shape[0]
        for t in range(0, min(Tep - k, args.max_frames)):
            T = k
            rgb0 = ep_img[t].unsqueeze(0).unsqueeze(0)            # (1,1,C,H,W) clean obs_t
            sched = vanilla_schedule(args.n_steps, T, discrete_timesteps=dts).to(device)
            obs_sched = sched.clone(); obs_sched[:, :1] = 0       # frame 0 (obs_t) clean
            ctx_b = rgb0.expand(n_samp, -1, -1, -1, -1)
            x_obs = torch.randn(n_samp, T, Ci, H, W, device=device); x_obs[:, :1] = ctx_b
            x_act = torch.randn(n_samp, T, A, device=device)
            for st in range(sched.shape[0] - 1):
                o_lev = obs_sched[st].unsqueeze(0).expand(n_samp, -1)
                a_lev = sched[st].unsqueeze(0).expand(n_samp, -1)
                v_img, v_act = algo.backbone(x_obs, o_lev, external_cond=None,
                                             action=x_act, action_noise_levels=a_lev)
                x_obs = algo._struct_ddim_step(diff, x_obs, v_img, obs_sched[st], obs_sched[st + 1])
                x_obs[:, :1] = ctx_b
                x_act = algo._struct_ddim_step(diff, x_act, v_act, sched[st], sched[st + 1])
            pred_chunk = x_act[:, 0:k].mean(0)[:, :A]             # (k,A)
            gt_chunk = ep_act[t:t + k, :A]
            sse += ((pred_chunk - gt_chunk) ** 2).mean(dim=1).detach().cpu().numpy()
            cnt += 1
            print(f"[ovl] frame {cnt}", flush=True)
            if len(vid) < 200:
                up = F.interpolate(ep_img[t].unsqueeze(0), size=(256, 256), mode="nearest")[0]
                im = (up.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
                for j in range(k):
                    big = (j == 0)
                    _mark(im, gt_chunk[j], np.array([0, 255, 0], np.uint8), big)
                    _mark(im, pred_chunk[j], np.array([255, 0, 0], np.uint8), big)
                vid.append(torch.from_numpy(im))
    print(f"[dec-overlay] windows={cnt}")
    for j in range(k):
        print(f"[dec-overlay] chunk_mse_step_{j:02d} = {sse[j] / max(1, cnt):.5f}")
    if vid:
        out = os.environ.get("OVL_VID", "/tmp/dec_overlay.mp4")
        tvio.write_video(out, torch.stack(vid), fps=6, video_codec="h264")
        print(f"[dec-overlay] wrote {out} frames={len(vid)} (green=GT, red=pred, big dot=immediate action)")


if __name__ == "__main__":
    main()
