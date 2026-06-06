"""Teacher-forced action-CHUNK eval for the pixel obs+action policies.

For each sliding window of REAL val-data frames: pin n_context observed frames
clean, denoise the next k frames, read the predicted action chunk, and compare
to the ground-truth action chunk (MSE, in normalized action space). No env, no
exposure bias -> pure offline action-prediction quality.

  python tf_chunk_eval.py --ckpt <ckpt> --config-path <run>/.hydra/config.yaml \
      --n-ctx 1 --k 8 --n-steps 25 --max-eps 4 --max-frames 30
Handles Design A (action = pooled broadcast channels) and Design B (action_head).
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as tvio
from omegaconf import OmegaConf
from hydra.utils import instantiate

from scripts.smoke_sim_eval import load_algo_from_ckpt
from egomimic.algo.dfot.sampling import vanilla_schedule
from egomimic.algo.dfot.discrete_diffusion import DiscreteDiffusion
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-path", required=True)
    ap.add_argument("--n-ctx", type=int, default=1)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=25)
    ap.add_argument("--max-eps", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=30)
    ap.add_argument("--emb", default="pushshapes_sim")
    ap.add_argument("--ctx-act-mode", default="gt", choices=["gt", "zero", "prev"],
                    help="context action plane: gt=true a_t (copying-contaminated), "
                         "zero=zeros (cold-start), prev=a_{t-1} (closed-loop-like)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo, _ = load_algo_from_ckpt(args.ckpt, args.config_path)
    algo.nets = algo.nets.to(device)
    algo.device = device
    algo.nets.eval()

    full_cfg = OmegaConf.load(args.config_path)
    dm = instantiate(full_cfg.data)
    dm.setup(stage="validate")
    first = next(iter(dm.val_dataloader()))
    batch = first[0] if isinstance(first, tuple) else first
    batch = algo.process_batch_for_training(batch)

    emb_id = get_embodiment_id(args.emb)
    b = batch[emb_id]
    outer = algo.outer_stage
    diff = algo.diffusion
    ac_key = algo.resolved_ac_keys[emb_id]
    is_regress = hasattr(outer, "action_head")
    Ci = int(outer._image_channels)
    H = W = int(outer._image_size)
    A = int(outer.action_dim)
    Ca = int(getattr(outer, "_action_channels", A))
    n_ctx, k = args.n_ctx, args.k
    T = n_ctx + k
    print(f"[tf] model={'B/regress' if is_regress else 'A/broadcast'} Ci={Ci} Ca={Ca} A={A} n_ctx={n_ctx} k={k}")

    imgs = b[outer.image_key]
    acts = b[ac_key]
    packed = b.get("_packed", False)
    if packed:
        cu = b["cu_seqlens"].long()
        spans = [(int(cu[i]), int(cu[i + 1])) for i in range(min(len(cu) - 1, args.max_eps))]
    else:
        spans = [(ep, ep + 1) for ep in range(min(imgs.shape[0], args.max_eps))]

    dts = int(diff.timesteps) if isinstance(diff, DiscreteDiffusion) else None
    clean = -1 if dts is not None else 0.0

    sse = np.zeros(k)
    cnt = 0
    vid_frames = []
    WSCALE = 256.0 / 512.0  # world [0,512] -> 256px frame

    def _mark(im, act_norm, color, big=False):
        aw = algo.norm_stats.unnormalize({ac_key: act_norm.unsqueeze(0)}, emb_id)[ac_key][0]
        px = int(np.clip(float(aw[0]) * WSCALE, 3, 252))
        py = int(np.clip(float(aw[1]) * WSCALE, 3, 252))
        r = 3 if big else 1
        im[py - r:py + r + 1, px - r:px + r + 1] = color

    for (s, e) in spans:
        if packed:
            ep_img = imgs[s:e].to(device).float()
            ep_act = acts[s:e].to(device).float()
        else:
            ep_img = imgs[s].to(device).float()
            ep_act = acts[s].to(device).float()
        if ep_img.max() > 1.5:
            ep_img = ep_img / 255.0
        Tep = ep_img.shape[0]
        for t in range(0, max(0, min(Tep - T, args.max_frames))):
            ctx_obs = ep_img[t:t + n_ctx]                       # (n_ctx,3,H,W)
            gt_chunk = ep_act[t + n_ctx:t + n_ctx + k, :A]      # (k,A) normalized GT
            if is_regress:
                ctx = ctx_obs
                C = Ci
            else:
                if args.ctx_act_mode == "zero":
                    ctx_act = torch.zeros(n_ctx, Ca, device=device)
                elif args.ctx_act_mode == "prev":
                    lo = max(0, t - 1)
                    ctx_act = ep_act[lo:lo + n_ctx, :Ca]
                else:
                    ctx_act = ep_act[t:t + n_ctx, :Ca]
                planes = ctx_act.reshape(n_ctx, Ca, 1, 1).expand(n_ctx, Ca, H, W)
                ctx = torch.cat([ctx_obs, planes], dim=1)
                C = Ci + Ca
            ctx_stack = ctx.unsqueeze(0)
            sched = vanilla_schedule(args.n_steps, T, discrete_timesteps=dts).to(device).clone()
            sched[:, :n_ctx] = clean
            x = torch.randn(1, T, C, H, W, device=device)
            x[:, :n_ctx] = ctx_stack
            for sidx in range(sched.shape[0] - 1):
                klev = sched[sidx].clamp_min(0).long().unsqueeze(0)
                v = algo.backbone(x, klev, external_cond=None)
                x = algo._struct_ddim_step(diff, x, v, sched[sidx], sched[sidx + 1])
                x[:, :n_ctx] = ctx_stack
            pred_frames = x[0, n_ctx:n_ctx + k]
            if is_regress:
                pred_chunk = outer.action_head(pred_frames)[:, :A]
            else:
                pred_chunk = pred_frames[:, Ci:Ci + Ca].mean(dim=(2, 3))[:, :A]
            err = ((pred_chunk - gt_chunk) ** 2).mean(dim=1)
            sse += err.detach().cpu().numpy()
            cnt += 1

            if len(vid_frames) < 200:
                base = ep_img[t + n_ctx].clamp(0, 1)                 # (3,H,W) GT obs of 1st predicted frame
                up = F.interpolate(base.unsqueeze(0), size=(256, 256), mode="nearest")[0]
                im = (up.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
                for j in range(k):                                          # full chunk: k green (GT) + k red (pred)
                    big = (j == 0)                                          # immediate/committed action drawn larger
                    _mark(im, gt_chunk[j], np.array([0, 255, 0], np.uint8), big)
                    _mark(im, pred_chunk[j], np.array([255, 0, 0], np.uint8), big)
                vid_frames.append(torch.from_numpy(im))

    if cnt == 0:
        print("[tf] NO WINDOWS")
        return
    print(f"[tf] windows={cnt}")
    for j in range(k):
        print(f"[tf] chunk_mse_step_{j:02d} = {sse[j] / cnt:.5f}")
    print(f"[tf] chunk_mse_OVERALL = {sse.sum() / (cnt * k):.5f}")

    if vid_frames:
        out = os.environ.get("TF_VID", "/tmp/tf_gtpred.mp4")
        tvio.write_video(out, torch.stack(vid_frames), fps=8, video_codec="h264")
        print(f"[tf] wrote video {out} frames={len(vid_frames)}  (green=GT action, red=pred action)")


if __name__ == "__main__":
    main()
