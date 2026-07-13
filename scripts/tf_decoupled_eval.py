"""Clean obs->action MSE for the DECOUPLED pixel policy. For each val frame:
pin the last n_ctx observed RGB frames CLEAN, denoise the action token, read the
action predicted at the CURRENT frame, compare to GT. NO action ever fed as input
-> this is the TRUE obs->action ability (the metric the fused model couldn't beat:
0.087). Low here => obs->action learned => closed-loop has a chance.

  python tf_decoupled_eval.py --ckpt <ckpt> --config-path <run>/.hydra/config.yaml --n-ctx 4
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
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
    ap.add_argument("--n-ctx", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=25)
    ap.add_argument("--max-eps", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--n-samples", type=int, default=1, help="avg K diffusion samples per frame (tests variance vs bias)")
    ap.add_argument("--emb", default="pushshapes_sim")
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
    H = W = int(outer._image_size)
    A = int(outer.action_dim)
    n_ctx = args.n_ctx

    imgs = b[outer.image_key]
    acts = b[ac_key]
    packed = b.get("_packed", False)
    if packed:
        cu = b["cu_seqlens"].long()
        spans = [(int(cu[i]), int(cu[i + 1])) for i in range(min(len(cu) - 1, args.max_eps))]
    else:
        spans = [(ep, ep + 1) for ep in range(min(imgs.shape[0], args.max_eps))]
    dts = int(diff.timesteps) if isinstance(diff, DiscreteDiffusion) else None
    clean = 0   # pinned-obs noise level: 0 = cleanest TRAINED level (NOT the -1 schedule sentinel)

    sse = 0.0
    cnt = 0
    for (s, e) in spans:
        ep_img = (imgs[s:e] if packed else imgs[s]).to(device).float()
        ep_act = (acts[s:e] if packed else acts[s]).to(device).float()
        if ep_img.max() > 1.5:
            ep_img = ep_img / 255.0
        Tep = ep_img.shape[0]
        for t in range(n_ctx - 1, min(Tep, args.max_frames)):
            idx = [max(0, t - n_ctx + 1 + i) for i in range(n_ctx)]
            rgb_ctx = ep_img[idx].unsqueeze(0)                 # (1,T,3,H,W) clean
            gt_a = ep_act[t, :A]
            obs_levels = torch.full(
                (1, n_ctx), clean, device=device,
                dtype=torch.long if dts is not None else torch.float32,
            )
            sched = vanilla_schedule(args.n_steps, n_ctx, discrete_timesteps=dts).to(device)
            samples = []
            for _ in range(args.n_samples):
                x_act = torch.randn(1, n_ctx, A, device=device)
                for si in range(sched.shape[0] - 1):
                    _, v_act = algo.backbone(
                        rgb_ctx, obs_levels, external_cond=None,
                        action=x_act, action_noise_levels=sched[si].unsqueeze(0),
                    )
                    x_act = algo._struct_ddim_step(diff, x_act, v_act, sched[si], sched[si + 1])
                samples.append(x_act[0, -1, :A])
            pred = torch.stack(samples).mean(0)
            sse += ((pred - gt_a) ** 2).mean().item()
            cnt += 1
    print(f"[dec-tf] clean_obs2action_MSE = {sse / max(1, cnt):.5f}  (n={cnt}, n_ctx={n_ctx})  "
          f"[fused baseline obs-only was 0.087]")


if __name__ == "__main__":
    main()
