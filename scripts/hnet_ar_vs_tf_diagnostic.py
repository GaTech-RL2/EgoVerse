"""Diagnose H-Net sim eval: teacher-forced vs auto-regressive rollout.

Load the fused checkpoint, pick one validation episode, then:
  1. Run forward() teacher-forced — pass GT actions, get model's predicted
     action_t at every step given the GT history.
  2. Run AR rollout via init_step_state + step — at each step, model uses
     its own previous prediction as the past action token.
  3. Compare predicted actions to GT.

If TF predictions track GT closely but AR drifts away after a few steps:
  documented exposure bias, not a bug.
If TF predictions already drift far from GT:
  the model didn't actually learn — either a code bug (norm stats, obs
  format, etc.) or data is genuinely too sparse.

Run on a compute node:
    PYTHONPATH=. python scripts/hnet_ar_vs_tf_diagnostic.py \
        --ckpt logs/hnet_variants/fused_circle_basic_2026-05-16_05-28-29/csv_logs/lightning_logs/version_0/checkpoints/epoch=49-step=400.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


def load_algo(ckpt_path: str, config_path: str):
    print(f"[load] ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    norm_state = hparams.get("norm_stats_state")
    if norm_state is None:
        raise SystemExit("No norm_stats_state in checkpoint")

    cfg = OmegaConf.load(config_path)
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    norm_stats = MultiDataset.from_state(norm_state)
    algo = instantiate(cfg.model.robomimic_model, norm_stats=norm_stats)

    state_dict = ckpt["state_dict"]
    new_sd = {}
    for k, v in state_dict.items():
        for prefix in ("nets.", "model.nets."):
            if k.startswith(prefix):
                new_sd[k[len(prefix) :]] = v
                break
        else:
            new_sd[k] = v
    missing, unexpected = algo.nets.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"[load] missing ({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"[load] unexpected ({len(unexpected)}): {unexpected[:3]}")
    return algo, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--max-frames", type=int, default=60)
    args = parser.parse_args()

    config_path = str(Path(args.ckpt).parents[4] / ".hydra" / "config.yaml")
    print(f"[config] {config_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo, cfg = load_algo(args.ckpt, config_path)
    algo.nets = algo.nets.to(device)
    algo.device = device
    algo.nets.eval()

    # Build val loader
    dm = instantiate(cfg.data)
    dm.setup(stage="validate")
    val_loader = dm.val_dataloader()
    first = next(iter(val_loader))
    if isinstance(first, tuple):
        batch = first[0]
    else:
        batch = first

    emb_name = next(iter(batch.keys()))
    _b = batch[emb_name]
    # Resolve to int id used by resolved_ac_keys.
    emb_id = (
        algo.embodiment_ids[emb_name] if hasattr(algo, "embodiment_ids") else emb_name
    )
    print(f"[batch] emb_name={emb_name} emb_id={emb_id} keys={sorted(_b.keys())}")
    cu = _b["cu_seqlens"]
    # Pick first episode of the batch
    s = int(cu[0].item())
    e = int(cu[1].item())
    T = min(int(e - s), args.max_frames)
    print(f"[ep0] T={int(e - s)} (capped to {T})")

    ac_key = algo.resolved_ac_keys[emb_id]
    print(f"[ep0] ac_key={ac_key}")

    # Extract per-frame data for episode 0
    ep_data = {}
    total = int(cu[-1].item())
    for k, v in _b.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == total:
            ep_data[k] = v[s : s + T].to(device)
    actions_gt_norm = ep_data[ac_key]  # (T, action_dim)
    print(f"[ep0] actions_gt_norm shape={tuple(actions_gt_norm.shape)}")

    # Build obs dict per the algo's expectation. process_batch_for_training
    # would normally do this, but we're slicing one episode here.
    obs_keys = ["state_agent_obj", "front_img_1"]
    obs = {k: ep_data[k].unsqueeze(0) for k in obs_keys if k in ep_data}  # (1, T, ...)
    actions_in = actions_gt_norm.unsqueeze(0)  # (1, T, A)

    # 1. Teacher-forced forward — pad to action_horizon if needed.
    print("\n[TF] running teacher-forced forward()")
    with torch.no_grad():
        policy = algo.nets["policy"]
        pred_tf, _ = policy.forward(actions_in, obs)
    pred_tf = pred_tf.squeeze(0)  # (T, A)
    diff_tf = (pred_tf - actions_gt_norm).abs().mean(dim=-1)
    print(
        f"[TF] per-step |pred-gt| mean (normalized): first 10 = {diff_tf[:10].cpu().tolist()}"
    )
    print(f"[TF] overall mean = {diff_tf.mean().item():.4f}")
    print(f"[TF] max = {diff_tf.max().item():.4f}")

    # 2. AR rollout
    print("\n[AR] running step-by-step")
    state = policy.init_step_state(batch_size=1, T_max=T, device=device)
    pred_ar = []
    for t in range(T):
        obs_t = {k: v[:, t : t + 1] for k, v in obs.items()}  # (1, 1, ...)
        a_t = policy.step(state, obs_t, t).squeeze(0).squeeze(0)  # (A,)
        pred_ar.append(a_t)
    pred_ar = torch.stack(pred_ar, dim=0)  # (T, A)
    diff_ar = (pred_ar - actions_gt_norm).abs().mean(dim=-1)
    print(
        f"[AR] per-step |pred-gt| mean (normalized): first 10 = {diff_ar[:10].cpu().tolist()}"
    )
    print(f"[AR] overall mean = {diff_ar.mean().item():.4f}")
    print(f"[AR] max = {diff_ar.max().item():.4f}")

    # 3. TF vs AR divergence
    print("\n[CMP] TF vs AR divergence (|pred_tf - pred_ar|.mean): first 10")
    div = (pred_tf - pred_ar).abs().mean(dim=-1)
    print(f"  {div[:10].cpu().tolist()}")
    print(f"  overall mean = {div.mean().item():.4f}")
    print(f"  max = {div.max().item():.4f}")

    # 4. Unnormalize for world-coord interpretation
    print("\n[WORLD] in pixel space:")
    # The norm_stats.unnormalize call expects a batch dict, single emb.
    from collections import OrderedDict

    one_to_unnorm_tf = OrderedDict()
    one_to_unnorm_tf[ac_key] = pred_tf.cpu()
    out_tf = algo.norm_stats.unnormalize(one_to_unnorm_tf, emb_id)
    one_to_unnorm_ar = OrderedDict()
    one_to_unnorm_ar[ac_key] = pred_ar.cpu()
    out_ar = algo.norm_stats.unnormalize(one_to_unnorm_ar, emb_id)
    one_to_unnorm_gt = OrderedDict()
    one_to_unnorm_gt[ac_key] = actions_gt_norm.cpu()
    out_gt = algo.norm_stats.unnormalize(one_to_unnorm_gt, emb_id)

    world_tf = out_tf[ac_key]
    world_ar = out_ar[ac_key]
    world_gt = out_gt[ac_key]

    tf_err_px = (world_tf - world_gt).abs().mean(dim=-1)
    ar_err_px = (world_ar - world_gt).abs().mean(dim=-1)
    print(
        f"  TF pixel err (mean per step, first 10): {[f'{x:.1f}' for x in tf_err_px[:10].tolist()]}"
    )
    print(
        f"  AR pixel err (mean per step, first 10): {[f'{x:.1f}' for x in ar_err_px[:10].tolist()]}"
    )
    print(f"  TF overall pixel err = {tf_err_px.mean().item():.2f} (over 512px range)")
    print(f"  AR overall pixel err = {ar_err_px.mean().item():.2f}")
    print(f"  GT actions[0..4] = {world_gt[:5].tolist()}")
    print(f"  TF preds [0..4]  = {world_tf[:5].tolist()}")
    print(f"  AR preds [0..4]  = {world_ar[:5].tolist()}")


if __name__ == "__main__":
    main()
