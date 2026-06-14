"""Boundary diagnostic probe for the bcrnnHnetC8FHR run.

Loads the ep400 checkpoint, runs the FHR H-Net core teacher-forced over a few
val episodes (windowed exactly like forward_training: non-overlapping
length-rnn_horizon obs windows, obs_stride subsampling), and captures the
ChunkerStage RoutingModule's per-position boundary_prob (boundary_prob[...,1]).

Reports boundaries-per-look vs the target (N=2.0 => 1 boundary per 2 looks) and
saves boundary-prob strip/heatmap PNGs per episode.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from egomimic.models.hnet_nets import routing as routing_mod
from egomimic.algo import bc_rnn as bcrnn_mod  # noqa: F401  (ensure import path)


# ---- capture sink: monkeypatch RoutingModule._forward_padded ----
_CAPTURE = []   # list of boundary_prob[...,1] tensors, shape (B, L)

_orig_fwd_padded = routing_mod.RoutingModule._forward_padded
def _patched_fwd_padded(self, hidden_states, mask, inference_params):
    out = _orig_fwd_padded(self, hidden_states, mask, inference_params)
    # boundary_prob: (B, L, 2); index -1 == P(boundary)
    bp = out.boundary_prob[..., 1].detach().float().cpu()
    _CAPTURE.append(bp)
    return out
routing_mod.RoutingModule._forward_padded = _patched_fwd_padded


def load_algo_from_ckpt(ckpt_path, config_path):
    print(f"[load] ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters") or ckpt.get("hparams") or {}
    cfg_for_model = (
        OmegaConf.create(hparams["config_tree"])
        if "config_tree" in hparams
        else OmegaConf.load(config_path)
    )
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
    norm_state = hparams.get("norm_stats_state")
    if norm_state is None:
        raise SystemExit("hyper_parameters has no norm_stats_state")
    norm_stats = MultiDataset.from_state(norm_state)
    algo = instantiate(cfg_for_model.model.robomimic_model, norm_stats=norm_stats)
    state_dict = ckpt["state_dict"]
    new_sd = {}
    for k, v in state_dict.items():
        for prefix in ("nets.", "model.nets."):
            if k.startswith(prefix):
                new_sd[k[len(prefix):]] = v
                break
        else:
            new_sd[k] = v
    missing, unexpected = algo.nets.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"[load] missing keys ({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"[load] unexpected keys ({len(unexpected)}): {unexpected[:3]}")
    return algo, cfg_for_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-path", required=True)
    ap.add_argument("--out-dir", default="/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/fhr_boundary_diag")
    ap.add_argument("--n-episodes", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    algo, cfg = load_algo_from_ckpt(args.ckpt, args.config_path)
    algo.nets = algo.nets.to(device); algo.device = device; algo.nets.eval()
    policy = algo.nets["policy"]
    H = int(algo.rnn_horizon)
    sigma = int(policy.obs_stride)
    target_N = float(policy.lstm.target_compression_ratio)
    print(f"[cfg] rnn_horizon(H)={H}  obs_stride(sigma)={sigma}  target_N={target_N}  "
          f"ratio_loss_weight={policy.lstm.ratio_loss_weight}")

    full_cfg = OmegaConf.load(args.config_path)
    dm = instantiate(full_cfg.data)
    dm.setup(stage="validate")
    val_loader = dm.val_dataloader()
    first = next(iter(val_loader))
    batch = first[0] if isinstance(first, tuple) else first
    batch = algo.process_batch_for_training(batch)
    emb_id = list(batch.keys())[0]
    _batch = batch[emb_id]
    print(f"[batch] emb_id={emb_id} keys={list(_batch.keys())[:8]}")

    obs_padded, actions_padded, mask, seq_lens = algo._unpack_obs_actions(_batch, emb_id)
    print(f"[batch] obs keys={list(obs_padded.keys())}  seq_lens={seq_lens.tolist()[:8]}")
    B = seq_lens.shape[0]

    # Per-episode: subsample obs by sigma, cut into non-overlapping length-H windows,
    # encode each window (captures boundary_prob via the monkeypatch), aggregate.
    per_ep_bp = []      # list of 1D np arrays: boundary_prob over obs-steps (concatenated windows)
    per_ep_forced = []  # list of 1D bool arrays: True at window-start (forced) positions
    n_eps = min(args.n_episodes, B)
    summary_rows = []
    for b in range(n_eps):
        L = int(seq_lens[b].item())
        n_obs = max(1, (L + sigma - 1) // sigma)  # number of obs-steps in this episode
        obs_idx = (torch.arange(n_obs) * sigma).clamp(max=L - 1)  # frame indices observed
        # build per-window obs dict and encode
        bp_chunks = []
        forced_chunks = []
        for s in range(0, n_obs, H):
            win_idx = obs_idx[s:s + H]
            Wlen = win_idx.shape[0]
            obs_win = {k: v[b:b+1, win_idx].to(device) for k, v in obs_padded.items()}
            _CAPTURE.clear()
            with torch.no_grad():
                _ = policy(obs_win)   # triggers obs_encoder + core -> chunker._forward_padded
            assert len(_CAPTURE) >= 1, "no boundary_prob captured"
            bp = _CAPTURE[-1][0].numpy()  # (Wlen,)
            bp = bp[:Wlen]
            forced = np.zeros(Wlen, dtype=bool); forced[0] = True  # position 0 forced=1.0
            bp_chunks.append(bp); forced_chunks.append(forced)
        bp_all = np.concatenate(bp_chunks); forced_all = np.concatenate(forced_chunks)
        per_ep_bp.append(bp_all); per_ep_forced.append(forced_all)

        # stats over NON-forced obs-steps (exclude window-start forced positions)
        real = ~forced_all
        bp_real = bp_all[real]
        n_real = bp_real.size
        n_bound = int((bp_real > 0.5).sum())   # argmax==1 == prob>0.5
        frac_bound = n_bound / max(1, n_real)
        mean_bp = float(bp_real.mean()) if n_real else 0.0
        # boundaries per look = fraction of real looks that are a boundary; target=1/N
        summary_rows.append((b, n_obs, n_real, n_bound, frac_bound, mean_bp))
        print(f"[ep {b}] L={L} n_obs={n_obs} real_looks={n_real} boundaries={n_bound} "
              f"frac_boundary={frac_bound:.3f} (target≈{1.0/target_N:.3f}) mean_bp={mean_bp:.3f}")

    # ---- aggregate report ----
    print("\n=== BOUNDARY SUMMARY (target N=%.1f => target frac=%.3f, ~1 boundary per %.1f looks) ===" %
          (target_N, 1.0/target_N, target_N))
    tot_real = sum(r[2] for r in summary_rows); tot_b = sum(r[3] for r in summary_rows)
    overall_frac = tot_b / max(1, tot_real)
    looks_per_bound = (tot_real / tot_b) if tot_b else float("inf")
    all_bp_real = np.concatenate([per_ep_bp[i][~per_ep_forced[i]] for i in range(n_eps)])
    print(f"overall real_looks={tot_real} boundaries={tot_b} frac_boundary={overall_frac:.3f} "
          f"=> ~1 boundary per {looks_per_bound:.2f} looks (target {target_N:.1f})")
    print(f"overall mean boundary_prob (real looks)={all_bp_real.mean():.3f}  "
          f"std={all_bp_real.std():.3f}  min={all_bp_real.min():.3f}  max={all_bp_real.max():.3f}")

    # ---- visualization: per-episode strip of boundary_prob vs obs-step ----
    fig, axes = plt.subplots(n_eps, 1, figsize=(14, 2.2 * n_eps), squeeze=False)
    for i in range(n_eps):
        ax = axes[i][0]
        bp = per_ep_bp[i]; forced = per_ep_forced[i]
        x = np.arange(bp.size)
        ax.bar(x, bp, width=1.0, color="steelblue", edgecolor="none")
        ax.axhline(0.5, color="red", ls="--", lw=0.8, label="boundary thresh (0.5)")
        # mark forced window-starts
        for fx in x[forced]:
            ax.axvline(fx, color="gray", ls=":", lw=0.6)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"ep{i}\nP(bound)")
        ax.set_xlim(-0.5, bp.size - 0.5)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
        if i == n_eps - 1:
            ax.set_xlabel("obs-step index (sigma=%d frames/look, dotted=forced window-start)" % sigma)
    fig.suptitle("FHR ep400 boundary_prob per obs-step  (target N=%.1f, frac=%.2f)" %
                 (target_N, 1.0/target_N), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p1 = out / "FHR_boundary_diag_ep400_strip.png"
    fig.savefig(p1, dpi=130); plt.close(fig)
    print(f"[viz] wrote {p1}")

    # ---- heatmap across episodes (padded to same length) ----
    maxL = max(b.size for b in per_ep_bp)
    M = np.full((n_eps, maxL), np.nan)
    for i, bp in enumerate(per_ep_bp):
        M[i, :bp.size] = bp
    fig2, ax2 = plt.subplots(figsize=(14, 1.0 + 0.7 * n_eps))
    im = ax2.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax2.set_yticks(range(n_eps)); ax2.set_yticklabels([f"ep{i}" for i in range(n_eps)])
    ax2.set_xlabel("obs-step index"); ax2.set_title("FHR ep400 boundary_prob heatmap (bright=boundary)")
    fig2.colorbar(im, ax=ax2, fraction=0.025, pad=0.01, label="P(boundary)")
    fig2.tight_layout()
    p2 = out / "FHR_boundary_diag_ep400_heatmap.png"
    fig2.savefig(p2, dpi=130); plt.close(fig2)
    print(f"[viz] wrote {p2}")

    # ---- histogram of boundary_prob over real looks ----
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.hist(all_bp_real, bins=40, color="teal", edgecolor="k", alpha=0.8)
    ax3.axvline(0.5, color="red", ls="--", label="thresh 0.5")
    ax3.set_xlabel("boundary_prob (real looks)"); ax3.set_ylabel("count")
    ax3.set_title("FHR ep400 boundary_prob distribution\nfrac>0.5=%.3f (target %.3f)" %
                  (overall_frac, 1.0/target_N))
    ax3.legend(); fig3.tight_layout()
    p3 = out / "FHR_boundary_diag_ep400_hist.png"
    fig3.savefig(p3, dpi=130); plt.close(fig3)
    print(f"[viz] wrote {p3}")
    print("DONE_BOUNDARY_PROBE")


if __name__ == "__main__":
    main()
