"""DTW-latent eval — cross-embodiment alignment metric on the APEX agnostic tokens.

The APEX tokens are the deepest (most-compressed) ``ComputeStage.main_network``
output — the embodiment-invariant representation the dual-stream design bets on
(the d1024 apex in dualstream sym_w1; the final ComputeStage in a single-stream /
pyramid model). For a correctly-agnostic model, the circle-pusher and
small-circle-pusher latent paths through this apex space should *align* even
though the two embodiments differ — this eval measures that alignment with DTW.

Pipeline (all offline, one teacher-forced packed forward per embodiment):
  1. collect_apex_tokens : hook every ``ComputeStage.main_network``; keep the
     fewest-rows (most compressed) output = apex. Split into per-episode
     ``(n_chunks_i, d)`` trajectories with the chunk-space ``cu_seqlens`` the
     apex was called with.
  2. fit_pca             : ONE PCA fit on the pooled apex tokens across ALL
     chunkviz episodes + BOTH embs; keep k comps for cum-EVR >= --evr (0.95).
  3. per cross-emb ALIGNED PAIR (circle ep i <-> small-circle ep i, same task
     setup): project both apex trajectories into the PCA space, run DTW
     (euclidean per-step cost, standard DP) and report the DTW distance
     NORMALIZED by warp-path length so different-length episodes compare.

The core (collect_apex_tokens -> fit_pca -> dtw_normalized) is factored into
importable functions so it can later be called as a validation metric during
training.

    python -m egomimic.eval.dtw_latent_eval \
        --ckpt logs/indomain_c4/main1599eval/checkpoints/last.ckpt \
        --config-path logs/indomain_c4/main1599eval/.hydra/config_viz.yaml \
        [--good-eps 0,2,4] [--evr 0.95] [--n-episodes 6]
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

from egomimic.eval.core.ckpt_loading import load_algo_from_ckpt
from egomimic.models.hnet.stages import ComputeStage

# emb ids (egomimic/rldb/embodiment/embodiment.py):
CIRCLE_EMB = 15        # PUSHSHAPES_SIM               (circle pusher)
SMALL_EMB = 17         # PUSHSHAPES_SIM_SMALL_CIRCLE  (small-circle pusher)
# observations.state = concat(pusher_xy[0:2], obj_xyangle[2:5]); the object
# (T-block) sub-vector [2:5] is the task identity (independent of pusher shape).
OBJ_SLICE = slice(2, 5)


# --------------------------------------------------------------------------- #
# 1. APEX token capture
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_apex_tokens(algo, batch, n_episodes=None):
    """Capture per-episode apex-token trajectories for every packed emb in ``batch``.

    Returns ``{emb_id: {"eps": [ (n_i, d) float32, ... ], "frame_cu": ndarray,
    "d": int, "n_stages": int}}`` where each ``eps[i]`` is the apex-token
    trajectory (one row per top-level chunk) for episode ``i``.
    """
    out = {}
    for emb_id, _b in batch.items():
        if not _b.get("_packed", False):
            continue
        policy = algo.policy

        captured = []  # list of (tokens (N,d) np.float32, cu np|None)

        def _hook(_module, _args, _kwargs, output):
            t = output.detach().float().cpu()
            if t.dim() == 3:
                t = t.reshape(-1, t.shape[-1])
            cu = _kwargs.get("cu_seqlens", None)
            # positional fallback: main_network(x, cu_seqlens, max_seqlen, ...)
            if cu is None and len(_args) >= 1 and isinstance(_args[0], torch.Tensor) \
                    and _args[0].dim() == 1:
                cu = _args[0]
            cu_np = cu.detach().cpu().numpy() if isinstance(cu, torch.Tensor) else None
            captured.append((t.numpy(), cu_np))

        handles = [
            m.main_network.register_forward_hook(_hook, with_kwargs=True)
            for m in policy.modules()
            if isinstance(m, ComputeStage)
        ]
        if not handles:
            raise RuntimeError(
                f"emb{emb_id}: no ComputeStage found under algo.policy — cannot "
                f"locate the apex network."
            )
        try:
            ac_key = algo.resolved_ac_keys[emb_id]
            obs = algo._build_obs(_b, emb_id)
            actions = _b[ac_key]
            cu = _b["cu_seqlens"]
            max_seqlen = int(_b["max_seq_len"])
            domain = getattr(algo, "domain_by_id", {}).get(emb_id)
            policy.forward_packed(actions, obs, cu, max_seqlen, embodiment_id=domain)
        finally:
            for h in handles:
                h.remove()

        if not captured:
            raise RuntimeError(
                f"emb{emb_id}: no ComputeStage.main_network fired during "
                f"forward_packed; cannot capture apex tokens."
            )

        # apex = the most-compressed capture (fewest rows).
        apex_tokens, apex_cu = min(captured, key=lambda c: c[0].shape[0])
        frame_cu = cu.detach().cpu().numpy().astype(np.int64)
        n_ep = len(frame_cu) - 1
        N = apex_tokens.shape[0]

        # per-episode split. primary: chunk-space cu the apex was called with.
        splits = None
        if (apex_cu is not None and len(apex_cu) == n_ep + 1
                and int(apex_cu[-1]) == N and int(apex_cu[0]) == 0):
            ac = apex_cu.astype(np.int64)
            splits = [apex_tokens[ac[i]:ac[i + 1]] for i in range(n_ep)]
        else:
            # fallback: per-episode top-level chunk counts from the boundary masks
            counts = _top_chunk_counts_per_episode(algo, _b, emb_id, frame_cu)
            if counts is not None and int(sum(counts)) == N:
                edges = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
                splits = [apex_tokens[edges[i]:edges[i + 1]] for i in range(n_ep)]
        if splits is None:
            raise RuntimeError(
                f"emb{emb_id}: could not split {N} apex tokens into {n_ep} "
                f"episodes (apex_cu={None if apex_cu is None else apex_cu.tolist()})."
            )

        if n_episodes is not None:
            splits = splits[:n_episodes]
        out[emb_id] = {
            "eps": [s.astype(np.float32) for s in splits],
            "frame_cu": frame_cu,
            "d": int(apex_tokens.shape[1]),
            "n_stages": len(captured),
        }
    return out


def _top_chunk_counts_per_episode(algo, _b, emb_id, frame_cu):
    """Fallback per-episode top-level chunk counts from the composed frame-res
    boundary mask (one True per top-level chunk)."""
    try:
        viz = algo.collect_chunkviz({emb_id: _b})
    except Exception:  # noqa: BLE001
        return None
    levels = viz.get(emb_id, {}).get("levels")
    if not levels:
        return None
    T_total = int(frame_cu[-1])
    try:
        from egomimic.eval.probes.eval_boundary_strip import (
            _compose_top_boundary_frames,
        )
        composed = _compose_top_boundary_frames(levels, T_total)
        mask = composed[1].numpy().astype(bool) if composed is not None else None
    except Exception:  # noqa: BLE001
        mask = None
    if mask is None:
        for _p, m in levels:  # single-chunker: own mask already frame-res
            m = m.numpy().astype(bool)
            if m.shape[0] == T_total:
                mask = m
                break
    if mask is None or mask.shape[0] != T_total:
        return None
    return [int(mask[frame_cu[i]:frame_cu[i + 1]].sum()) for i in range(len(frame_cu) - 1)]


# --------------------------------------------------------------------------- #
# 2. PCA (>=evr cumulative explained variance)
# --------------------------------------------------------------------------- #
def fit_pca(pooled, evr_threshold=0.95):
    """Fit PCA on ``pooled`` (M, d); keep k comps for cum-EVR >= threshold."""
    X = np.asarray(pooled, dtype=np.float64)
    mean = X.mean(0)
    Xc = X - mean
    _u, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = s ** 2
    evr = var / var.sum()
    cum = np.cumsum(evr)
    k = int(np.searchsorted(cum, evr_threshold) + 1)
    k = max(1, min(k, Vt.shape[0]))
    return {
        "mean": mean,
        "components": Vt[:k],  # (k, d)
        "k": k,
        "evr": float(cum[k - 1]),
        "evr_threshold": float(evr_threshold),
    }


def project(tokens, pca):
    """Project apex tokens (n, d) into the PCA space -> (n, k)."""
    return (np.asarray(tokens, dtype=np.float64) - pca["mean"]) @ pca["components"].T


# --------------------------------------------------------------------------- #
# 3. DTW (euclidean per-step cost, normalized by warp-path length)
# --------------------------------------------------------------------------- #
def dtw_normalized(A, B):
    """Standard DP DTW between trajectories A (n,k) and B (m,k) with euclidean
    per-step cost. Returns accumulated cost / warp-path length so different
    episode lengths are comparable. Low = the two latent paths align well."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n, m = len(A), len(B)
    if n == 0 or m == 0:
        return float("nan")
    # pairwise euclidean cost (n, m)
    D = np.sqrt(np.maximum(
        (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T, 0.0))
    C = np.full((n + 1, m + 1), np.inf)
    C[0, 0] = 0.0
    for i in range(1, n + 1):
        Di = D[i - 1]
        Ci, Cim1 = C[i], C[i - 1]
        for j in range(1, m + 1):
            Ci[j] = Di[j - 1] + min(Cim1[j], Ci[j - 1], Cim1[j - 1])
    # backtrack the optimal warp path to get its length
    i, j, L = n, m, 0
    while i > 0 and j > 0:
        L += 1
        diag, up, left = C[i - 1, j - 1], C[i - 1, j], C[i, j - 1]
        if diag <= up and diag <= left:
            i, j = i - 1, j - 1
        elif up <= left:
            i -= 1
        else:
            j -= 1
    L += i + j  # any remaining edge steps
    return float(C[n, m] / max(L, 1))


# --------------------------------------------------------------------------- #
# High-level: apex tokens -> per-pair normalized DTW
# --------------------------------------------------------------------------- #
def compute_dtw_from_apex(apex_by_emb, evr_threshold=0.95,
                          emb_a=CIRCLE_EMB, emb_b=SMALL_EMB, good_eps=None):
    """apex_by_emb (from collect_apex_tokens) -> results dict.

    Fits ONE PCA on the pooled apex tokens over BOTH embs + ALL episodes, then
    for each aligned pair (emb_a ep i <-> emb_b ep i) reports normalized DTW.
    """
    if emb_a not in apex_by_emb or emb_b not in apex_by_emb:
        raise RuntimeError(
            f"need both embs {emb_a} & {emb_b}; got {sorted(apex_by_emb)}")
    a_eps = apex_by_emb[emb_a]["eps"]
    b_eps = apex_by_emb[emb_b]["eps"]

    pooled = np.concatenate(
        [np.concatenate(a_eps, 0), np.concatenate(b_eps, 0)], 0)
    pca = fit_pca(pooled, evr_threshold)

    n_pairs = min(len(a_eps), len(b_eps))
    # Whitening: divide each retained PCA coordinate by the pooled per-dim std
    # so DTW distances are in units of the latent cloud's own spread -- the
    # unwhitened numbers are NOT comparable across models (scale leaks through
    # PCA). ``dtw`` keeps the legacy unwhitened value; ``dtw_w`` is the
    # whitened one; ``align_ratio`` = whitened aligned-pair DTW / mean
    # mismatched-pair DTW (scale-free null: <<1 = real alignment, ~1 = none).
    proj_a = [project(e, pca) for e in a_eps[:n_pairs]]
    proj_b = [project(e, pca) for e in b_eps[:n_pairs]]
    _std = np.concatenate(proj_a + proj_b, 0).std(0) + 1e-8
    wa = [pa / _std for pa in proj_a]
    wb = [pb / _std for pb in proj_b]
    per_pair = []
    for i in range(n_pairs):
        d_w = dtw_normalized(wa[i], wb[i])
        null = [dtw_normalized(wa[i], wb[j]) for j in range(n_pairs) if j != i]
        per_pair.append({
            "pair": i,
            "dtw": dtw_normalized(proj_a[i], proj_b[i]),
            "dtw_w": d_w,
            "null_w": float(np.mean(null)) if null else float("nan"),
            "align_ratio": float(d_w / np.mean(null)) if null else float("nan"),
            "len_a": len(a_eps[i]),
            "len_b": len(b_eps[i]),
        })
    if good_eps is None:
        good_eps = list(range(n_pairs))
    sel = [p["dtw"] for p in per_pair if p["pair"] in set(good_eps)]
    return {
        "pca_k": pca["k"],
        "pca_evr": pca["evr"],
        "pca_dim": pooled.shape[1],
        "per_pair": per_pair,
        "good_eps": list(good_eps),
        "mean_dtw_good": float(np.mean(sel)) if sel else float("nan"),
        "mean_dtw_all": float(np.mean([p["dtw"] for p in per_pair])) if per_pair else float("nan"),
        "mean_dtw_w_good": float(np.mean([p["dtw_w"] for p in per_pair if p["pair"] in set(good_eps)])) if per_pair else float("nan"),
        "mean_dtw_w_all": float(np.mean([p["dtw_w"] for p in per_pair])) if per_pair else float("nan"),
        "mean_align_ratio_good": float(np.mean([p["align_ratio"] for p in per_pair if p["pair"] in set(good_eps)])) if per_pair else float("nan"),
        "mean_align_ratio_all": float(np.mean([p["align_ratio"] for p in per_pair])) if per_pair else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Pairing verification — read RAW goal + obj-init straight from the zarrs
# (model/normalization-agnostic; the val loader reads these same folders in
# sorted-filename order with no drops for the 300-frame >> min_seq_len=64 eps).
# --------------------------------------------------------------------------- #
def verify_pairing(config_path, n_pairs, tol=1e-3):
    full = OmegaConf.load(config_path)
    vd = full.data.valid_datasets
    folders = {}
    for name, spec in vd.items():
        folders[name] = str(spec.resolver.folder_path)
    circle_f = next((f for n, f in folders.items() if "small" not in n), None)
    small_f = next((f for n, f in folders.items() if "small" in n), None)
    if circle_f is None or small_f is None:
        print(f"[verify] could not resolve circle/small folders from {folders}")
        return None
    try:
        import zarr
    except Exception as exc:  # noqa: BLE001
        print(f"[verify] zarr unavailable ({exc}); skipping raw pairing check")
        return None

    def _read(folder, n):
        eps = sorted(glob.glob(os.path.join(folder, "*.zarr")))[:n]
        recs = []
        for e in eps:
            g = zarr.open(e, mode="r")
            recs.append((
                os.path.basename(e),
                np.asarray(g["goal_pose"][0], dtype=np.float64),
                np.asarray(g["observations.state"][0], dtype=np.float64),
            ))
        return recs

    cr = _read(circle_f, n_pairs)
    sm = _read(small_f, n_pairs)
    print(f"\n=== PAIRING VERIFICATION (raw zarr, sorted-filename order) ===")
    print(f"  circle: {circle_f}")
    print(f"  small : {small_f}")
    ok = True
    for i in range(min(len(cr), len(sm))):
        cn, cg, cs = cr[i]
        sn, sg, ss = sm[i]
        obj_l2 = float(np.linalg.norm(cs[OBJ_SLICE] - ss[OBJ_SLICE]))
        goal_l2 = float(np.linalg.norm(cg - sg))
        match = obj_l2 <= tol and goal_l2 <= tol
        ok = ok and match
        print(f"  pair {i:2d}: {cn} <-> {sn} | obj-init L2={obj_l2:.2e} "
              f"goal L2={goal_l2:.2e} | pusher_xy c={cs[:2].round(3).tolist()} "
              f"s={ss[:2].round(3).tolist()} | {'OK' if match else 'MISMATCH'}")
    print(f"  --> index-pairing {'VERIFIED' if ok else 'FAILED — datasets NOT aligned by index'}")
    return ok


# --------------------------------------------------------------------------- #
# Orchestration / CLI
# --------------------------------------------------------------------------- #
def run_dtw_eval(ckpt, config_path, n_episodes=6, evr=0.95, good_eps=None,
                 verify=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo, _ = load_algo_from_ckpt(ckpt, config_path)
    algo.nets = algo.nets.to(device)
    algo.device = device
    algo.nets.eval()

    full = OmegaConf.load(config_path)
    dm = instantiate(full.data)
    dm.setup(stage="validate")
    first = next(iter(dm.val_dataloader()))
    batch = first[0] if isinstance(first, tuple) else first
    batch = algo.process_batch_for_training(batch)

    apex = collect_apex_tokens(algo, batch, n_episodes=n_episodes)
    for e, rec in sorted(apex.items()):
        lens = [len(x) for x in rec["eps"]]
        print(f"emb{e}: apex d={rec['d']} compute_stages={rec['n_stages']} "
              f"episodes={len(rec['eps'])} chunk_lens={lens}")

    pairing_ok = verify_pairing(config_path, n_episodes) if verify else None

    res = compute_dtw_from_apex(apex, evr_threshold=evr, good_eps=good_eps)
    res["pairing_verified"] = pairing_ok

    print(f"\n=== DTW-LATENT (apex agnostic tokens; emb{CIRCLE_EMB} circle <-> "
          f"emb{SMALL_EMB} small-circle) ===")
    print(f"  PCA: k={res['pca_k']} comps for cum-EVR={res['pca_evr']:.4f} "
          f">= {evr}  (apex dim={res['pca_dim']})")
    for p in res["per_pair"]:
        print(f"  pair {p['pair']:2d}: norm-DTW={p['dtw']:.4f}  "
        print(f"          whitened={p.get('dtw_w', float('nan')):.4f}  null={p.get('null_w', float('nan')):.4f}  align_ratio={p.get('align_ratio', float('nan')):.4f}")
              f"(len circle={p['len_a']}, small={p['len_b']})")
    print(f"  good_eps={res['good_eps']}  mean norm-DTW (good)={res['mean_dtw_good']:.4f}  "
    print(f"  WHITENED mean (good)={res.get('mean_dtw_w_good', float('nan')):.4f} | (all)={res.get('mean_dtw_w_all', float('nan')):.4f}  ALIGN-RATIO (good)={res.get('mean_align_ratio_good', float('nan')):.4f} | (all)={res.get('mean_align_ratio_all', float('nan')):.4f}")
          f"| mean (all)={res['mean_dtw_all']:.4f}")
    return res


def _parse_eps(s):
    if not s:
        return None
    return [int(x) for x in str(s).replace(" ", "").split(",") if x != ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-path", required=True)
    ap.add_argument("--good-eps", default=None,
                    help="comma-separated pair indices to average, e.g. 0,2,4 "
                         "(default: all pairs)")
    ap.add_argument("--evr", type=float, default=0.95,
                    help="cumulative explained-variance-ratio threshold for k")
    ap.add_argument("--n-episodes", type=int, default=6)
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the raw-zarr pairing verification")
    args = ap.parse_args()
    run_dtw_eval(
        args.ckpt, args.config_path,
        n_episodes=args.n_episodes, evr=args.evr,
        good_eps=_parse_eps(args.good_eps), verify=not args.no_verify,
    )
    print("DTW_LATENT_DONE")


if __name__ == "__main__":
    main()
