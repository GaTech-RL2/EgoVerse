"""DTW-latent alignment eval for BATCHFLOW ckpts.

Collector: run the pipeline per emb on the aligned probe subsets, read
``apex/tokens`` + the chunk-space cu whose total matches the apex token count,
split per episode. PCA fit on ALL episodes pooled; norm-DTW reported on the
good episodes [3,7,0] and on all pairs (protocol per user / dtw_latent_eval).
Core math (fit_pca / dtw_normalized / compute_dtw_from_apex) reused unchanged.
"""
import argparse
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

from egomimic.eval.core.ckpt_loading import load_algo_from_ckpt
from egomimic.eval.dtw_latent_eval import compute_dtw_from_apex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-path", required=True)
    ap.add_argument("--n-episodes", type=int, default=6)
    ap.add_argument("--evr", type=float, default=0.95)
    ap.add_argument("--good-eps", default="3,7,0")
    a = ap.parse_args()

    algo, _ = load_algo_from_ckpt(a.ckpt, a.config_path)
    algo.nets.cuda()
    algo.nets.eval()
    cfg = OmegaConf.load(a.config_path)
    dm = instantiate(cfg.data)
    dm.setup(stage="validate")
    first = next(iter(dm.val_dataloader()))
    batch = first[0] if isinstance(first, tuple) else first
    batch = algo.process_batch_for_training(batch)

    apex_by_emb = {}
    for emb_id, eb in batch.items():
        seed = algo._seed(emb_id, eb)
        seed = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in seed.items()}
        with torch.no_grad():
            b = algo.policy(seed)
        toks = b["apex/tokens"].float().cpu().numpy()
        M = toks.shape[0]
        cu = None
        for k in b:
            if k.startswith("chunk/L") and k.endswith("/cu_seqlens"):
                c = b[k].cpu().numpy()
                if int(c[-1]) == M:
                    cu = c
                    break
        assert cu is not None, f"no chunk cu matches apex token count {M}"
        eps = [toks[cu[i]:cu[i + 1]] for i in range(len(cu) - 1)][: a.n_episodes]
        apex_by_emb[emb_id] = eps
        print(f"[bfdtw] emb{emb_id}: {len(eps)} eps, apex lens {[len(e) for e in eps]}")

    good = [int(x) for x in a.good_eps.replace(" ", "").split(",") if x] or None
    res = compute_dtw_from_apex(apex_by_emb, evr_threshold=a.evr, good_eps=good)
    for k, v in res.items():
        if isinstance(v, (int, float)):
            print(f"  {k} = {v:.4f}" if isinstance(v, float) else f"  {k} = {v}")
        elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
            print(f"  {k} = {[round(float(x), 4) for x in v]}")
    print("BF_DTW_DONE")


if __name__ == "__main__":
    main()
