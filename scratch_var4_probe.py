"""Ground-truth forward probe for bf_prdec_var4 seam/bottom router behavior.

Mimics egomimic/eval/explorer/export.py loading exactly (load_algo_from_ckpt +
instantiate(cfg.data) + first val batch + process_batch_for_training), runs the
packed forward in eval mode, and prints PER-WINDOW router stats for each emb:
  - window length (raw frames) + anchor token count (post TargetBuilder stride)
  - bottom router (chunk/L1, cos router on anchor grid): boundary count,
    prob min/mean/max, frac>0.5
  - seam router (chunk/L0, on the bottom's kept-token grid): same stats
  - apex token count (seam kept count; cross-checked vs apex/tokens rows)

--full-episodes: override every valid_datasets.*.max_seq_len to None IN MEMORY
(the original yaml on disk is never touched) so full episodes go in.
"""
import argparse

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

from egomimic.eval.core.ckpt_loading import load_algo_from_ckpt


def stats(p):
    if p.size == 0:
        return "n/a"
    return (f"min={p.min():.4f} mean={p.mean():.4f} max={p.max():.4f} "
            f"frac>0.5={(p > 0.5).mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-path", required=True)
    ap.add_argument("--full-episodes", action="store_true",
                    help="set valid_datasets.*.max_seq_len=None (in memory)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    algo, _ = load_algo_from_ckpt(args.ckpt, args.config_path)
    algo.nets = algo.nets.to(device)
    algo.device = device
    algo.nets.eval()

    full = OmegaConf.load(args.config_path)
    if args.full_episodes:
        for name in list(full.data.valid_datasets.keys()):
            old = full.data.valid_datasets[name].get("max_seq_len", None)
            full.data.valid_datasets[name].max_seq_len = None
            print(f"[override] valid_datasets.{name}.max_seq_len: {old} -> None")
    dm = instantiate(full.data)
    dm.setup(stage="validate")
    first = next(iter(dm.val_dataloader()))
    batch = first[0] if isinstance(first, tuple) else first
    batch = algo.process_batch_for_training(batch)

    mode = "FULL-EPISODES (max_seq_len=None)" if args.full_episodes else "AS-SHIPPED (config max_seq_len)"
    print(f"\n================ PROBE MODE: {mode} ================")

    for emb_id, _b in batch.items():
        seed = algo._seed(emb_id, _b)
        seed = {k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in seed.items()}
        raw_cu = seed["cu_seqlens"].detach().cpu().numpy().astype(int)
        with torch.no_grad():
            b = algo.policy(seed)

        anchor_cu = b["cu_seqlens"].detach().cpu().numpy().astype(int)

        idxs = sorted({int(k.split("/")[1][1:]) for k in b
                       if k.startswith("chunk/L") and k.endswith("/cu_seqlens")})
        lev = {}
        for i in idxs:
            p = b[f"chunk/L{i}/boundary_prob"].detach().float().cpu().numpy()
            if p.ndim == 2:
                p = p[..., 1]
            m = b[f"chunk/L{i}/boundary_mask"].detach().cpu().numpy().astype(bool)
            cu = b[f"chunk/L{i}/cu_seqlens"].detach().cpu().numpy().astype(int)
            lev[i] = dict(p=p, m=m, cu=cu)

        # identify bottom = level whose prob grid == anchor grid;
        # seam = level whose prob grid == bottom's kept grid.
        bot = next((i for i in idxs if len(lev[i]["p"]) == anchor_cu[-1]), None)
        assert bot is not None, f"no level on anchor grid (anchor T={anchor_cu[-1]}, " \
            f"lens={[len(lev[i]['p']) for i in idxs]})"
        bcu = lev[bot]["cu"]
        seam = next((i for i in idxs if i != bot and len(lev[i]["p"]) == bcu[-1]), None)
        assert seam is not None, f"no level on bottom-kept grid (kept T={bcu[-1]})"
        scu = lev[seam]["cu"]
        apex = b.get("apex/tokens")
        n_apex_total = None if apex is None else int(apex.shape[0])

        print(f"\n--- emb{emb_id} | windows={len(raw_cu)-1} | raw_frames={raw_cu[-1]} "
              f"| anchor_tokens={anchor_cu[-1]} | bottom=chunk/L{bot} seam=chunk/L{seam} "
              f"| apex/tokens rows={n_apex_total} (seam kept cu total={scu[-1]}) ---")
        hdr = (f"{'w':>3} {'rawT':>6} {'ancT':>6} | {'botBnd':>6} {'bot prob (anchor grid)':<52}"
               f"| {'seamIn':>6} {'seamBnd':>7} {'seam prob (bottom-kept grid)':<52}| {'apex':>5}")
        print(hdr)
        print("-" * len(hdr))
        tot = dict(rawT=0, ancT=0, botB=0, seamIn=0, seamB=0, apex=0)
        for w in range(len(raw_cu) - 1):
            rawT = raw_cu[w + 1] - raw_cu[w]
            a0, a1 = anchor_cu[w], anchor_cu[w + 1]
            pb = lev[bot]["p"][a0:a1]
            mb = lev[bot]["m"][a0:a1]
            k0, k1 = bcu[w], bcu[w + 1]
            ps = lev[seam]["p"][k0:k1]
            ms = lev[seam]["m"][k0:k1]
            n_apex = scu[w + 1] - scu[w]
            print(f"{w:>3} {rawT:>6} {a1-a0:>6} | {int(mb.sum()):>6} {stats(pb):<52}"
                  f"| {k1-k0:>6} {int(ms.sum()):>7} {stats(ps):<52}| {n_apex:>5}")
            tot["rawT"] += int(rawT); tot["ancT"] += int(a1 - a0)
            tot["botB"] += int(mb.sum()); tot["seamIn"] += int(k1 - k0)
            tot["seamB"] += int(ms.sum()); tot["apex"] += int(n_apex)
        print(f"TOT {tot['rawT']:>6} {tot['ancT']:>6} | {tot['botB']:>6} "
              f"{stats(lev[bot]['p']):<52}| {tot['seamIn']:>6} {tot['seamB']:>7} "
              f"{stats(lev[seam]['p']):<52}| {tot['apex']:>5}")
        if n_apex_total is not None and n_apex_total != tot["apex"]:
            print(f"[WARN] apex/tokens rows {n_apex_total} != seam kept total {tot['apex']}")

    print("\nPROBE_DONE")


if __name__ == "__main__":
    main()
