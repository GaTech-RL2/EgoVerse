"""DTW-latent alignment eval for BATCHFLOW ckpts (v3).

Tiers: APEX tokens (skipped if median len < 4 -- collapsed seam) and the
BOTTOM chunker level (largest token count; alive in all arms). Episode
pairing by stride-aware frame-length fingerprint (seed cu is stride-4).
"""
import argparse
import glob as globlib
import numpy as np
import torch
import zarr as zarrlib
from omegaconf import OmegaConf
from hydra.utils import instantiate

from egomimic.eval.core.ckpt_loading import load_algo_from_ckpt
from egomimic.eval.dtw_latent_eval import compute_dtw_from_apex, CIRCLE_EMB, SMALL_EMB


def file_index_map(folder, lens, strides=(1, 4)):
    files = sorted(globlib.glob(str(folder) + "/*.zarr"))
    raw = []
    for f in files:
        z = zarrlib.open(f, mode="r")
        raw.append(int(z.attrs.get("total_frames") or z["actions"].shape[0]))
    best = None
    for s in strides:
        m = {}
        ok = 0
        for i, L in enumerate(lens):
            cand = [k for k, r in enumerate(raw)
                    if L in (r // s, (r + s - 1) // s, r // s + 1)]
            if len(cand) == 1:
                m[i] = cand[0]
                ok += 1
        if best is None or ok > best[1]:
            best = (m, ok, s)
    m, ok, s = best
    print(f"  [pairing] stride={s}: matched {ok}/{len(lens)} episodes")
    return m


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

    folders = {}
    for k, v in cfg.data.get("valid_datasets", {}).items():
        fp = v["resolver"]["folder_path"]
        folders[SMALL_EMB if "small" in str(fp) else CIRCLE_EMB] = fp

    tiers = {"apex": {}, "bottom": {}}
    for emb_id, eb in batch.items():
        seed = algo._seed(emb_id, eb)
        seed = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in seed.items()}
        frame_cu = seed["cu_seqlens"].cpu().numpy().tolist()
        with torch.no_grad():
            b = algo.policy(seed)
        lens = [int(frame_cu[i + 1] - frame_cu[i])
                for i in range(min(len(frame_cu) - 1, a.n_episodes))]
        idx_map = file_index_map(folders[emb_id], lens)

        def split(toks, cu):
            return [toks[cu[i]:cu[i + 1]] for i in range(min(len(cu) - 1, a.n_episodes))]

        # apex tier
        toks = b["apex/tokens"].float().cpu().numpy()
        cu = None
        for k in b:
            if k.startswith("chunk/L") and k.endswith("/cu_seqlens"):
                c = b[k].cpu().numpy()
                if int(c[-1]) == toks.shape[0]:
                    cu = c
                    break
        if cu is not None:
            eps = split(toks, cu)
            tiers["apex"][emb_id] = (eps, idx_map)
        # bottom tier: chunk level with the LARGEST token count
        cand = []
        for k in b:
            if k.startswith("chunk/L") and k.endswith("/tokens"):
                lvl = k[: -len("/tokens")]
                t = b[k].float().cpu().numpy()
                c = b[lvl + "/cu_seqlens"].cpu().numpy()
                if int(c[-1]) == t.shape[0]:
                    cand.append((t.shape[0], t, c, lvl))
        if cand:
            tiers.setdefault("_bottom_cand", {})[emb_id] = {
                lvl: (n, t, c) for n, t, c, lvl in cand}
            tiers["_bottom_idx", emb_id] = idx_map

    # bottom tier: pick ONE level shared by both embs (max combined tokens,
    # matching dims) -- per-emb argmax can pick different pyramid levels.
    bc = tiers.pop("_bottom_cand", {})
    if len(bc) == 2:
        embs_b = sorted(bc)
        shared = [l for l in bc[embs_b[0]] if l in bc[embs_b[1]]
                  and bc[embs_b[0]][l][1].shape[1] == bc[embs_b[1]][l][1].shape[1]]
        if shared:
            lvl = max(shared, key=lambda l: bc[embs_b[0]][l][0] + bc[embs_b[1]][l][0])
            print(f"  [bottom] shared level {lvl}")
            for e in embs_b:
                n, t, c = bc[e][lvl]
                def _split(toks, cu):
                    return [toks[cu[i]:cu[i + 1]] for i in range(min(len(cu) - 1, a.n_episodes))]
                tiers["bottom"][e] = (_split(t, c), tiers[("_bottom_idx", e)])
    for k in [k for k in list(tiers) if isinstance(k, tuple)]:
        tiers.pop(k)
    good = [int(x) for x in a.good_eps.replace(" ", "").split(",") if x]
    for tier, d in tiers.items():
        if len(d) < 2:
            print(f"== {tier}: missing an emb, skipped")
            continue
        embs = sorted(d)
        common = sorted(set(d[embs[0]][1].values()) & set(d[embs[1]][1].values()))
        apex_by_emb = {}
        for e in embs:
            eps_list, imap = d[e]
            inv = {v: k for k, v in imap.items()}
            apex_by_emb[e] = {"eps": [eps_list[inv[fi]] for fi in common]}
        lens0 = [len(x) for x in apex_by_emb[embs[0]]["eps"]]
        print(f"== {tier}: pairs={common} lens(emb{embs[0]})={lens0}")
        if not common or int(np.median([len(x) for x in apex_by_emb[embs[0]]["eps"]])) < 4:
            print(f"== {tier}: DEGENERATE (median len < 4) -- N/A")
            continue
        res = compute_dtw_from_apex(apex_by_emb, evr_threshold=a.evr,
                                    good_eps=good, pair_labels=common)
        for k in ("pca_k", "mean_dtw_w_good", "mean_dtw_w_all",
                  "mean_align_ratio_good", "mean_align_ratio_all"):
            v = res.get(k)
            print(f"  {tier}/{k} = {v:.4f}" if isinstance(v, float) else f"  {tier}/{k} = {v}")
    print("BF_DTW_DONE")


if __name__ == "__main__":
    main()
