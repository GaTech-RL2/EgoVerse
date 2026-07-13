"""Export chunkviz data (strip + PCA + per-frame trajectory) to a portable .npz
for the local/remote Streamlit explorer. Runs on the cluster (needs the model)."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

from egomimic.eval.core.ckpt_loading import _MockTrainer, load_algo_from_ckpt
from egomimic.eval.probes.eval_boundary_strip import _compose_bprobs_to_frame_level

TRAJ_PX = 200      # downsized trajectory frame size
TRAJ_JPEG_Q = 70


def find_by_type(ev, tname):
    stack, seen = [ev], set()
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        if type(e).__name__ == tname:
            return e
        for s in list(getattr(e, "evals", []) or []):
            stack.append(s)
    return None


def per_frame_chunk_ids(bprobs, T_total):
    if not bprobs:
        return []
    outer_idx = None
    for i, (_p, m) in enumerate(bprobs):
        if m.shape[0] == T_total:
            outer_idx = i
            break
    if outer_idx is None:
        outer_idx = len(bprobs) - 1
    results = []
    outer_mask = bprobs[outer_idx][1].numpy().astype(bool)
    cur_map = np.maximum(np.cumsum(outer_mask) - 1, 0)
    results.append(("Stage 0", cur_map.copy()))
    sn = 1
    for inner_i in range(outer_idx - 1, -1, -1):
        mask_i = bprobs[inner_i][1].numpy().astype(bool)
        if mask_i.shape[0] == 0:
            continue
        local_chunk = np.maximum(np.cumsum(mask_i) - 1, 0)
        idx = np.clip(cur_map, 0, mask_i.shape[0] - 1)
        frame_super = local_chunk[idx]
        results.append((f"Stage {sn}", frame_super.copy()))
        cur_map = frame_super
        sn += 1
    return results


def jpeg_frames(frames):
    """list/array of (H,W,3) uint8 RGB -> object array of jpeg bytes."""
    out = []
    for fr in frames:
        small = cv2.resize(fr, (TRAJ_PX, TRAJ_PX), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, TRAJ_JPEG_Q])
        out.append(buf.tobytes() if ok else b"")
    return np.array(out, dtype=object)


def _pca2(X):
    """2D PCA scores of rows of X (centered)."""
    Xc = X - X.mean(0, keepdims=True)
    _u, _s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return np.round(Xc @ Vt[:2].T, 3).astype(np.float32)


def _project_out(X, dirs):
    """Project the column space of `dirs` (d, k) out of rows of X (n, d)."""
    if dirs is None or dirs.size == 0:
        return X
    Q, _ = np.linalg.qr(dirs)
    return X - (X @ Q) @ Q.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-path", required=True)
    ap.add_argument("--out", default="chunkviz_data.npz")
    ap.add_argument("--n-episodes", type=int, default=6)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    algo, _ = load_algo_from_ckpt(args.ckpt, args.config_path)
    algo.nets = algo.nets.to(device)
    algo.device = device
    algo.nets.eval()
    full = OmegaConf.load(args.config_path)
    dm = instantiate(full.data)
    dm.setup(stage="validate")
    first = next(iter(dm.val_dataloader()))
    batch = first[0] if isinstance(first, tuple) else first
    batch = algo.process_batch_for_training(batch)

    evaluator = instantiate(full.evaluator)
    traj_eval = find_by_type(evaluator, "HNetEvalVideo")
    if traj_eval is not None:
        traj_eval.trainer = _MockTrainer(".", device)
        traj_eval.model = algo

    # Consistent chunkviz interface (user rule 2026-07-04): ONE call on the
    # algo; every chunker level publishes bpred + viz_tokens via ctx.aux
    # (dual-stream levels publish the AGNOSTIC (A) chunk tokens only).
    viz = algo.collect_chunkviz(batch)
    bprob_by_emb = {e: v["levels"] for e, v in viz.items()}
    anchor_by_emb = {e: bool(v.get("anchor", False)) for e, v in viz.items()}
    pca_out = {
        e: {"inner_tokens": v["tokens"]}
        for e, v in viz.items()
        if v["tokens"] is not None
    } or None

    # trajectory frames (per-frame, episodes concatenated with 5-frame black sep)
    traj_imgs = {}
    if traj_eval is not None:
        traj_eval.max_videos = args.n_episodes
        try:
            _m, traj_imgs = traj_eval.compute_metrics_and_viz(batch)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] trajectory eval failed: {exc}")
            traj_imgs = {}

    if not traj_imgs:
        # Batchflow fallback: the old HNetEvalVideo doesn't run on PipelineAlgo.
        # Use the batch's own camera frames (full-rate, same grid as frame cu).
        import torch as _torch
        for _emb, _b in batch.items():
            _img = _b.get("front_img_1")
            if _img is None:
                _img = next((v for k, v in _b.items()
                             if "img" in str(k) and _torch.is_tensor(v)
                             and v.dim() == 4), None)
            if _img is not None:
                _arr = (_img.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
                        * 255.0).clip(0, 255).astype("uint8")
                traj_imgs[_emb] = _arr
        if traj_imgs:
            print(f"[traj fallback] batch obs frames: "
                  f"{[(e, v.shape) for e, v in traj_imgs.items()]}")
            for _emb, _fr in traj_imgs.items():
                _H = _fr.shape[1]
                def _paint(fr, xy, color):
                    px = ((xy + 1.0) * 0.5 * (_H - 1)).round().astype(int).clip(0, _H - 1)
                    for _t in range(min(len(fr), len(px))):
                        y, x = px[_t, 1], px[_t, 0]
                        fr[_t, max(y-1,0):y+2, max(x-1,0):x+2] = color
                _gt = viz[_emb].get("gt_frame")
                _pd = viz[_emb].get("pred_frame")
                if _gt is not None:
                    _paint(_fr, _gt, (0, 255, 0))
                if _pd is not None:
                    _paint(_fr, _pd, (255, 40, 40))
            print("[traj fallback] overlays painted (GT green, pred red)")

    data = {}
    emb_ids = sorted(bprob_by_emb)
    data["emb_ids"] = np.array(emb_ids, dtype=np.int64)
    for emb in emb_ids:
        bprobs = bprob_by_emb[emb]
        cu = batch[emb]["cu_seqlens"]
        T_total = int(cu[-1].item())
        stages = per_frame_chunk_ids(bprobs, T_total)
        comp = list(reversed(_compose_bprobs_to_frame_level(bprobs, T_total)))
        if anchor_by_emb.get(emb):
            # first (outermost) stage is the fixed stride ANCHOR -- composition
            # only, not a learned chunker: exclude from display.
            stages = stages[1:]
            comp = comp[1:]
        labels = [f"Stage {i}" for i in range(len(stages))]
        cids = np.stack([s[1] for s in stages], 0)
        probs = np.stack([comp[i][0].numpy()[:T_total] for i in range(len(stages))], 0)
        crisp = np.zeros_like(cids, dtype=bool)
        crisp[:, 0] = True
        crisp[:, 1:] = cids[:, 1:] != cids[:, :-1]
        top_cid = cids[-1]

        if pca_out is not None and emb in pca_out and "inner_tokens" in pca_out[emb]:
            tokens = pca_out[emb]["inner_tokens"]
            X = tokens - tokens.mean(0, keepdims=True)
            _u, _s, Vt = np.linalg.svd(X, full_matrices=False)
            xy = (X @ Vt[:2].T).astype(np.float32)
        else:
            # no PCA (first_token) -> dummy points so the viewer length matches
            xy = np.zeros((int(top_cid.max()) + 1, 2), dtype=np.float32)

        data[f"emb{emb}_pca_xy"] = xy
        data[f"emb{emb}_labels"] = np.array(labels)
        Nep = min(args.n_episodes, len(cu) - 1)
        data[f"emb{emb}_n_episodes"] = np.int64(Nep)

        # trajectory: split concatenated stack by seq_lens + 5-frame separators
        traj_stack = traj_imgs.get(emb)
        seq_lens = batch[emb]["seq_lens"].cpu().numpy() if "seq_lens" in batch[emb] else None
        traj_offset = 0
        for i in range(Nep):
            s, e = int(cu[i].item()), int(cu[i + 1].item())
            c = cids[:, s:e]
            c = c - c.min(axis=1, keepdims=True)
            data[f"emb{emb}_ep{i}_cid"] = c.astype(np.int32)
            data[f"emb{emb}_ep{i}_prob"] = probs[:, s:e].astype(np.float32)
            data[f"emb{emb}_ep{i}_crisp"] = crisp[:, s:e]
            data[f"emb{emb}_ep{i}_topcid"] = top_cid[s:e].astype(np.int64)
            # trajectory slice for this episode
            if traj_stack is not None and seq_lens is not None:
                T_b = int(seq_lens[i])
                ep_fr = traj_stack[traj_offset:traj_offset + T_b]
                traj_offset += T_b + 5  # 5-frame black separator
                if ep_fr.shape[0] > 0:
                    data[f"emb{emb}_ep{i}_traj"] = jpeg_frames(ep_fr)
        print(f"emb{emb}: stages={labels} T_total={T_total} n_chunks={xy.shape[0]} "
              f"episodes={Nep} traj={'yes' if traj_stack is not None else 'NO'}")

    # shared cross-embodiment PCA: ONE SVD over all embs' inner-trunk tokens ->
    # drives the viewer's compare/alignment view (pca_shared + per-emb offset).
    _shared, _shared_emb, _offs, _off, _ok = [], [], {}, 0, pca_out is not None
    for emb in emb_ids:
        _t = pca_out.get(emb, {}).get("inner_tokens") if _ok else None
        if _t is None:
            _ok = False
            break
        _t = np.asarray(_t, dtype=np.float32)
        _offs[emb] = _off
        _shared.append(_t)
        _shared_emb.append(np.full(_t.shape[0], emb))
        _off += _t.shape[0]
    if _ok and _shared:
        _allX = np.concatenate(_shared, 0).astype(np.float64)
        _lab = np.concatenate(_shared_emb)
        _g = _allX.mean(0)
        _uniq = sorted(set(int(e) for e in _lab.tolist()))
        _means = {e: _allX[_lab == e].mean(0) for e in _uniq}
        # raw shared PCA
        data["pca_shared_xy"] = _pca2(_allX)
        # mean-difference removal: project out the between-emb mean deviations
        _Dmd = np.stack([_means[e] - _g for e in _uniq], 1)  # (d, n_emb)
        data["pca_shared_meandiff_xy"] = _pca2(_project_out(_allX, _Dmd))
        # LDA/Fisher removal: project out top Sw^-1 Sb directions
        try:
            _d = _allX.shape[1]
            _Sw = np.zeros((_d, _d))
            for e in _uniq:
                _Xe = _allX[_lab == e] - _means[e]
                _Sw += _Xe.T @ _Xe
            _Sw += (1e-3 * np.trace(_Sw) / _d) * np.eye(_d)
            _Sb = sum(int((_lab == e).sum()) * np.outer(_means[e] - _g, _means[e] - _g)
                      for e in _uniq)
            _ev, _evec = np.linalg.eig(np.linalg.solve(_Sw, _Sb))
            _ord = np.argsort(-_ev.real)[:max(1, len(_uniq) - 1)]
            data["pca_shared_lda_xy"] = _pca2(_project_out(_allX, _evec[:, _ord].real))
        except Exception as _exc:  # noqa: BLE001
            print(f"[warn] LDA removal failed ({_exc}); skipping LDA variant")
        data["pca_shared_emb"] = _lab.astype(np.int64)
        for emb in emb_ids:
            data[f"emb{emb}_pca_offset"] = np.int64(_offs[emb])
        print(f"shared PCA: {len(data['pca_shared_xy'])} chunks "
              f"(raw + mean-diff + LDA removal)")

    np.savez_compressed(args.out, **data)
    print(f"wrote {args.out}")
    print("EXPORT_DONE")


if __name__ == "__main__":
    main()
