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
    strip_eval = find_by_type(evaluator, "BoundaryStripEval")
    pca_eval = find_by_type(evaluator, "PCATokenEval")
    traj_eval = find_by_type(evaluator, "HNetEvalVideo")
    for e in (strip_eval, pca_eval, traj_eval):
        if e is not None:
            e.trainer = _MockTrainer(".", device)
            e.model = algo

    bprob_by_emb = strip_eval._run_forward_and_collect_bprobs(batch)
    pca_out = pca_eval._capture_tokens(batch)

    # trajectory frames (per-frame, episodes concatenated with 5-frame black sep)
    traj_imgs = {}
    if traj_eval is not None:
        traj_eval.max_videos = args.n_episodes
        try:
            _m, traj_imgs = traj_eval.compute_metrics_and_viz(batch)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] trajectory eval failed: {exc}")
            traj_imgs = {}

    data = {}
    emb_ids = sorted(bprob_by_emb)
    data["emb_ids"] = np.array(emb_ids, dtype=np.int64)
    for emb in emb_ids:
        bprobs = bprob_by_emb[emb]
        cu = batch[emb]["cu_seqlens"]
        T_total = int(cu[-1].item())
        stages = per_frame_chunk_ids(bprobs, T_total)
        labels = [s[0] for s in stages]
        cids = np.stack([s[1] for s in stages], 0)
        comp = list(reversed(_compose_bprobs_to_frame_level(bprobs, T_total)))
        probs = np.stack([comp[i][0].numpy()[:T_total] for i in range(len(stages))], 0)
        crisp = np.zeros_like(cids, dtype=bool)
        crisp[:, 0] = True
        crisp[:, 1:] = cids[:, 1:] != cids[:, :-1]
        top_cid = cids[-1]

        tokens = pca_out[emb]["inner_tokens"]
        X = tokens - tokens.mean(0, keepdims=True)
        _u, _s, Vt = np.linalg.svd(X, full_matrices=False)
        xy = (X @ Vt[:2].T).astype(np.float32)

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

    np.savez_compressed(args.out, **data)
    print(f"wrote {args.out}")
    print("EXPORT_DONE")


if __name__ == "__main__":
    main()
