"""Render a FULL episode at normal speed with subtask [pred]/[gt] overlays.

Unlike the eval-video pipeline (random sampled frames, one per batch item),
this plays every frame of one episode at its native fps, decoding the subtask
autoregressively every ``--stride`` frames and holding the ``[pred]`` text in
between; the ``[gt]`` line is the exact subtask span active at each frame.

Usage:
    python -m egomimic.scripts.episode_subtask_video \
        --ckpt <last.ckpt> --data-config eval_aria_sort \
        --embodiment human_bimanual [--episode-index 0 | --episode-hash H] \
        --stride 30 --out /path/out.mp4 [--max-frames 2400]
"""

import argparse
import json
import os


def wrap_text(text, width=76):
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-config", default="eval_aria_sort")
    ap.add_argument("--embodiment", default="human_bimanual")
    ap.add_argument("--episode-index", type=int, default=0)
    ap.add_argument("--episode-hash", type=str, default=None)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=2400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import cv2
    import hydra
    import numpy as np
    import torch
    import zarr

    from egomimic.pl_utils.pl_data_utils import annotation_collate
    from egomimic.pl_utils.pl_model import ModelWrapper
    from egomimic.utils.hydra_utils import load_config

    # ---- model (rollout.py's proven loading path) ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = ModelWrapper.load_from_checkpoint(
        args.ckpt, weights_only=False, map_location="cpu"
    )
    policy = policy.to(device)
    policy.eval()
    policy.model.device = device
    algo = policy.model
    print(f"[ok] ckpt loaded: {args.ckpt}")

    # ---- dataset: the same pipeline training/eval reads ----
    cfg = load_config(f"data/{args.data_config}")
    ds = hydra.utils.instantiate(cfg.valid_datasets[args.embodiment])
    keys = sorted(ds.datasets.keys())
    ep_key = (
        args.episode_hash
        if args.episode_hash is not None
        else keys[args.episode_index % len(keys)]
    )
    leaf = ds.datasets[ep_key]
    total = leaf.total_frames
    print(f"[ok] episode {ep_key}: {total} frames")

    # Fetch samples through the MultiDataset wrapper (normalization + the
    # extra fields process_batch expects) — indexing the leaf directly skips
    # both. index_map: global idx -> (episode, local frame); invert for ours.
    ep_global = {}
    for gi, (name, li) in enumerate(ds.index_map):
        if name == ep_key:
            ep_global[li] = gi

    # raw images + gt spans straight from the zarr
    g = zarr.open(str(leaf.episode_path), mode="r")
    imgs = g["images.front_1"]
    gt_spans = []
    if "annotations_subtask" in set(g.array_keys()):
        for x in g["annotations_subtask"][:]:
            d = json.loads(x.decode() if isinstance(x, (bytes, bytearray)) else str(x))
            if isinstance(d, dict):
                gt_spans.append((int(d["start_idx"]), int(d["end_idx"]), d["text"]))

    def gt_at(f):
        texts = [t for s, e, t in gt_spans if s <= f < e]
        return texts[0] if texts else None

    n = min(total, args.max_frames)

    def decode_at(f):
        """One AR subtask decode at frame ``f`` through the standard pipeline."""
        sample = ds[ep_global[f]]
        batch = annotation_collate([sample])
        with torch.no_grad():
            pb = algo.process_batch_for_training(
                {args.embodiment: batch}
            )
            preds = algo.forward_eval(pb)
        key = f"{args.embodiment}_subtask_pred"
        pred = preds.get(key, ["<no decode>"])
        prompt = pb[list(pb.keys())[0]].get("sampled_prompt", [""])[0]
        # strip the State block for a readable overlay
        prompt_show = prompt.split(", State:")[0]
        return prompt_show, (pred[0] if pred else "<none>")

    # ---- render ----
    raw0 = imgs[0]
    im0 = cv2.imdecode(
        np.frombuffer(bytes(raw0), np.uint8), cv2.IMREAD_COLOR
    )
    H, W = im0.shape[:2]
    pad = 110  # black text band below the frame
    vw = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H + pad)
    )
    prompt_show, pred = "", ""
    for f in range(n):
        if f % args.stride == 0:
            try:
                prompt_show, pred = decode_at(f)
            except Exception as e:  # keep rendering; mark decode failure
                pred = f"<decode error: {type(e).__name__}>"
            print(f"[{f}/{n}] pred={pred[:70]!r}", flush=True)
        im = cv2.imdecode(np.frombuffer(bytes(imgs[f]), np.uint8), cv2.IMREAD_COLOR)
        canvas = np.zeros((H + pad, W, 3), np.uint8)
        canvas[:H] = im
        gt = gt_at(f)
        y = H + 14
        for label, txt, col in (
            ("task", prompt_show, (200, 200, 200)),
            ("pred", pred, (80, 200, 255)),
            ("gt", gt if gt else "(none)", (120, 255, 120)),
        ):
            for line in wrap_text(f"[{label}] {txt}", 92)[:2]:
                cv2.putText(
                    canvas, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    col, 1, cv2.LINE_AA,
                )
                y += 15
        vw.write(canvas)
    vw.release()
    print(f"[done] wrote {args.out} ({n} frames @ {args.fps}fps)")


if __name__ == "__main__":
    main()
