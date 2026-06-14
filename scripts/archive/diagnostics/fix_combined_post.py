"""Post-process: read per-emb videos, vstack, write combined with opencv."""
import numpy as np
import torch
import torchvision.io as tvio
from pathlib import Path

base = Path("/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hnet_cotrain/cotrain_combined_eval_ep399_v4_2026-05-24_16-58-27/videos/epoch_0")

rows = []
for emb in ["PUSHSHAPES_SIM", "PUSHSHAPES_SIM_STICK"]:
    emb_dir = base / emb
    mp4s = sorted(f.name for f in emb_dir.iterdir() if f.name.endswith(".mp4"))
    chunks = []
    for mp4 in mp4s:
        v, _, _ = tvio.read_video(str(emb_dir / mp4), pts_unit="sec")
        print(f"  {emb}/{mp4}: {v.shape}")
        chunks.append(v)
    # Pad chunks to same H/W before catting along time
    max_h = max(c.shape[1] for c in chunks)
    max_w = max(c.shape[2] for c in chunks)
    padded = []
    for c in chunks:
        n, h, w, ch = c.shape
        if h < max_h or w < max_w:
            p = torch.zeros(n, max_h, max_w, ch, dtype=c.dtype)
            p[:, :h, :w, :] = c
            padded.append(p)
        else:
            padded.append(c)
    row = torch.cat(padded, dim=0)
    print(f"{emb}: combined {row.shape}")
    rows.append(row)

# Pad rows to same N and W
max_n = max(r.shape[0] for r in rows)
max_w = max(r.shape[2] for r in rows)
padded_rows = []
for r in rows:
    n, h, w, c = r.shape
    if n < max_n:
        last = r[-1:].expand(max_n - n, -1, -1, -1)
        r = torch.cat([r, last], dim=0)
        n = max_n
    if w < max_w:
        r = torch.cat([r, torch.zeros(n, h, max_w - w, c, dtype=r.dtype)], dim=2)
    padded_rows.append(r)

combined = torch.cat(padded_rows, dim=1)
# Ensure even dims
n, h, w, c = combined.shape
if h % 2: combined = combined[:, :h-1]
if w % 2: combined = combined[:, :, :w-1]
print(f"Combined: {combined.shape}")

out = str(base / "combined_rows_fixed.mp4")
tvio.write_video(out, combined, fps=30, video_codec="h264")
print(f"Wrote {out}")
