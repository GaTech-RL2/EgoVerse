"""Preview the intentionally-aligned eva<->aria pairs as side-by-side mp4s.

The ``true_pairs_alignment`` tier of ``human_robot_pairs.json`` records the same
pick-and-place scenes captured by robot (eva) and human (aria), but does NOT store
an explicit 1:1 link -- the note says "pair eva<->aria by capture time". So this
script pairs each aria episode with its nearest-in-capture-time eva episode and
renders a time-normalized side-by-side clip (robot left | human right). The Δt of
each pair is printed and burned into the video, so tight same-session pairs
(small Δt) are obvious vs the 2026-04-26 recollected aria (huge Δt).

Both episodes are resampled to the SAME number of frames, so if the pair is truly
aligned the grasp/lift/place happen at the same horizontal sweep position.

No embeddings / GPU needed -- decodes images.front_1 only.

Run (EgoVerse venv + this repo on PYTHONPATH):
  PYTHONPATH=/storage/project/r-dxu345-0/rco3/EgoVerse2 \\
    /storage/project/r-dxu345-0/rco3/EgoVerse/emimic/bin/python \\
    egomimic/ricl/scripts/preview_aligned_pairs.py --out egomimic/ricl/outputs/aligned_pairs_preview
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import cv2
import imageio
import numpy as np

from egomimic.ricl.retrieval import load_pairs
from egomimic.ricl.scripts.viz_retrieval import EpisodeStore

AGAO = "/storage/project/r-dxu345-0/agao81/pick_place"
TILE_HW = (360, 480)  # (h, w) per panel


def parse_ts(h: str) -> datetime:
    """'2026-04-14-03-33-23-804000' -> datetime."""
    y, mo, d, hh, mm, ss, us = h.split("-")
    return datetime(int(y), int(mo), int(d), int(hh), int(mm), int(ss), int(us))


def resample(n: int, N: int) -> list[int]:
    """N evenly-spaced indices spanning [0, n-1] (time-normalize an episode)."""
    if n <= 1:
        return [0] * N
    return [round(i * (n - 1) / (N - 1)) for i in range(N)]


def panel(img: np.ndarray, caption: str, color) -> np.ndarray:
    t = cv2.resize(img, (TILE_HW[1], TILE_HW[0]), interpolation=cv2.INTER_AREA)
    t = cv2.copyMakeBorder(t, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
    cv2.putText(
        t, caption, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA
    )
    return t


def render_pair(eva_h: str, aria_h: str, dt_min: float, out_mp4: str, fps: int) -> None:
    eva = EpisodeStore(
        os.path.join(AGAO, eva_h),
        "images.front_1",
        "observations.embeddings.dinov2.front_1",
    )
    aria = EpisodeStore(
        os.path.join(AGAO, aria_h),
        "images.front_1",
        "observations.embeddings.dinov2.front_1",
    )
    ne, na = len(eva), len(aria)
    N = min(240, max(ne, na))
    ei, ai = resample(ne, N), resample(na, N)
    sep = np.full((TILE_HW[0] + 30, 8, 3), 50, np.uint8)
    writer = imageio.get_writer(out_mp4, fps=fps, macro_block_size=None)
    for j in range(N):
        pct = int(100 * j / max(1, N - 1))
        le = panel(eva.image(ei[j]), f"ROBOT {eva_h[11:19]} {pct}%", (80, 160, 255))
        ra = panel(
            aria.image(ai[j]),
            f"HUMAN {aria_h[11:19]} dt={dt_min:.0f}m",
            (120, 230, 120),
        )
        writer.append_data(np.concatenate([le, sep, ra], axis=1))
    writer.close()
    print(f"  wrote {out_mp4}  ({N} frames, eva={ne}f aria={na}f, dt={dt_min:.1f}min)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out",
        default="egomimic/ricl/outputs/aligned_pairs_preview",
        help="output dir",
    )
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument(
        "--pairs",
        default=None,
        help="manual override: 'eva_hash:aria_hash,eva_hash:aria_hash'",
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.pairs:
        pairs = []
        for tok in args.pairs.split(","):
            e, a = tok.split(":")
            dt = abs((parse_ts(e) - parse_ts(a)).total_seconds()) / 60.0
            pairs.append((e, a, dt, "manual"))
    else:
        tpa = load_pairs()["tiers"]["true_pairs_alignment"]
        pairs = []
        for set_name, payload in tpa.items():
            if not isinstance(payload, dict):
                continue
            evas = sorted(e["episode_hash"] for e in payload["eva_bimanual"])
            arias = sorted(a["episode_hash"] for a in payload["aria_bimanual"])
            # pair each aria with its nearest-in-time eva
            for a in arias:
                ta = parse_ts(a)
                e = min(evas, key=lambda x: abs((parse_ts(x) - ta).total_seconds()))
                dt = abs((parse_ts(e) - ta).total_seconds()) / 60.0
                pairs.append((e, a, dt, set_name))

    print(f"pairing {len(pairs)} eva<->aria clips (nearest capture time):")
    for i, (e, a, dt, tag) in enumerate(pairs):
        flag = "  <-- same-session" if dt < 20 else "  (recollect? large dt)"
        print(f"  [{tag}] eva {e}  <->  aria {a}   dt={dt:.1f}min{flag}")

    print("\nrendering:")
    for e, a, dt, tag in pairs:
        name = f"{tag}_eva-{e[11:19]}_aria-{a[11:19]}_dt{dt:.0f}m.mp4"
        render_pair(e, a, dt, os.path.join(args.out, name), args.fps)
    print(f"\ndone -> {args.out}/")


if __name__ == "__main__":
    main()
