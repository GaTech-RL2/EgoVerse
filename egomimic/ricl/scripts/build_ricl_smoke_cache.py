"""Build a MINIMAL RICL retrieval cache for a training smoke test.

Reuses patch-token DINOv2 embeddings (key
``observations.embeddings.dinov2.front_1``, shape (T,256,768)) already present
in the ``agao81/pick_place`` collection (written by ``zarr_embedding.py
--transform dinov2``) — so no GPU embedding pass is needed here. Memory-safe:
pools the patch tokens in frame-chunks instead of loading whole multi-GB arrays.

Run (EgoVerse venv + EgoVerse2 on PYTHONPATH):
  PYTHONPATH=/storage/project/r-dxu345-0/rco3/EgoVerse2 \
    /storage/project/r-dxu345-0/rco3/EgoVerse/emimic/bin/python \
    egomimic/ricl/scripts/build_ricl_smoke_cache.py --groups 2 --out egomimic/ricl/outputs/ricl_cache_smoke
"""

import argparse
import os

import numpy as np
import zarr

from egomimic.ricl.retrieval import (
    build_cache,
    groups_from_pairs,
    load_pairs,
    pool_patch_embeddings,
)

AGAO = "/storage/project/r-dxu345-0/agao81/pick_place"
ZKEY = "observations.embeddings.dinov2.front_1"


def has_patch_emb(h: str) -> bool:
    p = os.path.join(AGAO, h)
    if not os.path.isdir(p):
        return False
    try:
        s = zarr.open(p, mode="r")
        return ZKEY in s and s[ZKEY].ndim == 3
    except Exception:
        return False


def chunked_provider(episode_hash: str) -> np.ndarray:
    """Read (T,256,768) patch tokens in frame-chunks, pool -> (T,49152)."""
    a = zarr.open(os.path.join(AGAO, episode_hash), mode="r")[ZKEY]
    T = a.shape[0]
    out = None
    step = 256
    for i in range(0, T, step):
        blk = pool_patch_embeddings(a[i : i + step])
        if out is None:
            out = np.empty((T, blk.shape[-1]), dtype=np.float32)
        out[i : i + blk.shape[0]] = blk
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=2, help="max covered groups")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--out", default="egomimic/ricl/outputs/ricl_cache_smoke")
    args = ap.parse_args()

    pairs = load_pairs()
    groups = groups_from_pairs(pairs, "cross_similar")

    covered = []
    for g in groups:
        if not all(has_patch_emb(b) for b in g.bank_hashes):
            continue
        qs = [q for q in g.query_hashes if has_patch_emb(q)]
        if not qs:
            continue
        g.query_hashes = qs
        covered.append(g)
        if len(covered) >= args.groups:
            break

    if not covered:
        raise SystemExit("no fully-covered cross_similar groups in agao81/pick_place")

    print(f"chosen {len(covered)} group(s):")
    for g in covered:
        print(f"  {g.group_id}  bank={g.bank_hashes}  queries={g.query_hashes}")

    cache = build_cache(
        covered, chunked_provider, k=args.k, meta={"mode": "cross_similar", "k": args.k}
    )
    cache.save(args.out)
    print(f"\nwrote cache: {len(cache.query_hashes)} query episodes -> {args.out}")
    for qh in cache.query_hashes:
        bh, bf, dd = cache.neighbors(qh, 0)
        print(
            f"  q={qh} frame0 -> bank={list(bh)} frame={list(bf)} dist={np.round(dd,3).tolist()}"
        )


if __name__ == "__main__":
    main()
