"""
Build a single consolidated kNN index from a folder of per-episode DINOv2
embedding stores.

The embedding step (``egomimic/scripts/embedding_process/zarr_embedding.py``
with ``--output-root``) writes one zarr store per episode at ``<root>/<hash>``,
each holding the raw patch tokens ``observations.embeddings.dinov2.front_1``
(T, 256, 768). Those per-episode files are durable, resumable *storage* — they
are not scanned at query time.

This script does the one-time consolidation that makes kNN efficient: it reads
every episode store, pools each frame to the 64-patch descriptor
(``egomimic.ricl.retrieval.pool_patch_embeddings``, padding stripped via the
store's ``total_frames``), and stacks them into one contiguous matrix with a
parallel ``row -> (episode_hash, frame_idx)`` map. Repeated retrieval then loads
this one matrix instead of re-opening hundreds of stores.

Outputs under ``--out`` (default ``<root>/_index``):
  - ``vectors.npy``   (N_frames, D) float32   — pooled descriptors, mmap-friendly
  - ``refs.npz``      episode_hash (N,) <U, frame_idx (N,) int32
  - ``manifest.json`` {zarr_key, pool, l2_normalize, embed_dim, n_frames,
                       n_episodes, episodes:[{hash, start, count}]}

Use ``build_retrieval_index(out_dir)`` to load it back into a
``egomimic.ricl.retrieval.RetrievalIndex`` (cKDTree) for querying / cache build.

Example:
    python egomimic/ricl/scripts/build_embedding_index.py \\
        --root egomimic/ricl/outputs/ricl_embeddings/pickplace_eva
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np

from egomimic.ricl.retrieval import (
    DINOV2_ZARR_KEY,
    BankRefs,
    RetrievalIndex,
    pool_patch_embeddings,
)

logger = logging.getLogger(__name__)


def _episode_dirs(root: str, zarr_key: str) -> list[str]:
    """Hash-named subdirs of ``root`` whose store contains ``zarr_key``."""
    import zarr

    out: list[str] = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if name.startswith("_") or not os.path.isdir(path):
            continue
        try:
            store = zarr.open(path, mode="r")
            if zarr_key in store:
                out.append(name)
        except Exception:
            logger.warning("skipping unreadable store %s", path)
    return out


def _episode_len(store, zarr_key: str) -> int:
    """True frame count (strip the chunk zero-padding _write_numeric added)."""
    arr = store[zarr_key]
    return int(store.attrs.get("total_frames", arr.shape[0]))


def build_index(
    root: str,
    out_dir: str | None = None,
    zarr_key: str = DINOV2_ZARR_KEY,
    pool: str = "64patches",
    l2_normalize: bool = False,
) -> str:
    """Consolidate ``<root>/<hash>`` embedding stores into one index under ``out_dir``."""
    import zarr

    out_dir = out_dir or os.path.join(root, "_index")
    os.makedirs(out_dir, exist_ok=True)

    hashes = _episode_dirs(root, zarr_key)
    if not hashes:
        raise SystemExit(f"no episode stores with key {zarr_key!r} under {root}")
    logger.info("found %d episode stores under %s", len(hashes), root)

    # Pass 1: lengths + descriptor dim (pool the first frame of episode 0).
    lengths: list[int] = []
    for h in hashes:
        store = zarr.open(os.path.join(root, h), mode="r")
        lengths.append(_episode_len(store, zarr_key))
    n_frames = int(sum(lengths))

    store0 = zarr.open(os.path.join(root, hashes[0]), mode="r")
    probe = pool_patch_embeddings(
        np.asarray(store0[zarr_key][:1]), method=pool, l2_normalize=l2_normalize
    )
    embed_dim = int(probe.shape[1])
    logger.info("n_frames=%d embed_dim=%d", n_frames, embed_dim)

    # Pass 2: fill a memmap so we never hold the whole stack in RAM.
    vectors_path = os.path.join(out_dir, "vectors.npy")
    vectors = np.lib.format.open_memmap(
        vectors_path, mode="w+", dtype=np.float32, shape=(n_frames, embed_dim)
    )
    ref_hash = np.empty(n_frames, dtype=object)
    ref_frame = np.empty(n_frames, dtype=np.int32)
    episodes_meta: list[dict] = []

    row = 0
    for h, t in zip(hashes, lengths):
        store = zarr.open(os.path.join(root, h), mode="r")
        arr = np.asarray(store[zarr_key][:t])  # strip padding
        pooled = pool_patch_embeddings(arr, method=pool, l2_normalize=l2_normalize)
        if pooled.shape != (t, embed_dim):
            raise RuntimeError(
                f"{h}: pooled shape {pooled.shape} != ({t}, {embed_dim})"
            )
        vectors[row : row + t] = pooled
        ref_hash[row : row + t] = h
        ref_frame[row : row + t] = np.arange(t, dtype=np.int32)
        episodes_meta.append({"hash": h, "start": row, "count": t})
        row += t
        logger.info("  %s -> %d frames (rows %d:%d)", h, t, row - t, row)
    vectors.flush()

    np.savez(
        os.path.join(out_dir, "refs.npz"),
        episode_hash=np.asarray(ref_hash).astype("U"),
        frame_idx=ref_frame,
    )
    manifest = {
        "zarr_key": zarr_key,
        "pool": pool,
        "l2_normalize": l2_normalize,
        "embed_dim": embed_dim,
        "n_frames": n_frames,
        "n_episodes": len(hashes),
        "episodes": episodes_meta,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "wrote consolidated index: %d frames x %d dims over %d episodes -> %s",
        n_frames,
        embed_dim,
        len(hashes),
        out_dir,
    )
    return out_dir


def build_retrieval_index(out_dir: str) -> RetrievalIndex:
    """Load a consolidated index dir into a ``RetrievalIndex`` (cKDTree)."""
    vectors = np.load(os.path.join(out_dir, "vectors.npy"), mmap_mode="r")
    refs = np.load(os.path.join(out_dir, "refs.npz"))
    return RetrievalIndex(
        np.ascontiguousarray(vectors, dtype=np.float32),
        BankRefs(
            episode_hash=refs["episode_hash"].astype("U"),
            frame_idx=refs["frame_idx"].astype(np.int32),
        ),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        required=True,
        help="folder of per-episode embedding stores (<root>/<hash>)",
    )
    p.add_argument(
        "--out", default=None, help="output index dir (default <root>/_index)"
    )
    p.add_argument("--zarr-key", default=DINOV2_ZARR_KEY)
    p.add_argument("--pool", default="64patches")
    p.add_argument("--l2-normalize", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    build_index(
        args.root,
        out_dir=args.out,
        zarr_key=args.zarr_key,
        pool=args.pool,
        l2_normalize=args.l2_normalize,
    )


if __name__ == "__main__":
    main()
