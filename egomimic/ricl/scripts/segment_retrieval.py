"""
Segment-scoped leave-one-segment-out (LOSO) retrieval caches.

Unlike ``retrieval.py`` (which scopes retrieval at the *episode* level via
``human_robot_pairs.json``), this builds a cache whose pool is the set of frames
**inside selected action segments**. An episode is split into ``units`` —
``[start, end)`` frame spans — by ``action_segments.csv`` (one CSV row per
segment, columns ``episode,scene,start,end,verb,object,hand,...``). A ``--filter``
lambda over the row dict selects which segments enter the pool.

Bank and query are the *same* in-segment pool. Retrieval is per-frame, with
**leave-one-segment-out**: a query frame may not retrieve frames from its own
segment (``--loso-scope segment``) or its own episode (``--loso-scope episode``).
Self-retrieval is therefore excluded for free (a frame always shares its own
segment/episode key).

This is offline only — the output is a ``RetrievalCache`` in the exact on-disk
format ``retrieval.py`` produces (``manifest.json`` + ``q_<hash>.npz``), keyed by
**absolute** frame index, so it plugs into the same training/eval data path.
Out-of-segment query frames get padding entries (``""`` / ``-1`` / ``inf``). Build
several caches by re-running with different ``--filter`` / ``--segments-csv``.

Vectors come from the consolidated index built by ``build_embedding_index.py``
(``--index <dir>``; recommended) or, alternatively, per-episode zarr stores
(``--zarr-root '<root>/{hash}'``).

Example:
    python egomimic/ricl/scripts/segment_retrieval.py \\
        --index   egomimic/ricl/outputs/ricl_embeddings/pickplace_eva/_index \\
        --segments-csv egomimic/ricl/episode_lists/action_segments.csv \\
        --filter  "lambda r: r['object'] == 'blue marker'" \\
        --loso-scope segment -k 4 \\
        --out     egomimic/ricl/outputs/ricl_caches/blue_marker_loso
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np

from egomimic.ricl.retrieval import (
    DINOV2_ZARR_KEY,
    RetrievalCache,
    make_tree,
    make_zarr_dinov2_provider,
    pool_patch_embeddings,
)

logger = logging.getLogger(__name__)

# this file lives in egomimic/ricl/scripts/; episode_lists is a sibling of scripts/
DEFAULT_SEGMENTS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "episode_lists", "action_segments.csv"
)


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


def _compile_filter(filter_str: str | None):
    """Compile a ``"lambda r: ..."`` predicate string into a callable (or None)."""
    if not filter_str:
        return None
    fn = eval(filter_str)  # noqa: S307 - trusted CLI input, mirrors DatasetFilter
    if not callable(fn):
        raise ValueError(f"--filter did not evaluate to a callable: {filter_str!r}")
    return fn


def load_segments(
    csv_path: str, filter_fn=None
) -> dict[str, list[tuple[int, int, str]]]:
    """Parse ``action_segments.csv`` -> ``{episode_hash: [(start, end, seg_id)]}``.

    ``filter_fn(row_dict)`` (optional) keeps only matching rows. ``seg_id`` is
    ``"<episode>:<start>:<end>"`` (one CSV row == one segment). Spans are
    half-open ``[start, end)``.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    required = {"episode", "start", "end"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns {missing}")

    segments: dict[str, list[tuple[int, int, str]]] = {}
    n_rows = n_kept = 0
    for row in df.to_dict(orient="records"):
        n_rows += 1
        if filter_fn is not None and not filter_fn(row):
            continue
        n_kept += 1
        h = str(row["episode"])
        start, end = int(row["start"]), int(row["end"])
        segments.setdefault(h, []).append((start, end, f"{h}:{start}:{end}"))
    logger.info(
        "segments: kept %d/%d rows -> %d episodes, %d segments",
        n_kept,
        n_rows,
        len(segments),
        sum(len(v) for v in segments.values()),
    )
    if not segments:
        raise SystemExit("--filter matched no segments")
    return segments


# ---------------------------------------------------------------------------
# Vector sources: (T, D) pooled descriptors per episode, padding stripped
# ---------------------------------------------------------------------------


class IndexVectorSource:
    """Read pooled descriptors from a consolidated index (build_embedding_index)."""

    def __init__(self, index_dir: str):
        with open(os.path.join(index_dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        self.vectors = np.load(
            os.path.join(index_dir, "vectors.npy"), mmap_mode="r"
        )  # (N, D), pooled + padding-stripped
        self.ranges = {
            e["hash"]: (int(e["start"]), int(e["count"]))
            for e in self.manifest["episodes"]
        }

    def episode_vectors(self, episode_hash: str) -> np.ndarray:
        if episode_hash not in self.ranges:
            raise KeyError(episode_hash)
        start, count = self.ranges[episode_hash]
        return np.asarray(self.vectors[start : start + count], dtype=np.float32)

    @property
    def meta(self) -> dict:
        return {
            "vector_source": "index",
            "pool": self.manifest.get("pool"),
            "l2_normalize": self.manifest.get("l2_normalize"),
            "embed_dim": self.manifest.get("embed_dim"),
        }


class ZarrVectorSource:
    """Pool per-episode DINOv2 patch tokens on the fly from a {hash} zarr root."""

    def __init__(
        self,
        zarr_root: str,
        zarr_key: str = DINOV2_ZARR_KEY,
        pool: str = "64patches",
        l2_normalize: bool = False,
        frame_chunk_size: int = 256,
    ):
        self._provider = make_zarr_dinov2_provider(
            lambda h: zarr_root.format(hash=h),
            zarr_key=zarr_key,
            pool=pool,
            l2_normalize=l2_normalize,
        )
        self.zarr_root = zarr_root
        self.zarr_key = zarr_key
        self.pool = pool
        self.l2_normalize = l2_normalize
        self.frame_chunk_size = int(frame_chunk_size)
        self._meta = {
            "vector_source": "zarr",
            "zarr_root": zarr_root,
            "pool": pool,
            "l2_normalize": l2_normalize,
        }

    def episode_vectors(self, episode_hash: str) -> np.ndarray:
        try:
            return np.asarray(self._provider(episode_hash), dtype=np.float32)
        except (FileNotFoundError, KeyError) as e:
            raise KeyError(episode_hash) from e

    def _open_array(self, episode_hash: str):
        import zarr

        store = zarr.open(self.zarr_root.format(hash=episode_hash), mode="r")
        return store, store[self.zarr_key]

    def episode_length(self, episode_hash: str) -> int:
        try:
            store, arr = self._open_array(episode_hash)
        except (FileNotFoundError, KeyError) as e:
            raise KeyError(episode_hash) from e
        return int(store.attrs.get("total_frames", arr.shape[0]))

    def episode_frame_vectors(
        self, episode_hash: str, frames: np.ndarray
    ) -> np.ndarray:
        try:
            _, arr = self._open_array(episode_hash)
        except (FileNotFoundError, KeyError) as e:
            raise KeyError(episode_hash) from e

        frames = np.asarray(frames, dtype=np.int64)
        out = []
        breaks = np.flatnonzero(np.diff(frames) != 1) + 1
        runs = np.split(frames, breaks)
        for run in runs:
            run_start = int(run[0])
            run_end = int(run[-1]) + 1
            for start in range(run_start, run_end, self.frame_chunk_size):
                end = min(run_end, start + self.frame_chunk_size)
                patches = np.asarray(arr[start:end])
                out.append(
                    pool_patch_embeddings(
                        patches, method=self.pool, l2_normalize=self.l2_normalize
                    )
                )
        return np.concatenate(out, axis=0).astype(np.float32, copy=False)

    @property
    def meta(self) -> dict:
        return dict(self._meta)


# ---------------------------------------------------------------------------
# Pool build + LOSO cache
# ---------------------------------------------------------------------------


def _build_frame_set(segments, source):
    """Gather in-segment frames across episodes into flat pool arrays + blocks.

    Returns ``(vectors (P,D), hashes (P,), frames (P,), seg_ids (P,),
    blocks)`` where ``blocks`` is a list of ``(hash, T, frames (n,), row_start, n)``
    used to scatter results back into per-episode ``(T, k)`` cache arrays.
    """
    vec_parts, hash_parts, frame_parts, seg_parts, blocks = [], [], [], [], []
    row_start = 0
    skipped: list[str] = []
    for h, units in segments.items():
        try:
            if hasattr(source, "episode_length"):
                T = int(source.episode_length(h))
                V = None
            else:
                V = source.episode_vectors(h)  # (T, D)
                T = int(V.shape[0])
        except KeyError:
            skipped.append(h)
            continue
        taken: set[int] = set()
        frames, seg_ids = [], []
        for start, end, seg_id in units:
            for f in range(max(0, start), min(T, end)):
                if f in taken:
                    continue
                taken.add(f)
                frames.append(f)
                seg_ids.append(seg_id)
        if not frames:
            continue
        frames_np = np.asarray(frames, dtype=np.int32)
        n = frames_np.shape[0]
        if hasattr(source, "episode_frame_vectors"):
            selected_vectors = source.episode_frame_vectors(h, frames_np)
        else:
            selected_vectors = V[frames_np]
        vec_parts.append(np.asarray(selected_vectors, dtype=np.float32))
        hash_parts.append(np.full(n, h, dtype=object))
        frame_parts.append(frames_np)
        seg_parts.append(np.asarray(seg_ids, dtype=object))
        blocks.append((h, T, frames_np, row_start, n))
        row_start += n

    if skipped:
        logger.warning(
            "%d/%d segment episodes had no vectors (not embedded?), skipped: %s%s",
            len(skipped),
            len(segments),
            skipped[:5],
            "..." if len(skipped) > 5 else "",
        )
    if not vec_parts:
        raise SystemExit("no in-segment frames had vectors; nothing to index")

    vectors = np.concatenate(vec_parts, axis=0)
    hashes = np.concatenate(hash_parts).astype("U")
    frames = np.concatenate(frame_parts).astype(np.int32)
    seg_ids = np.concatenate(seg_parts).astype("U")
    return vectors, hashes, frames, seg_ids, blocks


def _build_pool(segments, source):
    return _build_frame_set(segments, source)


def _torch_topk_dist(
    query_vectors: np.ndarray,
    pool_vectors: np.ndarray,
    k: int,
    chunk_size: int = 512,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Euclidean top-k using chunked torch matmul."""
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    pool = torch.as_tensor(
        np.ascontiguousarray(pool_vectors, dtype=np.float32), device=dev
    )
    pool_norm = (pool * pool).sum(dim=1)
    kk = int(min(k, pool.shape[0]))

    all_dist: list[np.ndarray] = []
    all_idx: list[np.ndarray] = []
    for start in range(0, query_vectors.shape[0], chunk_size):
        q_np = np.ascontiguousarray(
            query_vectors[start : start + chunk_size], dtype=np.float32
        )
        q = torch.as_tensor(q_np, device=dev)
        d2 = (q * q).sum(dim=1, keepdim=True) - 2.0 * (q @ pool.T) + pool_norm
        vals, idx = torch.topk(d2, k=kk, dim=1, largest=False, sorted=True)
        vals = torch.clamp(vals, min=0.0).sqrt()
        all_dist.append(vals.cpu().numpy())
        all_idx.append(idx.cpu().numpy())
        del q, d2, vals, idx

    return np.concatenate(all_dist, axis=0), np.concatenate(all_idx, axis=0)


def build_segment_loso_cache(
    segments,
    source,
    k: int,
    loso_scope: str = "segment",
    meta: dict | None = None,
    bank_segments=None,
    knn_backend: str = "tree",
    chunk_size: int = 512,
    torch_device: str = "auto",
) -> RetrievalCache:
    """Per-frame kNN over segment frames -> RetrievalCache.

    By default ``segments`` is both query and bank, giving true LOSO retrieval.
    Pass ``bank_segments`` to use a separate bank pool, e.g. query=train+valid
    and bank=train so held-out eval task frames never enter the retrieved
    context.
    """
    if loso_scope not in ("segment", "episode"):
        raise ValueError(
            f"loso_scope must be 'segment' or 'episode', got {loso_scope!r}"
        )

    bank_segments = segments if bank_segments is None else bank_segments
    pool_vectors, pool_hash, pool_frame, pool_seg, _ = _build_pool(
        bank_segments, source
    )
    query_vectors, query_hash, query_frame, query_seg, blocks = _build_frame_set(
        segments, source
    )
    P = pool_vectors.shape[0]
    Q = query_vectors.shape[0]

    exclude_keys = pool_seg if loso_scope == "segment" else pool_hash
    _, exclude_code = np.unique(exclude_keys, return_inverse=True)
    max_excl = int(np.bincount(exclude_code).max())
    # Querying k + the largest excluded block is exact: after removing at most
    # max_excl same-segment/same-episode rows, the true k external nearest
    # neighbors remain in the candidate set.
    K = int(min(P, k + max_excl))
    logger.info(
        "bank: %d frames, %d segments, %d episodes; query: %d frames, %d segments, %d episodes; K=%d (k=%d, max_excl=%d, backend=%s)",
        P,
        len(np.unique(pool_seg)),
        len(np.unique(pool_hash)),
        Q,
        len(np.unique(query_seg)),
        len(np.unique(query_hash)),
        K,
        k,
        max_excl,
        knn_backend,
    )

    if knn_backend == "torch":
        dist, idx = _torch_topk_dist(
            query_vectors,
            pool_vectors,
            k=K,
            chunk_size=chunk_size,
            device=torch_device,
        )
    elif knn_backend == "tree":
        tree = make_tree(pool_vectors)
        dist, idx = tree.query(query_vectors, k=K)
    else:
        raise ValueError(f"unknown knn_backend {knn_backend!r}")
    if idx.ndim == 1:  # k==1 edge (shouldn't happen: max_excl>=1)
        dist, idx = dist[:, None], idx[:, None]
    idx = idx.astype(np.int64)

    # per-query-row top-k after LOSO filtering
    res_bh = np.empty((Q, k), dtype=object)
    res_bh[:] = ""
    res_bf = np.full((Q, k), -1, dtype=np.int32)
    res_dd = np.full((Q, k), np.inf, dtype=np.float32)
    for i in range(Q):
        ci, di = idx[i], dist[i]
        query_exclude = query_seg[i] if loso_scope == "segment" else query_hash[i]
        bank_exclude = pool_seg[ci] if loso_scope == "segment" else pool_hash[ci]
        keep = bank_exclude != query_exclude
        ci, di = ci[keep], di[keep]
        m = min(k, ci.shape[0])
        if m:
            res_bh[i, :m] = pool_hash[ci[:m]]
            res_bf[i, :m] = pool_frame[ci[:m]]
            res_dd[i, :m] = di[:m]

    # scatter into per-episode (T, k) cache arrays (absolute frame index)
    cache = RetrievalCache(k=k, meta=meta)
    for h, T, frames_np, r0, n in blocks:
        rows = np.arange(r0, r0 + n)
        BH = np.empty((T, k), dtype=object)
        BH[:] = ""
        BF = np.full((T, k), -1, dtype=np.int32)
        DD = np.full((T, k), np.inf, dtype=np.float32)
        BH[frames_np] = res_bh[rows]
        BF[frames_np] = res_bf[rows]
        DD[frames_np] = res_dd[rows]
        cache.add(h, BH, BF, DD, group_id="segment_loso")
    return cache


# ---------------------------------------------------------------------------
# Smoke self-test
# ---------------------------------------------------------------------------


class _SyntheticVectorSource:
    """Per-episode (T, D) where each segment's frames cluster around a centroid."""

    def __init__(self, segments, dim=16, seed=0):
        self.dim = dim
        rng = np.random.default_rng(seed)
        self._V = {}
        for h, units in segments.items():
            T = max(e for _, e, _ in units)
            V = rng.standard_normal((T, dim)).astype(np.float32)
            for j, (s, e, _) in enumerate(units):
                centroid = (10.0 * (j + 1)) * np.ones(dim, dtype=np.float32)
                V[s:e] = centroid + 0.01 * rng.standard_normal((e - s, dim)).astype(
                    np.float32
                )
            self._V[h] = V

    def episode_vectors(self, h):
        if h not in self._V:
            raise KeyError(h)
        return self._V[h]

    @property
    def meta(self):
        return {"vector_source": "synthetic"}


def _smoke() -> None:
    import tempfile

    print("== segment LOSO retrieval smoke (numpy+scipy, no GPU/data) ==")
    segments = {
        "epA": [(0, 5, "epA:0:5"), (10, 14, "epA:10:14")],
        "epB": [(0, 6, "epB:0:6")],
        "epC": [(2, 7, "epC:2:7")],
    }
    src = _SyntheticVectorSource(segments)
    k = 3
    cache = build_segment_loso_cache(
        segments, src, k=k, loso_scope="segment", meta={"mode": "segment_loso", "k": k}
    )

    seg_of = {}  # (hash, frame) -> seg_id
    for h, units in segments.items():
        for s, e, sid in units:
            for f in range(s, e):
                seg_of[(h, f)] = sid

    n_checked = 0
    for h, units in segments.items():
        for s, e, sid in units:
            for f in range(s, e):
                bh, bf, dd = cache.neighbors(h, f)
                valid = [j for j in range(k) if bh[j] != ""]
                # LOSO: no neighbor shares the query's segment
                for j in valid:
                    assert seg_of[(bh[j], int(bf[j]))] != sid, (h, f, sid, bh[j], bf[j])
                # distances ascending over valid entries
                vd = dd[[j for j in valid]]
                assert np.all(np.diff(vd) >= -1e-5), (h, f, dd)
                n_checked += 1
    print(f"  LOSO + ordering: checked {n_checked} in-segment query frames, OK")

    # out-of-segment frame -> padding
    bh, bf, dd = cache.neighbors("epA", 7)  # frame 7 is between the two units
    assert np.all(bh == "") and np.all(bf == -1) and np.all(np.isinf(dd))
    print("  out-of-segment frame -> padding, OK")

    with tempfile.TemporaryDirectory() as d:
        cache.save(d)
        re = RetrievalCache.load(d)
        bh1, bf1, dd1 = cache.neighbors("epB", 0)
        bh2, bf2, dd2 = re.neighbors("epB", 0)
        assert np.array_equal(bh1, bh2) and np.array_equal(bf1, bf2)
        assert np.allclose(dd1, dd2)
    print("  save/load roundtrip: OK")
    print("ALL SMOKE CHECKS PASSED")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help="offline self-test and exit")
    p.add_argument(
        "--index",
        default=None,
        help="consolidated index dir (vectors.npy + manifest.json)",
    )
    p.add_argument(
        "--zarr-root",
        default=None,
        help="alternative to --index: {hash} template to per-episode zarr stores",
    )
    p.add_argument("--segments-csv", default=DEFAULT_SEGMENTS_CSV)
    p.add_argument(
        "--filter", default=None, help='"lambda r: ..." over the CSV row dict'
    )
    p.add_argument(
        "--query-filter",
        default=None,
        help=(
            '"lambda r: ..." for query segments. Defaults to --filter. Use with '
            "--bank-filter to keep eval query tasks out of the retrieval bank."
        ),
    )
    p.add_argument(
        "--bank-filter",
        default=None,
        help=(
            '"lambda r: ..." for bank segments. Defaults to --filter. If set, '
            "retrieval is query->bank instead of one shared LOSO pool."
        ),
    )
    p.add_argument("--loso-scope", default="segment", choices=["segment", "episode"])
    p.add_argument("-k", type=int, default=4)
    p.add_argument("--out", default=None, help="output cache dir")
    p.add_argument(
        "--pool", default="64patches", help="--zarr-root only (index is pre-pooled)"
    )
    p.add_argument("--l2-normalize", action="store_true", help="--zarr-root only")
    p.add_argument(
        "--knn-backend",
        default="tree",
        choices=["tree", "torch"],
        help="exact kNN implementation; torch is faster for DINOv2 descriptors",
    )
    p.add_argument("--chunk-size", type=int, default=512, help="torch query chunk size")
    p.add_argument(
        "--torch-device",
        default="auto",
        help="torch device for --knn-backend torch, e.g. auto/cuda/cpu",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    if args.smoke:
        _smoke()
        return

    if (args.index is None) == (args.zarr_root is None):
        raise SystemExit("provide exactly one vector source: --index OR --zarr-root")
    if args.out is None:
        raise SystemExit("--out is required")

    if args.index is not None:
        source = IndexVectorSource(args.index)
    else:
        source = ZarrVectorSource(
            args.zarr_root, pool=args.pool, l2_normalize=args.l2_normalize
        )

    query_filter = args.query_filter if args.query_filter is not None else args.filter
    bank_filter = args.bank_filter if args.bank_filter is not None else args.filter
    query_segments = load_segments(args.segments_csv, _compile_filter(query_filter))
    bank_segments = (
        None
        if bank_filter == query_filter
        else load_segments(args.segments_csv, _compile_filter(bank_filter))
    )

    meta = {
        "mode": "segment_loso",
        "k": args.k,
        "loso_scope": args.loso_scope,
        "segments_csv": args.segments_csv,
        "filter": args.filter,
        "query_filter": query_filter,
        "bank_filter": bank_filter,
        **source.meta,
    }
    cache = build_segment_loso_cache(
        query_segments,
        source,
        k=args.k,
        loso_scope=args.loso_scope,
        meta=meta,
        bank_segments=bank_segments,
        knn_backend=args.knn_backend,
        chunk_size=args.chunk_size,
        torch_device=args.torch_device,
    )
    cache.save(args.out)
    print(
        f"wrote segment LOSO cache for {len(cache.query_hashes)} query episodes -> {args.out}"
    )


if __name__ == "__main__":
    main()
