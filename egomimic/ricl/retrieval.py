"""RICL retrieval index: DINOv2(base_0_rgb) -> cKDTree -> per-query top-k cache. (P1)

Pipeline
--------
1. For every bank + query episode, take per-frame DINOv2 patch tokens of the
   ``base_0_rgb`` view (zarr ``observations.embeddings.dinov2.front_1``,
   shape ``(T, 256, 768)``) and pool the 16x16 patch grid into an 8x8 grid of
   super-patches, concatenated to a ``(T, 49152)`` descriptor — ricl_openpi's
   ``EMBEDDING_TYPE='64PATCHES'``. Raw L2 metric, **no** L2-normalization
   (matching its autofaiss ``metric_type='l2'`` index).
2. Per *task group* (from ``human_robot_pairs.json``) build a ``cKDTree`` over the
   bank frames and, for each query frame, store the top-k nearest bank frames.
3. Save a cache keyed by query ``episode_hash`` that the RICL collate reads at
   train/eval time to fetch the retrieved (image, state, action) blocks.

Cross-embodiment (the headline): **bank = aria (human)**, **query = eva (robot)**.
Within-embodiment sanity (D0): bank = eva, query = eva.

Design notes
------------
- ``base_0_rgb`` is the only view both embodiments share (eva front exterior vs
  aria egocentric); see ``egomimic/algo/pi.py`` / ``rldb/embodiment/{eva,human}.py``.
- Heavy deps (torch for DINOv2, zarr/boto3 for data) are imported **lazily** so
  the pure index logic runs with just numpy+scipy. Run the self-test with
  ``python -m egomimic.ricl.retrieval --smoke`` (no GPU / no data).
- The encoder replicates ricl_openpi's reference
  (``torch.hub dinov2_vitb14``, resize-with-pad 224, ImageNet normalize,
  ``forward_features()['x_norm_patchtokens']``); the in-repo embed pass is
  ``egomimic/scripts/embedding_process/dinov2_embedding.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

PAIRS_JSON_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "scripts", "human_robot_pairs.json"
)
DINOV2_ZARR_KEY = "observations.embeddings.dinov2.front_1"  # patch tokens (T, 256, 768)
TOKEN_DIM = 768  # DINOv2 ViT-B/14 hidden dim
EMBED_DIM = 64 * TOKEN_DIM  # 64-super-patch descriptor (ricl_openpi 64PATCHES)

# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def pool_patch_embeddings(
    emb, method: str = "64patches", l2_normalize: bool = False
) -> np.ndarray:
    """Pool per-frame patch-token embeddings to a single per-frame descriptor.

    Args:
        emb: ``(T, N, D)`` patch tokens (DINOv2: N=256, D=768) or already-pooled
            ``(T, D')``.
        method: ``64patches`` | ``mean`` | ``max`` | ``cls`` (``cls`` = token 0).
            ``64patches`` replicates ricl_openpi's ``EMBEDDING_TYPE='64PATCHES'``:
            the sqrt(N) x sqrt(N) patch grid is averaged into an 8x8 grid of
            super-patches and concatenated -> ``(T, 64*D)``.
        l2_normalize: if True, unit-normalize each frame vector (cosine
            retrieval). ricl_openpi indexes RAW descriptors with an L2 metric,
            so the default is False.

    Returns:
        ``(T, 64*D)`` float32 for ``64patches``, else ``(T, D)``.
    """
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim == 3:
        if method == "64patches":
            t, n, d = emb.shape
            side = int(round(n**0.5))
            if side * side != n or side % 8 != 0:
                raise ValueError(
                    "64patches needs a square patch grid divisible into 8x8 "
                    f"blocks; got N={n}"
                )
            blk = side // 8
            # (T, 8, blk, 8, blk, D) -> mean each blk x blk block; row-major
            # block order concatenated along features == ricl_openpi
            # utils.embed()'s 64PATCHES loop.
            v = emb.reshape(t, 8, blk, 8, blk, d).mean(axis=(2, 4)).reshape(t, 64 * d)
        elif method == "mean":
            v = emb.mean(axis=1)
        elif method == "max":
            v = emb.max(axis=1)
        elif method == "cls":
            v = emb[:, 0, :]
        else:
            raise ValueError(f"unknown pool method {method!r}")
    elif emb.ndim == 2:
        v = emb
    else:
        raise ValueError(f"expected (T,N,D) or (T,D), got shape {emb.shape}")
    if l2_normalize:
        v = v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8, None)
    return np.ascontiguousarray(v, dtype=np.float32)


# ---------------------------------------------------------------------------
# kNN tree (scipy cKDTree with a numpy brute-force fallback)
# ---------------------------------------------------------------------------


class _BruteForceTree:
    """Minimal cKDTree-compatible fallback (used only if scipy is unavailable)."""

    def __init__(self, data: np.ndarray):
        self.data = np.ascontiguousarray(data, dtype=np.float32)

    def query(self, x, k: int = 1):
        x = np.atleast_2d(np.asarray(x, dtype=np.float32))
        # (Q, N) squared distances
        d2 = (
            (x * x).sum(1)[:, None]
            - 2.0 * x @ self.data.T
            + (self.data * self.data).sum(1)[None, :]
        )
        k = min(k, self.data.shape[0])
        idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        # sort the k by true distance
        order = np.argsort(np.take_along_axis(d2, idx, axis=1), axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        dist = np.sqrt(np.clip(np.take_along_axis(d2, idx, axis=1), 0.0, None))
        return dist, idx


def make_tree(vectors: np.ndarray):
    """Build a cKDTree (or brute-force fallback) over ``(N, D)`` vectors."""
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    try:
        from scipy.spatial import cKDTree

        return cKDTree(vectors)
    except Exception:  # pragma: no cover - scipy is a hard dep in EgoVerse
        return _BruteForceTree(vectors)


# ---------------------------------------------------------------------------
# Index: bank vectors + per-row (episode_hash, frame_idx) refs
# ---------------------------------------------------------------------------


@dataclass
class BankRefs:
    """Per-bank-row provenance, aligned with the rows fed to the tree."""

    episode_hash: np.ndarray  # (N,) <U
    frame_idx: np.ndarray  # (N,) int32

    def __len__(self) -> int:
        return int(self.episode_hash.shape[0])


class RetrievalIndex:
    """cKDTree over bank frame vectors + their (episode_hash, frame_idx) refs."""

    def __init__(self, vectors: np.ndarray, refs: BankRefs):
        assert vectors.shape[0] == len(refs), "vectors/refs length mismatch"
        self.dim = int(vectors.shape[1])
        self.refs = refs
        self.tree = make_tree(vectors)

    def query(self, query_vectors: np.ndarray, k: int):
        """Top-k bank rows for each query vector.

        Returns ``(hashes, frames, dists)`` each shaped ``(Q, k)``. When the bank
        has fewer than k rows the trailing columns are padded (``""`` / ``-1`` /
        ``inf``).
        """
        q = np.atleast_2d(np.asarray(query_vectors, dtype=np.float32))
        n = len(self.refs)
        kk = min(k, n)
        dist, idx = self.tree.query(q, k=kk)
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx).astype(np.int64)
        # scipy returns shape (Q,) when kk==1; normalize to (Q, kk)
        if dist.shape != (q.shape[0], kk):
            dist = dist.reshape(q.shape[0], kk)
            idx = idx.reshape(q.shape[0], kk)
        hashes = self.refs.episode_hash[idx]
        frames = self.refs.frame_idx[idx].astype(np.int32)
        if kk < k:  # pad to a fixed width k
            pad = k - kk
            hashes = np.pad(hashes, ((0, 0), (0, pad)), constant_values="")
            frames = np.pad(frames, ((0, 0), (0, pad)), constant_values=-1)
            dist = np.pad(dist, ((0, 0), (0, pad)), constant_values=np.inf)
        return hashes, frames, dist.astype(np.float32)


def index_from_episodes(
    episode_hashes: Sequence[str],
    vectors_provider: Callable[[str], np.ndarray],
) -> RetrievalIndex:
    """Stack pooled vectors from several bank episodes into one RetrievalIndex."""
    vecs, hsh, frm = [], [], []
    for h in episode_hashes:
        v = np.asarray(vectors_provider(h), dtype=np.float32)
        if v.ndim != 2:
            raise ValueError(
                f"provider for {h} returned shape {v.shape}, expected (T,D)"
            )
        vecs.append(v)
        hsh.append(np.full(v.shape[0], h, dtype=object))
        frm.append(np.arange(v.shape[0], dtype=np.int32))
    if not vecs:
        raise ValueError("no bank episodes provided")
    vectors = np.concatenate(vecs, axis=0)
    refs = BankRefs(
        episode_hash=np.asarray(np.concatenate(hsh)).astype("U"),
        frame_idx=np.concatenate(frm),
    )
    return RetrievalIndex(vectors, refs)


# ---------------------------------------------------------------------------
# Task groups from human_robot_pairs.json
# ---------------------------------------------------------------------------


@dataclass
class TaskGroup:
    """A retrieval scope: a bank (episodes searched) + queries (episodes scored)."""

    group_id: str
    bank_hashes: list[str]
    query_hashes: list[str]
    meta: dict = field(default_factory=dict)


def load_pairs(path: str = PAIRS_JSON_DEFAULT) -> dict:
    with open(path) as f:
        return json.load(f)


def groups_from_pairs(pairs: dict, mode: str, loeo: bool = False) -> list[TaskGroup]:
    """Build retrieval scopes from the pairs JSON.

    Modes:
      - ``cross_similar``  : Tier-2. bank = [aria_hash], query = its eva matches.
      - ``cross_alignment``: Tier-1. bank = aria set, query = eva set (per set).
      - ``within_alignment``: D0 sanity. bank = eva set, query = eva set (per set).

    ``loeo`` (``within_alignment`` only): leave-one-episode-out — expand each
    alignment set into one group per query whose bank is the set minus that
    episode, so no query can retrieve frames of itself. Without it every
    within_alignment query trivially finds its own episode at distance ~0.
    """
    if loeo and mode != "within_alignment":
        raise ValueError(
            "loeo only applies to within_alignment (the only mode whose "
            f"queries appear in their own bank); got mode={mode!r}"
        )
    groups: list[TaskGroup] = []
    if mode == "cross_similar":
        for entry in pairs["tiers"]["similar_task_language_pairs"]:
            matches = entry.get("matches", [])
            if not matches:
                continue
            groups.append(
                TaskGroup(
                    group_id=f"aria:{entry['aria_hash']}",
                    bank_hashes=[entry["aria_hash"]],
                    query_hashes=[m["eva_hash"] for m in matches],
                    meta={
                        "aria_scene": entry.get("aria_scene", []),
                        "shared_objects": {
                            m["eva_hash"]: m.get("shared_objects", []) for m in matches
                        },
                    },
                )
            )
    elif mode in ("cross_alignment", "within_alignment"):
        sets = pairs["tiers"]["true_pairs_alignment"]
        for set_name, payload in sets.items():
            if not isinstance(payload, dict) or "eva_bimanual" not in payload:
                continue  # skip the "note" string
            eva = [e["episode_hash"] for e in payload["eva_bimanual"]]
            aria = [e["episode_hash"] for e in payload.get("aria_bimanual", [])]
            bank = aria if mode == "cross_alignment" else eva
            if loeo:
                for qh in eva:
                    groups.append(
                        TaskGroup(
                            group_id=f"{mode}:{set_name}:loeo:{qh}",
                            bank_hashes=[h for h in bank if h != qh],
                            query_hashes=[qh],
                            meta={"set": set_name, "loeo": True},
                        )
                    )
            else:
                groups.append(
                    TaskGroup(
                        group_id=f"{mode}:{set_name}",
                        bank_hashes=bank,
                        query_hashes=eva,
                        meta={"set": set_name},
                    )
                )
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return groups


def merge_bank(groups: list[TaskGroup], group_id: str = "merged") -> list[TaskGroup]:
    """Collapse to a single group: union of all banks, union of all queries.

    Use for a global cross-embodiment bank (every query searches all human demos)
    instead of per-task-scoped retrieval.
    """
    bank = sorted({h for g in groups for h in g.bank_hashes})
    query = sorted({h for g in groups for h in g.query_hashes})
    return [TaskGroup(group_id=group_id, bank_hashes=bank, query_hashes=query)]


# ---------------------------------------------------------------------------
# Cache: per-query-episode top-k bank refs
# ---------------------------------------------------------------------------


@dataclass
class _QueryEntry:
    bank_hash: np.ndarray  # (T, k) <U
    bank_frame: np.ndarray  # (T, k) int32
    dist: np.ndarray  # (T, k) float32
    group_id: str


class RetrievalCache:
    """Maps a query (episode_hash, frame_idx) to its top-k bank refs.

    On-disk layout (``save(dir)``):
      ``manifest.json``        - {k, mode, pool, l2_normalize, queries:{hash:group_id}}
      ``q_<episode_hash>.npz`` - bank_hash (T,k) U, bank_frame (T,k) i4, dist (T,k) f4
    """

    def __init__(self, k: int, meta: dict | None = None):
        self.k = int(k)
        self.meta = dict(meta or {})
        self._entries: dict[str, _QueryEntry] = {}

    def add(self, query_hash, bank_hash, bank_frame, dist, group_id):
        self._entries[query_hash] = _QueryEntry(
            bank_hash=np.asarray(bank_hash).astype("U"),
            bank_frame=np.asarray(bank_frame, dtype=np.int32),
            dist=np.asarray(dist, dtype=np.float32),
            group_id=group_id,
        )

    @property
    def query_hashes(self) -> list[str]:
        return list(self._entries.keys())

    def neighbors(self, query_hash: str, frame_idx: int):
        """Return (bank_hash[k], bank_frame[k], dist[k]) for one query frame.

        ``frame_idx`` is clamped into range so a query chunk near the episode end
        still resolves (retrieval is per-observation, indexed by start frame).
        """
        e = self._entries[query_hash]
        i = int(np.clip(frame_idx, 0, e.bank_hash.shape[0] - 1))
        return e.bank_hash[i], e.bank_frame[i], e.dist[i]

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        manifest = {
            "k": self.k,
            "meta": self.meta,
            "queries": {h: e.group_id for h, e in self._entries.items()},
        }
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        for h, e in self._entries.items():
            np.savez(
                os.path.join(out_dir, f"q_{h}.npz"),
                bank_hash=e.bank_hash,
                bank_frame=e.bank_frame,
                dist=e.dist,
            )
        return out_dir

    @classmethod
    def load(cls, out_dir: str) -> "RetrievalCache":
        with open(os.path.join(out_dir, "manifest.json")) as f:
            manifest = json.load(f)
        cache = cls(k=manifest["k"], meta=manifest.get("meta", {}))
        for h, group_id in manifest["queries"].items():
            npz = np.load(os.path.join(out_dir, f"q_{h}.npz"), allow_pickle=False)
            cache.add(
                h, npz["bank_hash"], npz["bank_frame"], npz["dist"], group_id=group_id
            )
        return cache


def build_cache(
    groups: list[TaskGroup],
    vectors_provider: Callable[[str], np.ndarray],
    k: int,
    meta: dict | None = None,
) -> RetrievalCache:
    """Build per-group cKDTree indices and the top-k cache for all query frames.

    ``vectors_provider(episode_hash) -> (T, D)`` pooled, normalized vectors.
    A query episode appearing in multiple groups keeps the *first* group's
    neighbors (groups are processed in order); use ``merge_bank`` upstream to
    avoid that if a single global bank is desired.
    """
    cache = RetrievalCache(k=k, meta=meta)
    for g in groups:
        index = index_from_episodes(g.bank_hashes, vectors_provider)
        for qh in g.query_hashes:
            if qh in cache._entries:
                continue
            qv = np.asarray(vectors_provider(qh), dtype=np.float32)
            bh, bf, dd = index.query(qv, k=k)
            cache.add(qh, bh, bf, dd, group_id=g.group_id)
    return cache


# ---------------------------------------------------------------------------
# Vector providers (lazy: zarr / DINOv2)
# ---------------------------------------------------------------------------


def make_zarr_dinov2_provider(
    resolve_store: Callable[[str], str],
    zarr_key: str = DINOV2_ZARR_KEY,
    pool: str = "64patches",
    l2_normalize: bool = False,
):
    """Provider that reads precomputed DINOv2 patch tokens from zarr and pools.

    ``resolve_store(episode_hash) -> path/URL`` of the episode's zarr store.
    Requires ``zarr`` (and ``fsspec`` for remote stores) at call time.
    """

    def provider(episode_hash: str) -> np.ndarray:
        import zarr  # lazy

        store = zarr.open(resolve_store(episode_hash), mode="r")
        arr = store[zarr_key]  # (T_padded, 256, 768)
        # _write_numeric pads up to a chunk multiple with zero rows; slice to
        # the true length so padding frames never enter the index.
        total = int(store.attrs.get("total_frames", arr.shape[0]))
        arr = np.asarray(arr[:total])  # (T, 256, 768)
        return pool_patch_embeddings(arr, method=pool, l2_normalize=l2_normalize)

    return provider


class DinoV2Embedder:
    """On-the-fly DINOv2 encoder for ``base_0_rgb`` frames (lazy torch).

    Replicates ricl_openpi's reference encoder (``load_dinov2`` + ``embed``) via
    ``egomimic/scripts/embedding_process/dinov2_embedding.py``. Use when
    precomputed zarr embeddings are unavailable. GPU strongly recommended; on an
    M4 this runs on MPS/CPU and is only practical for a handful of frames.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: str | None = None,
        pool: str = "64patches",
        l2_normalize: bool = False,
    ):
        import torch  # lazy

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        self.device = device
        self.pool = pool
        self.l2_normalize = l2_normalize
        self.model = (
            torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()
        )

    def embed(self, images_thwc: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Embed ``(T, H, W, 3)`` uint8 frames -> ``(T, EMBED_DIM)`` pooled."""
        import torch  # lazy

        from egomimic.scripts.embedding_process.dinov2_embedding import (
            preprocess_dinov2,
        )

        images = np.asarray(images_thwc)
        out = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                pixel_values = preprocess_dinov2(images[i : i + batch_size]).to(
                    self.device
                )
                feats = self.model.forward_features(pixel_values)
                patches = feats["x_norm_patchtokens"]  # (B, 256, 768)
                out.append(patches.float().cpu().numpy())
        emb = np.concatenate(out, axis=0)
        return pool_patch_embeddings(
            emb, method=self.pool, l2_normalize=self.l2_normalize
        )


# ---------------------------------------------------------------------------
# CLI + self-test
# ---------------------------------------------------------------------------


def _synthetic_provider(dim: int = EMBED_DIM, frames: int = 24):
    """Deterministic per-hash vectors for the offline smoke (no data/GPU).

    Frames from the same episode cluster together (shared per-episode centroid),
    so retrieval is sane to assert on.
    """

    def provider(episode_hash: str) -> np.ndarray:
        seed = abs(hash(episode_hash)) % (2**32)
        rng = np.random.default_rng(seed)
        centroid = rng.standard_normal(dim).astype(np.float32)
        v = centroid[None, :] + 0.05 * rng.standard_normal((frames, dim)).astype(
            np.float32
        )
        return pool_patch_embeddings(v)  # (T, dim) passthrough, raw L2

    return provider


def _smoke() -> None:
    print("== RICL retrieval smoke (numpy+scipy, no GPU/data) ==")
    # 1) pooling: 64patches must reproduce ricl_openpi utils.embed()'s
    #    64PATCHES loop exactly (16x16 grid -> 8x8 blocks of 2x2, row-major,
    #    concatenated along features; raw, not unit-normalized).
    emb = (
        np.random.default_rng(0).standard_normal((7, 256, TOKEN_DIM)).astype(np.float32)
    )
    pooled = pool_patch_embeddings(emb)
    assert pooled.shape == (7, EMBED_DIM), pooled.shape
    # block (0,0) = patches {(0,0),(0,1),(1,0),(1,1)} = flat [0, 1, 16, 17]
    expect0 = emb[:, [0, 1, 16, 17], :].mean(axis=1)
    assert np.allclose(pooled[:, :TOKEN_DIM], expect0, atol=1e-5)
    # block (0,1) = patches {(0,2),(0,3),(1,2),(1,3)} = flat [2, 3, 18, 19]
    expect1 = emb[:, [2, 3, 18, 19], :].mean(axis=1)
    assert np.allclose(pooled[:, TOKEN_DIM : 2 * TOKEN_DIM], expect1, atol=1e-5)
    # last block (7,7) = flat [238, 239, 254, 255]
    expect63 = emb[:, [238, 239, 254, 255], :].mean(axis=1)
    assert np.allclose(pooled[:, -TOKEN_DIM:], expect63, atol=1e-5)
    assert not np.allclose(np.linalg.norm(pooled, axis=-1), 1.0, atol=1e-2)
    normed = pool_patch_embeddings(emb, l2_normalize=True)
    assert np.allclose(np.linalg.norm(normed, axis=-1), 1.0, atol=1e-5)
    print(
        f"  pool_patch_embeddings[64patches]: (7,256,{TOKEN_DIM}) -> {pooled.shape} "
        "(matches ricl_openpi block order; raw by default)"
    )

    # 2) groups from the REAL pairs json
    pairs = load_pairs()
    for mode in ("cross_similar", "cross_alignment", "within_alignment"):
        groups = groups_from_pairs(pairs, mode)
        nb = sum(len(g.bank_hashes) for g in groups)
        nq = len({q for g in groups for q in g.query_hashes})
        print(
            f"  groups_from_pairs[{mode}]: {len(groups)} groups, "
            f"{nb} bank-eps, {nq} unique query-eps"
        )

    # 2b) leave-one-episode-out: one group per query, never containing itself
    loeo_groups = groups_from_pairs(pairs, "within_alignment", loeo=True)
    base_groups = groups_from_pairs(pairs, "within_alignment")
    assert len(loeo_groups) == sum(len(g.query_hashes) for g in base_groups)
    for g in loeo_groups:
        assert len(g.query_hashes) == 1
        assert g.query_hashes[0] not in g.bank_hashes, g.group_id
        assert g.bank_hashes, f"{g.group_id}: empty bank"
    print(
        f"  groups_from_pairs[within_alignment, loeo]: {len(loeo_groups)} "
        "per-query groups, no self-retrieval"
    )

    # 3) build + query + roundtrip on a synthetic cross_similar cache
    groups = groups_from_pairs(pairs, "cross_similar")
    provider = _synthetic_provider()
    k = 4
    cache = build_cache(groups, provider, k=k, meta={"mode": "cross_similar", "k": k})
    qh = cache.query_hashes[0]
    bh, bf, dd = cache.neighbors(qh, frame_idx=0)
    assert bh.shape == (k,) and bf.shape == (k,) and dd.shape == (k,)
    # neighbors must come from the query's group bank
    grp = next(g for g in groups if qh in g.query_hashes)
    assert set(bh.tolist()) <= set(grp.bank_hashes), (bh.tolist(), grp.bank_hashes)
    assert np.all(np.diff(dd) >= -1e-5), "distances must be sorted ascending"
    print(
        f"  build_cache: {len(cache.query_hashes)} query eps; "
        f"sample q={qh} -> banks={sorted(set(bh.tolist()))} dist[0]={dd[0]:.4f}"
    )

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        cache.save(d)
        re = RetrievalCache.load(d)
        bh2, bf2, dd2 = re.neighbors(qh, 0)
        assert np.array_equal(bh, bh2) and np.array_equal(bf, bf2)
        assert np.allclose(dd, dd2)
    print("  save/load roundtrip: OK")
    print("ALL SMOKE CHECKS PASSED")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--smoke", action="store_true", help="run offline self-test and exit"
    )
    p.add_argument("--pairs", default=PAIRS_JSON_DEFAULT)
    p.add_argument(
        "--mode",
        default="cross_similar",
        choices=["cross_similar", "cross_alignment", "within_alignment"],
    )
    p.add_argument(
        "--merge-bank",
        action="store_true",
        help="use a single global bank (union of all banks) for every query",
    )
    p.add_argument(
        "--loeo",
        action="store_true",
        help="within_alignment only: leave-one-episode-out (a query's own "
        "episode is excluded from its bank)",
    )
    p.add_argument("-k", type=int, default=4)
    p.add_argument("--out", default=None, help="output cache dir")
    p.add_argument(
        "--zarr-root",
        default=None,
        help="dir/URL template with {hash} for per-episode zarr stores; "
        "if unset, requires precomputed embeddings elsewhere",
    )
    args = p.parse_args()

    if args.smoke:
        _smoke()
        return

    pairs = load_pairs(args.pairs)
    groups = groups_from_pairs(pairs, args.mode, loeo=args.loeo)
    if args.merge_bank:
        if args.loeo:
            raise SystemExit("--merge-bank would undo --loeo (banks re-merge)")
        groups = merge_bank(groups)

    if args.zarr_root is None:
        raise SystemExit(
            "Provide --zarr-root (a {hash} template to per-episode zarr stores) to "
            "build a real cache, or use --smoke for the offline self-test."
        )

    provider = make_zarr_dinov2_provider(lambda h: args.zarr_root.format(hash=h))
    cache = build_cache(
        groups,
        provider,
        k=args.k,
        meta={"mode": args.mode, "k": args.k, "loeo": args.loeo},
    )
    out = args.out or f"retrieval_cache_{args.mode}_k{args.k}"
    cache.save(out)
    print(f"wrote cache for {len(cache.query_hashes)} query episodes -> {out}")


if __name__ == "__main__":
    main()
