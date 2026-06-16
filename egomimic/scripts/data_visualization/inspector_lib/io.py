"""CSV / .pt sidecar I/O for the latent inspector."""

from __future__ import annotations

import csv
import glob
import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)


# Reduction-coord tensors the inspector only needs on-demand (one of
# five reductions selected per layer switch). Each entry is (dtype,
# fill_value); fill_value=None means "leave as None if the field is
# missing from the .pt", which lets the scatter view show a clear
# "no PCA exists for this layer" message instead of plotting.
LAZY_FIELD_SPECS: dict[str, tuple] = {
    "umap_xyz": (np.float32, None),
    "pca_umap_xyz": (np.float32, None),
    "tsne2d_xy": (np.float32, None),
    "tsne3d_xyz": (np.float32, None),
    "pca_xyz": (np.float32, None),
}
LAZY_FIELDS = tuple(LAZY_FIELD_SPECS)


class LazyStringArray:
    """Lightweight (vocab, codes) wrapper that behaves like a (N,) numpy
    object array of strings without ever materializing one. Equality
    against a single string is an O(N) int compare instead of an O(N)
    Python str compare. Boolean / int-array indexing returns a new
    LazyStringArray sharing the vocab; scalar indexing returns a str.
    `np.asarray(x)` / `x.tolist()` force a full decode on demand."""

    __slots__ = ("vocab", "codes", "_decoded")

    def __init__(self, vocab, codes):
        if isinstance(vocab, np.ndarray):
            self.vocab = vocab.tolist()
        else:
            self.vocab = list(vocab)
        codes_np = codes.cpu().numpy() if hasattr(codes, "cpu") else np.asarray(codes)
        self.codes = codes_np.astype(np.int32, copy=False)
        self._decoded: np.ndarray | None = None

    def __len__(self):
        return int(self.codes.shape[0])

    @property
    def shape(self):
        return self.codes.shape

    @property
    def dtype(self):
        return np.dtype(object)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self.vocab[int(self.codes[int(key)])]
        new_codes = self.codes[key]
        if np.ndim(new_codes) == 0:
            return self.vocab[int(new_codes)]
        return LazyStringArray(self.vocab, new_codes)

    def __eq__(self, other):  # type: ignore[override]
        if isinstance(other, str):
            # Vocab is typically small (hundreds of unique videos) — list
            # .index is fine and avoids building a dict every call.
            try:
                target = self.vocab.index(other)
            except ValueError:
                return np.zeros(self.codes.shape[0], dtype=bool)
            return self.codes == np.int32(target)
        if isinstance(other, LazyStringArray):
            return np.asarray(self) == np.asarray(other)
        return np.asarray(self) == other

    def __ne__(self, other):  # type: ignore[override]
        return ~self.__eq__(other)

    def __hash__(self):  # not hashable, but matches ndarray semantics
        raise TypeError("LazyStringArray is not hashable")

    def __iter__(self):
        vocab = self.vocab
        for c in self.codes:
            yield vocab[int(c)]

    def tolist(self):
        vocab = self.vocab
        return [vocab[int(c)] for c in self.codes.tolist()]

    def __array__(self, dtype=None):
        if self._decoded is None:
            self._decoded = np.asarray(self.vocab, dtype=object)[self.codes]
        if dtype is not None and np.dtype(dtype) != np.dtype(object):
            return self._decoded.astype(dtype)
        return self._decoded

    def __repr__(self):
        return f"LazyStringArray(N={len(self)}, vocab_size={len(self.vocab)})"


class LazyLayerData(dict):
    """Dict subclass where the reduction fields (`umap_xyz`, …) are loaded
    on first access instead of at file-open. `hashes` and `embs` are held
    as LazyStringArray (decode-on-demand). `frame_idx` / `token_idx` are
    eager but cheap. Keeps a ref to the torch payload so the mmap stays
    alive.

    Logs per-field materialization time so it's easy to see which step
    actually costs you on layer-switch / scatter-render.
    """

    def __init__(self, eager: dict, lazy: dict, *, path: str, payload, n: int):
        super().__init__(eager)
        self._n = n
        # Initialize every lazy slot to None so callers can do
        # `data.get("umap_xyz") is None` without triggering a load.
        for key in lazy:
            if key not in self:
                super().__setitem__(key, None)
        self._lazy = lazy  # field name → (torch_tensor_or_none, dtype, fill)
        self._path = path
        self._payload = payload  # keep mmap alive

    def __getitem__(self, key):
        if key in self._lazy:
            tensor, dtype, fill = self._lazy.pop(key)
            t0 = time.perf_counter()
            if tensor is None:
                arr = np.full(self._n, fill, dtype=dtype) if fill is not None else None
            else:
                if hasattr(tensor, "detach"):
                    tensor = tensor.detach().cpu().numpy()
                arr = np.asarray(tensor, dtype=dtype)
            super().__setitem__(key, arr)
            shape = arr.shape if arr is not None else "None"
            logger.info(
                "Materialized %s.%s: %.2fs (shape=%s)",
                os.path.basename(self._path),
                key,
                time.perf_counter() - t0,
                shape,
            )
            return arr
        return super().__getitem__(key)

    def get(self, key, default=None):
        try:
            v = self[key]
        except KeyError:
            return default
        return v if v is not None else default


def list_layer_csvs(latent_dir: str) -> list[str]:
    """Returns one path per layer. Prefers `<layer>.csv` when present; falls
    back to `<layer>_keys.pt` when only the .pt sidecar exists (eval ran
    with `skip_csv=true`). The path returned is whatever exists — readers
    use the suffix to dispatch."""
    csvs = sorted(glob.glob(os.path.join(latent_dir, "*.csv")))
    csvs = [p for p in csvs if os.path.basename(p) != "comparison.csv"]
    csv_layer_names = {os.path.splitext(os.path.basename(p))[0] for p in csvs}
    pts = sorted(glob.glob(os.path.join(latent_dir, "*_keys.pt")))
    pts_only = []
    for p in pts:
        layer = os.path.basename(p)[: -len("_keys.pt")]
        if layer not in csv_layer_names:
            pts_only.append(p)
    return csvs + pts_only


def discover_runs(root_dir: str) -> list[tuple[str, str]]:
    """Walk `root_dir` for any directory that contains per-layer *.csv files
    OR *_keys.pt files (excluding comparison.csv). Returns a list of
    (display_name, abs_path) sorted by display_name. The display name is
    the relative path from root, e.g. 'cotrain_partial_aug/2026-04-24_03-08-44/latent_random_2026-05-02_00-09-18/latents/epoch_0'.
    """
    if not os.path.isdir(root_dir):
        return []
    out: list[tuple[str, str]] = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        has_csv = any(f.endswith(".csv") and f != "comparison.csv" for f in filenames)
        has_pt = any(f.endswith("_keys.pt") for f in filenames)
        if has_csv or has_pt:
            rel = os.path.relpath(dirpath, root_dir)
            # Trim the trailing "/latents/epoch_N" so the dropdown shows the
            # human-meaningful run identifier.
            display = rel
            for trim in ("/latents/epoch_0", "/latents/epoch_1", "/latents/epoch_2"):
                if display.endswith(trim):
                    display = display[: -len(trim)]
                    break
            out.append((display, os.path.abspath(dirpath)))
    out.sort(key=lambda x: x[0])
    return out


def read_pt_payload(path: str, load_keys: bool = False) -> dict:
    """Read the metadata fields of a v3 `<layer>_keys.pt` eagerly and
    defer the reduction-coord tensors (umap_xyz, pca_umap_xyz, etc.) so
    they only load on first access. Returns a dict-like (`LazyLayerData`)
    that drop-in replaces the old eager dict.

    Detailed timing per phase is logged so it's clear which step is slow
    (torch.load / string-decode / frame_idx / token_idx) — the reduction
    fields are timed in `LazyLayerData.__getitem__` when accessed.
    """
    import torch

    # mmap=True so the giant `keys` tensor isn't read from disk unless we
    # actually touch it. Without mmap, even load_keys=False pays the full
    # ~19 GB I/O for paligemma img layers because torch eagerly
    # deserializes every storage in the archive.
    t0 = time.perf_counter()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu", mmap=True)
    t_load = time.perf_counter() - t0

    # Legacy v1 .pt files are a bare Tensor (no metadata). Synthesize a
    # minimal dict-shaped payload so the rest of this function can
    # uniformly call `payload.get(...)` without type-checking.
    if not isinstance(payload, dict):
        payload = {"keys": payload}

    def _lazy_str_array(prefix: str):
        """Return a LazyStringArray for a metadata field, or a legacy
        numpy object-array if the .pt predates the vocab/codes format."""
        vocab = payload.get(f"{prefix}_vocab")
        codes = payload.get(f"{prefix}_codes")
        if vocab is not None and codes is not None:
            return LazyStringArray(vocab, codes)
        legacy = payload.get(prefix)
        if legacy is None:
            return None
        # Legacy list[str] path: build a vocab on the fly so downstream
        # equality / sampling still hits the fast int-codes route.
        legacy_arr = np.asarray(legacy, dtype=object)
        uniq, inv = np.unique(legacy_arr, return_inverse=True)
        return LazyStringArray(uniq.tolist(), inv)

    # Derive N from whichever metadata field is present without paying
    # any I/O on the giant `keys` tensor (its shape is in the pickle
    # header that torch.load already parsed).
    def _len_of(field_name):
        v = payload.get(field_name)
        if v is None:
            return None
        if hasattr(v, "shape"):
            return int(v.shape[0])
        try:
            return len(v)
        except TypeError:
            return None

    n = (
        _len_of("hashes_codes")
        or _len_of("frame_idx")
        or _len_of("token_idx")
        or _len_of("keys")
        or 0
    )

    def _to_np(v, dtype):
        if v is None:
            return None
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()
        return np.asarray(v, dtype=dtype)

    t1 = time.perf_counter()
    hashes_lazy = _lazy_str_array("hashes")
    if hashes_lazy is None:
        hashes_lazy = LazyStringArray(["?"], np.zeros(n, dtype=np.int32))
    t_hashes = time.perf_counter() - t1

    t1 = time.perf_counter()
    embs_lazy = _lazy_str_array("embs")
    if embs_lazy is None:
        embs_lazy = LazyStringArray([""], np.zeros(n, dtype=np.int32))
    t_embs = time.perf_counter() - t1

    # frame_idx / token_idx stay eager: every scatter render and every
    # click handler needs them, materialization is just a single mmap
    # fault (~0.1-0.3s at 10M rows), and downstream `sub[fld][mask]`
    # paths crash if these are still None placeholders.
    t1 = time.perf_counter()
    frame_idx = _to_np(payload.get("frame_idx"), np.int64)
    if frame_idx is None:
        frame_idx = np.full(n, -1, dtype=np.int64)
    t_frame = time.perf_counter() - t1

    t1 = time.perf_counter()
    token_idx = _to_np(payload.get("token_idx"), np.int64)
    if token_idx is None:
        token_idx = np.full(n, -1, dtype=np.int64)
    t_token = time.perf_counter() - t1

    t1 = time.perf_counter()
    keys = None
    if load_keys and "keys" in payload:
        keys_t = payload["keys"]
        if torch.is_tensor(keys_t):
            keys = keys_t.to(torch.float32).cpu().numpy()
        else:
            keys = np.asarray(keys_t, dtype=np.float32)
    t_keys = time.perf_counter() - t1

    # `hashes` / `embs` go in the eager dict but as LazyStringArray —
    # construction is near-free (no per-row alloc) and equality uses the
    # int codes, so we save the full object-array build for callers that
    # actually need it (np.asarray / .tolist materialize on demand).
    eager = {
        "hashes": hashes_lazy,
        "embs": embs_lazy,
        "frame_idx": frame_idx,
        "token_idx": token_idx,
        "keys": keys,
    }
    # Lazy: reduction coords only — exactly one of these is needed per
    # layer-switch, so the other four are deferred (logged per-field on
    # first access from `LazyLayerData.__getitem__`).
    lazy = {
        key: (payload.get(key), dtype, fill)
        for key, (dtype, fill) in LAZY_FIELD_SPECS.items()
    }
    logger.info(
        "Read %s eager: load=%.2fs hashes=%.2fs embs=%.2fs frame=%.2fs "
        "token=%.2fs keys=%.2fs N=%d (reductions deferred)",
        os.path.basename(path),
        t_load,
        t_hashes,
        t_embs,
        t_frame,
        t_token,
        t_keys,
        n,
    )
    return LazyLayerData(eager, lazy, path=path, payload=payload, n=n)


def read_csv(path: str, load_keys: bool = False):
    """Returns dict of arrays read straight from the CSV. ALL reduction
    coords (umap, tsne2d, tsne3d, pca) come from columns written by
    eval_latent — no client-side recompute. Missing columns become None.

    By default, the huge `k0..kN` raw key columns are SKIPPED (load_keys=False)
    because the inspector never reads them — saves ~98% of memory for
    image-token slices (1.5M rows × 256 floats = ~3GB → ~50MB).

    When `path` ends in `_keys.pt` (eval ran with skip_csv=true), reads
    the same fields from the torch payload instead of parsing CSV. Much
    faster (binary load vs. Python row loop).
    """
    if path.endswith("_keys.pt"):
        return read_pt_payload(path, load_keys=load_keys)
    rows_hash, rows_emb, rows_frame, rows_token = [], [], [], []
    rows_umap, rows_pca_umap, rows_tsne2d, rows_tsne3d, rows_pca, rows_keys = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_umap = {"umap_x", "umap_y", "umap_z"}.issubset(fieldnames)
        has_pca_umap = {"pca_umap_x", "pca_umap_y", "pca_umap_z"}.issubset(fieldnames)
        has_tsne2d = {"tsne2d_x", "tsne2d_y"}.issubset(fieldnames)
        has_tsne3d = {"tsne3d_x", "tsne3d_y", "tsne3d_z"}.issubset(fieldnames)
        has_pca = {"pca_x", "pca_y", "pca_z"}.issubset(fieldnames)
        has_frame = "frame_idx" in fieldnames
        has_token = "token_idx" in fieldnames
        k_cols = []
        if load_keys:
            k_cols = sorted(
                [c for c in fieldnames if c.startswith("k") and c[1:].isdigit()],
                key=lambda c: int(c[1:]),
            )
        for r in reader:
            rows_hash.append(r["video_hash"])
            rows_emb.append(r.get("embodiment", ""))
            rows_frame.append(int(r["frame_idx"]) if has_frame else -1)
            rows_token.append(int(r["token_idx"]) if has_token else -1)
            if has_umap:
                rows_umap.append(
                    (float(r["umap_x"]), float(r["umap_y"]), float(r["umap_z"]))
                )
            if has_pca_umap:
                rows_pca_umap.append(
                    (
                        float(r["pca_umap_x"]),
                        float(r["pca_umap_y"]),
                        float(r["pca_umap_z"]),
                    )
                )
            if has_tsne2d:
                rows_tsne2d.append((float(r["tsne2d_x"]), float(r["tsne2d_y"])))
            if has_tsne3d:
                rows_tsne3d.append(
                    (float(r["tsne3d_x"]), float(r["tsne3d_y"]), float(r["tsne3d_z"]))
                )
            if has_pca:
                rows_pca.append(
                    (float(r["pca_x"]), float(r["pca_y"]), float(r["pca_z"]))
                )
            if load_keys:
                rows_keys.append([float(r[c]) for c in k_cols])

    return {
        "hashes": np.asarray(rows_hash),
        "embs": np.asarray(rows_emb),
        "frame_idx": np.asarray(rows_frame, dtype=np.int64),
        "token_idx": np.asarray(rows_token, dtype=np.int64),
        "umap_xyz": np.asarray(rows_umap, dtype=np.float32) if has_umap else None,
        "pca_umap_xyz": np.asarray(rows_pca_umap, dtype=np.float32)
        if has_pca_umap
        else None,
        "tsne2d_xy": np.asarray(rows_tsne2d, dtype=np.float32) if has_tsne2d else None,
        "tsne3d_xyz": np.asarray(rows_tsne3d, dtype=np.float32) if has_tsne3d else None,
        "pca_xyz": np.asarray(rows_pca, dtype=np.float32) if has_pca else None,
        "keys": np.asarray(rows_keys, dtype=np.float32) if load_keys else None,
    }


def stratified_sample(groups, k: int, seed: int = 0) -> np.ndarray:
    """Sample `k` row indices stratified by group. Accepts:
      - `LazyStringArray` → uses its int codes directly (fastest)
      - integer ndarray  → uses values as codes
      - object/str ndarray → falls back to `np.unique(return_inverse=True)`
        for a one-pass O(N) dedupe (still avoids per-group `np.where`).

    Implementation is O(N log N) regardless of group count: one argsort
    on the codes, then contiguous block slicing — vs. the previous
    O(N * #groups) of `np.where(groups == g)` running per unique group
    on a Python-string object array."""
    n = len(groups)
    if k >= n:
        return np.arange(n)

    if hasattr(groups, "codes"):
        codes = np.asarray(groups.codes)
    elif isinstance(groups, np.ndarray) and groups.dtype.kind in ("i", "u"):
        codes = groups
    else:
        _, codes = np.unique(np.asarray(groups), return_inverse=True)

    if codes.size == 0:
        return np.empty(0, dtype=np.int64)

    # Sort once, then find run boundaries to get contiguous per-group
    # index blocks without scanning N per group.
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    diffs = np.diff(sorted_codes)
    breaks = np.concatenate([[0], np.flatnonzero(diffs != 0) + 1, [codes.size]]).astype(
        np.int64
    )
    n_groups = breaks.size - 1
    per_group = max(1, k // n_groups)

    rng = np.random.default_rng(seed)
    keep = []
    for g in range(n_groups):
        block = order[breaks[g] : breaks[g + 1]]
        if block.size <= per_group:
            keep.append(block)
        else:
            keep.append(rng.choice(block, per_group, replace=False))
    return np.concatenate(keep) if keep else np.empty(0, dtype=np.int64)
