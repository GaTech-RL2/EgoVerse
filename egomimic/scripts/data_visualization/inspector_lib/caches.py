"""LRU cache layer (LayerStore) for per-(run, layer) reduction coords,
raw key tensors, PCA features, and precomputed cross-embodiment KNN.

Originally a quartet of OrderedDict closures nested inside `build_app`."""

from __future__ import annotations

import logging
import os
from collections import OrderedDict

import numpy as np

from .io import (
    list_layer_csvs,
    read_csv,
)

logger = logging.getLogger(__name__)


class LayerStore:
    """Bounded LRU caches keyed by (run_path, layer).

    - `full_cache` holds the reduction-coord dict (umap/tsne/pca, hashes,
      embs, frame/token idx). Each entry is roughly 50MB without keys.
    - `keys_cache` holds raw `<layer>_keys.pt` torch tensors (1-3GB each
      for image-token slices) — capped tighter.
    - `pca_cache` holds fitted PCA features per (run, layer, n_components).
    - `knn_cache` holds precomputed cross-embodiment KNN (lazy mmap-backed).
    """

    def __init__(
        self,
        full_max: int = 4,
        keys_max: int = 2,
        pca_max: int = 4,
        knn_max: int = 8,
    ):
        self.full_max = full_max
        self.keys_max = keys_max
        self.pca_max = pca_max
        self.knn_max = knn_max
        self.full_cache: "OrderedDict[tuple[str, str], dict]" = OrderedDict()
        self.keys_cache: "OrderedDict[tuple[str, str], np.ndarray]" = OrderedDict()
        self.pca_cache: "OrderedDict[tuple[str, str, int], np.ndarray]" = OrderedDict()
        self.knn_cache: "OrderedDict[tuple[str, str], dict | None]" = OrderedDict()

    def layers_for(self, run_path: str) -> list[str]:
        paths = list_layer_csvs(run_path)
        layers = []
        for p in paths:
            base = os.path.basename(p)
            if base.endswith("_keys.pt"):
                layers.append(base[: -len("_keys.pt")])
            else:
                layers.append(os.path.splitext(base)[0])
        return layers

    def load(self, run_path: str, layer: str) -> dict:
        import time

        key = (run_path, layer)
        if key in self.full_cache:
            self.full_cache.move_to_end(key)
            logger.info(
                "LayerStore.load HIT %s | %s (cached)",
                os.path.basename(run_path),
                layer,
            )
            return self.full_cache[key]
        logger.info("Loading %s | %s ...", os.path.basename(run_path), layer)
        t0 = time.perf_counter()
        csv_path = os.path.join(run_path, f"{layer}.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(run_path, f"{layer}_keys.pt")
        # load_keys=False — inspector never reads k0..kN, saves ~98% of RAM.
        self.full_cache[key] = read_csv(csv_path, load_keys=False)
        logger.info(
            "LayerStore.load MISS %s | %s in %.2fs",
            os.path.basename(run_path),
            layer,
            time.perf_counter() - t0,
        )
        while len(self.full_cache) > self.full_max:
            evicted_key, _ = self.full_cache.popitem(last=False)
            logger.info(
                "Cache evict: %s | %s", os.path.basename(evicted_key[0]), evicted_key[1]
            )
        return self.full_cache[key]

    def _read_keys_pt_payload(self, pt_path: str):
        """Load `<layer>_keys.pt` and return (keys_arr_or_none,
        knn_entry_or_none). Handles both v2 (dict bundling keys + optional
        knn fields) and legacy (bare tensor) on-disk formats. Returns
        (None, None) if the file is unreadable."""
        if not os.path.isfile(pt_path):
            return None, None
        try:
            import torch

            try:
                obj = torch.load(pt_path, map_location="cpu", weights_only=False)
            except Exception:
                obj = torch.load(pt_path, map_location="cpu")
        except Exception as e:
            logger.warning("failed to load %s: %s", pt_path, e)
            return None, None
        # v2: dict with 'keys' (+ optional 'knn_indices'/'knn_distances'/...).
        # legacy: bare tensor.
        if isinstance(obj, dict):
            tensor = obj.get("keys")
            keys_arr = (
                tensor.to(torch.float32).cpu().numpy() if tensor is not None else None
            )
            knn_entry = None
            if "knn_indices" in obj and "knn_distances" in obj:
                idx = obj["knn_indices"]
                dist = obj["knn_distances"]
                if hasattr(idx, "cpu"):
                    idx = idx.cpu().numpy()
                if hasattr(dist, "cpu"):
                    dist = dist.cpu().numpy()
                knn_entry = {
                    "indices": np.asarray(idx, dtype=np.int32),
                    "distances": np.asarray(dist, dtype=np.float32),
                    "k": int(obj.get("knn_k", idx.shape[1])),
                    "embs": list(obj.get("embs", [])),
                    "space": str(obj.get("knn_space", "raw")),
                }
        else:
            keys_arr = obj.to(torch.float32).cpu().numpy()
            knn_entry = None
        return keys_arr, knn_entry

    def load_keys(self, run_path: str, layer: str):
        """Load raw `<layer>_keys.pt` keys as a (N, D) numpy array, cached
        with its own LRU (max 2 layers — these are 1-3GB each). Also warms
        the KNN cache opportunistically when the file bundles a knn
        sidecar (v2 format), so a later `load_knn` is free."""
        key = (run_path, layer)
        if key in self.keys_cache:
            self.keys_cache.move_to_end(key)
            return self.keys_cache[key]
        pt_path = os.path.join(run_path, f"{layer}_keys.pt")
        keys_arr, knn_entry = self._read_keys_pt_payload(pt_path)
        if keys_arr is None:
            return None
        logger.info(
            "Loaded keys.pt for %s | %s (shape=%s%s)",
            os.path.basename(run_path),
            layer,
            keys_arr.shape,
            f", +knn K={knn_entry['k']}" if knn_entry is not None else "",
        )
        self.keys_cache[key] = keys_arr
        while len(self.keys_cache) > self.keys_max:
            evicted_key, _ = self.keys_cache.popitem(last=False)
            logger.info(
                "Keys cache evict: %s | %s",
                os.path.basename(evicted_key[0]),
                evicted_key[1],
            )
        if knn_entry is not None and key not in self.knn_cache:
            self.knn_cache[key] = knn_entry
            while len(self.knn_cache) > self.knn_max:
                evicted_key, _ = self.knn_cache.popitem(last=False)
                logger.info(
                    "KNN cache evict: %s | %s",
                    os.path.basename(evicted_key[0]),
                    evicted_key[1],
                )
        return self.keys_cache[key]

    def load_knn(self, run_path: str, layer: str):
        """Load precomputed cross-embodiment KNN bundled inside
        `<layer>_keys.pt` (v2 format, written by eval_latent). Returns
        dict with 'indices' (N, K) int32, 'distances' (N, K) float32,
        'k' int, 'embs' list[str], or None if the file is missing or
        doesn't carry KNN fields. Cached LRU; falls through to a quick
        peek at the .pt file when not yet warm."""
        import time

        t0 = time.perf_counter()
        key = (run_path, layer)
        logger.debug(
            "LayerStore.load_knn ENTER %s | %s (cache_size=%d/%d)",
            os.path.basename(run_path),
            layer,
            len(self.knn_cache),
            self.knn_max,
        )
        if key in self.knn_cache:
            self.knn_cache.move_to_end(key)
            cached = self.knn_cache[key]
            logger.info(
                "LayerStore.load_knn HIT %s | %s (cached=%s)",
                os.path.basename(run_path),
                layer,
                "None"
                if cached is None
                else f"N={cached['indices'].shape[0]} K={cached['indices'].shape[1]} space={cached['space']}",
            )
            return cached
        pt_path = os.path.join(run_path, f"{layer}_keys.pt")
        logger.debug("LayerStore.load_knn probing path=%s", pt_path)
        if not os.path.isfile(pt_path):
            logger.info(
                "LayerStore.load_knn MISS %s | %s (.pt file not found at %s)",
                os.path.basename(run_path),
                layer,
                pt_path,
            )
            self.knn_cache[key] = None
            return None
        try:
            file_size = os.path.getsize(pt_path)
            logger.debug(
                "LayerStore.load_knn file exists size=%.1fMB",
                file_size / (1024 * 1024),
            )
        except OSError:
            pass
        # Lazy load: mmap the file and keep the knn_* tensors as torch
        # views over the mmap'd storage. We deliberately do NOT materialize
        # them with .numpy() / .clone() here — that would page the full
        # (N, K) int32 + (N, K) float32 (≈200 MB at N=2.5M) into RAM up
        # front. With the views, only the rows that callers actually slice
        # ([src_idx]) get faulted in from disk, so first-click is fast and
        # memory grows only with how many points the user inspects.
        knn_entry = None
        try:
            import torch

            try:
                obj = torch.load(
                    pt_path, map_location="cpu", weights_only=False, mmap=True
                )
                logger.debug(
                    "LayerStore.load_knn torch.load OK (weights_only=False, mmap)"
                )
            except Exception as e1:
                logger.debug(
                    "LayerStore.load_knn torch.load(weights_only=False) failed: %s — retrying without flag",
                    e1,
                )
                obj = torch.load(pt_path, map_location="cpu", mmap=True)
                logger.debug("LayerStore.load_knn torch.load OK (fallback, mmap)")
            if isinstance(obj, dict):
                logger.debug(
                    "LayerStore.load_knn payload is dict, keys=%s",
                    sorted(obj.keys()),
                )
            else:
                logger.debug(
                    "LayerStore.load_knn payload is %s (legacy bare-tensor — no KNN possible)",
                    type(obj).__name__,
                )
            if (
                isinstance(obj, dict)
                and "knn_indices" in obj
                and "knn_distances" in obj
            ):
                idx_t = obj["knn_indices"]
                dist_t = obj["knn_distances"]
                logger.debug(
                    "LayerStore.load_knn found knn_indices shape=%s dtype=%s | "
                    "knn_distances shape=%s dtype=%s",
                    tuple(idx_t.shape),
                    idx_t.dtype,
                    tuple(dist_t.shape),
                    dist_t.dtype,
                )
                knn_entry = {
                    "indices": idx_t,  # torch tensor, mmap-backed
                    "distances": dist_t,  # torch tensor, mmap-backed
                    "k": int(obj.get("knn_k", idx_t.shape[1])),
                    "embs": list(obj.get("embs", [])),
                    "space": str(obj.get("knn_space", "raw")),
                }
            elif isinstance(obj, dict):
                missing = [f for f in ("knn_indices", "knn_distances") if f not in obj]
                logger.debug(
                    "LayerStore.load_knn dict missing KNN field(s): %s",
                    missing,
                )
        except Exception as e:
            logger.warning(
                "LayerStore.load_knn failed to peek knn from %s: %s (%s)",
                pt_path,
                e,
                type(e).__name__,
            )
            knn_entry = None
        if knn_entry is not None:
            logger.info(
                "LayerStore.load_knn MISS %s | %s in %.2fs (N=%d K=%d space=%s)",
                os.path.basename(run_path),
                layer,
                time.perf_counter() - t0,
                knn_entry["indices"].shape[0],
                knn_entry["indices"].shape[1],
                knn_entry["space"],
            )
        else:
            logger.info(
                "LayerStore.load_knn MISS %s | %s in %.2fs (no KNN in file)",
                os.path.basename(run_path),
                layer,
                time.perf_counter() - t0,
            )
        self.knn_cache[key] = knn_entry
        while len(self.knn_cache) > self.knn_max:
            evicted_key, _ = self.knn_cache.popitem(last=False)
            logger.info(
                "KNN cache evict: %s | %s",
                os.path.basename(evicted_key[0]),
                evicted_key[1],
            )
        return self.knn_cache[key]

    def pca_features(self, run_path: str, layer: str, n_components: int = 50):
        """Fit PCA(n_components) on the layer's raw keys and return the
        transformed (N, n_components) array. Cached. Returns None if raw
        keys aren't available."""
        cache_key = (run_path, layer, n_components)
        if cache_key in self.pca_cache:
            self.pca_cache.move_to_end(cache_key)
            return self.pca_cache[cache_key]
        raw = self.load_keys(run_path, layer)
        if raw is None:
            return None
        n, d = raw.shape
        k = min(n_components, n, d)
        try:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=k)
            feats = pca.fit_transform(raw)
        except Exception as e:
            logger.warning(
                "PCA failed for %s | %s: %s", os.path.basename(run_path), layer, e
            )
            return None
        feats = feats.astype(np.float32)
        logger.info(
            "PCA features for %s | %s shape=%s (var explained=%.3f)",
            os.path.basename(run_path),
            layer,
            feats.shape,
            float(getattr(pca, "explained_variance_ratio_", np.zeros(1)).sum()),
        )
        self.pca_cache[cache_key] = feats
        while len(self.pca_cache) > self.pca_max:
            evicted_key, _ = self.pca_cache.popitem(last=False)
            logger.info(
                "PCA cache evict: %s | %s | n=%s",
                os.path.basename(evicted_key[0]),
                evicted_key[1],
                evicted_key[2],
            )
        return self.pca_cache[cache_key]
