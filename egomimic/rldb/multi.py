"""The training-loop dataset: a dict of episode leaves (zarr or Lance) with the
index map, bounds check, normalization stats and batched-fetch dispatch on top."""

from __future__ import annotations

import concurrent.futures as cf
import copy
import functools
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.filters import DatasetFilter, PinError
from egomimic.utils.action_utils import (
    _apply_norm_one,
    _apply_unnorm_one,
    _degenerate,
    _norm_center_scale,
)
from egomimic.utils.gpu_orphans import die_with_parent
from egomimic.utils.pose_utils import (
    bimanual_cartesian_layout,
    bimanual_keypoint_layout,
    rot6d_channels,
)

logger = logging.getLogger(__name__)


@functools.cache
def _fetch_pool() -> cf.ThreadPoolExecutor:
    """Per-process pool for the concurrent parts of a batched leaf fetch."""
    return cf.ThreadPoolExecutor(max_workers=8, thread_name_prefix="batch-fetch")


SEED = 42


def split_dataset_names(dataset_names, valid_ratio=0.2, seed=SEED):
    """
    Split a list of dataset names into train/valid sets.
    Args:
        dataset_names (Iterable[str])
        valid_ratio (float): fraction of datasets to put in valid.
        seed (int): for deterministic shuffling.


    Returns:
        train_set (set[str]), valid_set (set[str])
    """
    names = sorted(dataset_names)
    if not names:
        return set(), set()

    rng = random.Random(seed)
    rng.shuffle(names)

    if not (0.0 <= valid_ratio <= 1.0):
        raise ValueError(f"valid_ratio must be in [0,1], got {valid_ratio}")

    n_valid = int(len(names) * valid_ratio)
    if valid_ratio > 0.0:
        n_valid = max(1, n_valid)

    valid = set(names[:n_valid])
    train = set(names[n_valid:])
    return train, valid


def _infer_key_type(key_name: str) -> str | None:
    """
    Heuristically infer a key_type for a post-transform key that wasn't
    declared in the leaf's key_map (typically produced by a transform like
    ``ConcatKeys``). Returns one of the recognised key_type strings or None
    when the key looks like metadata that NormStats shouldn't normalize.

    Recognition is conservative — only positive matches return a type.
    """
    name = key_name.lower()
    # Camera image keys
    if "image" in name or "/img" in name or name.endswith("_img") or ".img" in name:
        return "camera_keys"
    # Action keys (model outputs)
    if name.startswith("actions") or "/cmd_" in name or ".cmd_" in name:
        return "action_keys"
    # Proprioception (observed state)
    if (
        name.startswith("observations.state")
        or "/obs_" in name
        or ".obs_" in name
        or "joint_positions" in name
        or "ee_pose" in name
        or "keypoints" in name
    ):
        return "proprio_keys"
    # Language / annotations
    if "annotation" in name or "tokenized" in name or "lang" in name:
        return "annotation_keys"
    return None


def _bounds_check_channels(zarr_key: str, width: int) -> list[int] | None:
    """Channels of a known bimanual layout that quantile bounds apply to
    (translation, gripper, keypoints), or ``None`` to check the full vector.
    Rotation channels (Euler or rot6d columns) are excluded."""
    if zarr_key in ("actions_cartesian", "observations.state.ee_pose"):
        layout = bimanual_cartesian_layout(width)
        if layout is not None:
            return list(layout["xyz"]) + list(layout["grip"])
    elif zarr_key in ("actions_keypoints", "observations.state.keypoints"):
        layout = bimanual_keypoint_layout(width)
        if layout is not None:
            return sorted(list(layout["wrist_xyz"]) + list(layout["keypoints"]))
    return None


def worker_context(leaves) -> str | None:
    """DataLoader start method for ``leaves``: forkserver if any is not fork-safe
    (Lance runs Rust threads; https://lance.org/integrations/pytorch/), else the
    default fork."""
    for leaf in leaves:
        for sub in (
            MultiDataset._iter_leaves(leaf)
            if isinstance(leaf, MultiDataset)
            else [leaf]
        ):
            if not getattr(sub, "fork_safe", True):
                return "forkserver"
    return None


class _IndexMap:
    """Global index -> (dataset name, local index), as offsets rather than one tuple per sample.

    A per-sample list costs ~100 B a sample in every process: tens of GB at 10^7 samples,
    copied page by page into forked workers and pickled whole into spawned ones.
    """

    def __init__(self, datasets: dict):
        self.names = list(datasets)
        self.offsets = np.cumsum(
            [0] + [len(d) for d in datasets.values()], dtype=np.int64
        )

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, idx: int) -> tuple[str, int]:
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        j = int(np.searchsorted(self.offsets, idx, side="right")) - 1
        return self.names[j], int(idx - self.offsets[j])

    def __iter__(self):
        for j, name in enumerate(self.names):
            for local in range(int(self.offsets[j + 1] - self.offsets[j])):
                yield name, local

    def ranges(self) -> dict[str, range]:
        return {
            n: range(int(self.offsets[j]), int(self.offsets[j + 1]))
            for j, n in enumerate(self.names)
        }


class MultiDataset(torch.utils.data.Dataset):
    """
    Wraps a dict of child datasets (Zarr leaves or other MultiDatasets) and
    also owns the normalization-stats descriptor and helpers that used to live
    in separate ``NormStats`` / ``NormalizingMultiDataset`` classes.

    Two construction modes:
      - **Data mode** (default): pass ``datasets`` to wrap a real dataset graph
        (existing behaviour, used during training). Stats fields start empty;
        call ``populate_from_datasets()`` and ``infer_norm_from_dataset(...)``
        to fill them in, then ``attach_normalize_transforms()`` to wire
        normalize/reject transforms onto each leaf's ``transform_list``.
      - **State mode** (``state=...``, ``datasets=None``): construct a
        stats-only instance for deploy/eval where the dataset graph isn't
        available. ``self.datasets`` is empty; only the stats fields are
        populated. Used by checkpoint reconstruction.
    """

    NORMALIZE_KEY_TYPES = ("proprio_keys", "action_keys")
    # Saved in the state: a model trained before rot6d passed through
    # normalization must keep normalizing it (see _load_state).
    rot6d_passthrough = True

    # Bounds-check slack per (timestep, channel) cell; see _check_bounds.
    BOUNDS_ABS_TOL = 1e-6
    BOUNDS_REL_SLACK = 0.5

    def __init__(
        self,
        datasets: (dict[str, MultiDataset | torch.utils.data.Dataset] | None) = None,
        mode: str = "train",
        percent: float = 0.1,
        valid_ratio: float = 0.2,
        norm_mode: str = "zscore",
        state: dict | None = None,
        **kwargs,
    ):
        """
        Args:
            datasets: Dict of child datasets. None for state-only construction.
            mode: One of "train", "valid", "total", "percent" — which split to keep.
            percent: Fraction (when mode="percent").
            valid_ratio: Train/valid split ratio.
            norm_mode: One of "zscore", "minmax", "quantile".
            state: If provided, populate stats fields from this dict (deploy mode).
        """
        super().__init__()

        # ---- Stats fields (always present, may be empty) ----
        self.norm_mode = norm_mode
        self.embodiments: set[int] = set()
        self.key_types: dict[int, dict[str, str]] = {}
        self.zarr_keys: dict[int, dict[str, str]] = {}
        self.shapes: dict[int, dict[str, tuple]] = {}
        self.norm_stats: dict[int, dict[str, dict[str, np.ndarray]]] = {}
        self._norm_run_metadata: dict[str, float | int | None] | None = None

        # ---- Dataset graph fields ----
        self.datasets: dict = {}
        self.index_map: list = []
        self._global_indices_by_dataset: dict[str, list[int]] = {}
        # Dedup bounds-check warnings: keyed by f"bounds:{episode}:{zarr_key}"
        self._warned_violations: set[str] = set()
        self.train_collections: set = set()
        self.valid_collections: set = set()

        if state is not None:
            # Deploy / state-only construction — no dataset graph.
            self._load_state(state)
            return

        if datasets is None:
            raise ValueError("MultiDataset requires either `datasets` or `state`.")

        # ---- Normal data-mode construction ----
        self.train_collections, self.valid_collections = split_dataset_names(
            datasets.keys(), valid_ratio=valid_ratio, seed=SEED
        )

        if mode == "train":
            chosen = self.train_collections
        elif mode == "valid":
            chosen = self.valid_collections
        elif mode == "total":
            chosen = set(datasets.keys())
        elif mode == "percent":
            all_names = sorted(datasets.keys())
            rng = random.Random(SEED)
            rng.shuffle(all_names)
            n_keep = int(len(all_names) * percent)
            if percent > 0.0:
                n_keep = max(1, n_keep)
            chosen = set(all_names[:n_keep])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.datasets = {rid: ds for rid, ds in datasets.items() if rid in chosen}
        assert self.datasets, "No datasets left after applying mode split."

        self.index_map = _IndexMap(self.datasets)
        self._global_indices_by_dataset = self.index_map.ranges()

    def __len__(self) -> int:
        return len(self.index_map)

    @staticmethod
    def _episode_name_for_dataset(dataset, dataset_name: str) -> str:
        episode_path = getattr(dataset, "episode_path", None)
        if episode_path is None:
            return dataset_name
        return Path(episode_path).name

    def set_norm_stats_from(self, source: "MultiDataset") -> None:
        """Share stats with this dataset (and its nested MultiDatasets) by
        reference. After this call ``__getitem__`` will bounds-check + normalize
        each sample using ``source``'s ``norm_stats``/``key_types``/``zarr_keys``.

        Use this *instead of* ``attach_normalize_transforms`` — it doesn't mutate
        any leaf-level ``transform`` list, so it can't accumulate duplicate
        passes when leaves share a transform list reference.
        """
        self.norm_stats = source.norm_stats
        self.key_types = source.key_types
        self.zarr_keys = source.zarr_keys
        self.shapes = source.shapes
        self.embodiments = source.embodiments
        self.norm_mode = source.norm_mode
        # Each MultiDataset keeps its own warning-dedup state.
        self._warned_violations = set()
        for ds in self.datasets.values():
            if isinstance(ds, MultiDataset):
                ds.set_norm_stats_from(source)

    def _check_bounds(
        self, data: dict, dataset, idx: int, dataset_name: str
    ) -> str | None:
        """Return a violation message if any tracked key in ``data`` has NaN/Inf
        or values outside per-key quantile bounds. ``None`` means the sample
        passes. Logs each (episode, key) violation once.
        """
        embodiment_id = data.get("embodiment")
        if embodiment_id is None:
            return None
        per_emb_stats = self.norm_stats.get(embodiment_id, {})
        if not per_emb_stats:
            return None

        episode_name = self._episode_name_for_dataset(dataset, dataset_name)

        for key_name, stats in per_emb_stats.items():
            zarr_key = self.zarr_keys.get(embodiment_id, {}).get(key_name)
            if zarr_key is None or zarr_key not in data:
                continue
            v = data[zarr_key]
            if isinstance(v, torch.Tensor):
                arr = v.float()
            elif isinstance(v, np.ndarray):
                arr = torch.from_numpy(v).float()
            else:
                continue

            q_low = stats.get(
                "quantile_0_01", stats.get("quantile_0_1", stats["quantile_1"])
            )
            q_high = stats.get(
                "quantile_99_99", stats.get("quantile_99_9", stats["quantile_99"])
            )
            q_low = torch.as_tensor(q_low, device=arr.device, dtype=torch.float32)
            q_high = torch.as_tensor(q_high, device=arr.device, dtype=torch.float32)
            try:
                q_low = torch.broadcast_to(q_low, arr.shape)
                q_high = torch.broadcast_to(q_high, arr.shape)
            except RuntimeError:
                # Stats were computed for a different layout than this sample
                # (e.g. a stale precomputed norm_stats.json). Say so once
                # instead of silently disabling the bounds check for the key;
                # normalize() raises on the same mismatch anyway.
                warn_key = f"bounds-shape:{zarr_key}"
                if warn_key not in self._warned_violations:
                    self._warned_violations.add(warn_key)
                    logger.warning(
                        f"[MultiDataset] bounds check skipped for {zarr_key}: "
                        f"stats shape {tuple(q_low.shape)} does not broadcast to "
                        f"sample shape {tuple(arr.shape)} (norm stats computed "
                        "for a different layout?)"
                    )
                continue

            if torch.any(torch.isnan(arr)) or torch.any(torch.isinf(arr)):
                prefix = f"NaN/Inf in {zarr_key} ep={episode_name} frame={idx}"
                warn_key = f"naninf:{episode_name}:{zarr_key}"
                if warn_key not in self._warned_violations:
                    self._warned_violations.add(warn_key)
                    logger.warning(prefix)
                return prefix

            # Rotation channels are either Euler ypr (wraps at +-pi) or
            # continuous 6D columns; quantile bounds on them are meaningless
            # and reject otherwise-valid frames, so only the translation /
            # gripper / keypoint channels are bounds-checked. Unrecognized
            # widths fall through to a full-vector check; NaN/Inf above still
            # covers the full vector.
            check_idx = _bounds_check_channels(zarr_key, arr.shape[-1])
            if check_idx is not None:
                arr_q = arr[..., check_idx]
                q_low = q_low[..., check_idx]
                q_high = q_high[..., check_idx]
            else:
                arr_q = arr

            # Slack on the quantile bounds: a relative part (a fraction of
            # each cell's own quantile range) plus a 1e-6 absolute floor.
            # The bounds are per (timestep, channel) cell and the check is
            # any-cell, so with 13 200 cells per keypoint chunk a strict
            # compare against a subsampled stats pass rejects valid frames in
            # bulk. Measured on the 2815-frame flagship episode through the
            # real keypoints_wristframe_6d pipeline, norm_mode=zscore, stats
            # from infer_norm_from_dataset(sample_frac=0.10) over its own
            # frames, 403 probe frames, a frame counted as rejected if ANY of
            # actions_keypoints / state.keypoints / state.ee_pose fails:
            #   slack    0     0.05   0.1    0.25   0.5    1.0
            #   any     39.5%  17.9%  11.4%   5.2%   3.2%   1.0%
            # The 132-channel action chunk drives it (39.0 % -> 3.2 %); the 8
            # checked ee_pose channels never exceed 2.7 %. Which frames the
            # stats came from moves the absolute numbers, so re-measure the
            # same way before comparing. Corrupt values
            # (fill constants, wrong-frame data) sit orders of magnitude
            # outside the range and are still caught.
            tol = self.BOUNDS_ABS_TOL + self.BOUNDS_REL_SLACK * (q_high - q_low)
            below = arr_q < q_low - tol
            above = arr_q > q_high + tol
            # Cells normalize() treats as constant (wrist-frame t=0, the kp0 /
            # kp9 structural zeros) cannot blow up the loss whatever offset
            # they carry, so don't reject on them. Which stat decides that is
            # norm_mode's business -- std under zscore, the quantile range
            # under quantile -- and every shipped config is zscore, so reading
            # q99 - q1 here would exempt cells the normalizer still divides by.
            constant = self._degenerate_cells(stats, arr.shape)
            if constant is not None:
                if check_idx is not None:
                    constant = constant[..., check_idx]
                below = below & ~constant
                above = above & ~constant
            if torch.any(below) or torch.any(above):
                prefix = f"Bounds violation in {zarr_key} ep={episode_name} frame={idx}"
                warn_key = f"bounds:{episode_name}:{zarr_key}"
                if warn_key not in self._warned_violations:
                    self._warned_violations.add(warn_key)
                    n_below = int(below.sum().item())
                    n_above = int(above.sum().item())
                    logger.warning(
                        f"{prefix} | n_below={n_below} n_above={n_above} "
                        f"arr_range=[{arr_q.min().item():.4f}, {arr_q.max().item():.4f}]"
                    )
                return prefix
        return None

    def _degenerate_cells(self, stats: dict, shape) -> torch.Tensor | None:
        """Mask of the cells ``normalize()`` treats as constant, or None when
        this key's stats do not carry the range ``norm_mode`` reads."""
        try:
            _, scale = _norm_center_scale(
                stats, self.norm_mode, torch.empty(0, dtype=torch.float32)
            )
        except (KeyError, ValueError):
            return None
        return torch.broadcast_to(_degenerate(scale, self.norm_mode), shape)

    # Retries stay within the failing episode this many times, then widen to
    # the full index space: a single wholly-bad episode must not exhaust the
    # sampler and take the run down (NCCL-timeouts the other ranks).
    GLOBAL_FALLBACK_ATTEMPTS = 25
    # Cap on consecutive bad samples, after which the worker raises. It is a
    # systemic-failure detector, not a tolerance dial: `attempts` resets per
    # __getitem__, so with p = the bad fraction of the whole index space the
    # cap fires with probability p**(N - GLOBAL_FALLBACK_ATTEMPTS), a step
    # function in p. At N = 100 the firing band sits at p >~ 0.8, so anything
    # it can detect is systemic; at N = 1000 it needed ~99 % of the store to be
    # bad, and got there via 975 sequential zarr/S3 reads per call per worker,
    # which looks like a hang rather than a failure.
    MAX_FALLBACK_ATTEMPTS = 100
    # Episodes tried per key_map/transform group before populate_from_datasets
    # falls back to the raw key_map.
    POPULATE_PROBES = 8

    def __getitem__(self, idx, _attempts: int | None = None):
        attempts = _attempts
        while True:
            dataset_name, local_idx = self.index_map[idx]
            dataset = self.datasets[dataset_name]
            try:
                data = dataset[local_idx]
            except Exception as e:
                reason = (
                    f"Sample failed ({type(e).__name__}: {e}) at "
                    f"{dataset_name}[{local_idx}]"
                )
                idx, attempts = self._next_after_failure(
                    idx, dataset_name, attempts, reason=reason
                )
                continue

            # If this leaf is itself a MultiDataset, it already ran bounds +
            # normalize for its returned sample. Pass through unchanged.
            if isinstance(dataset, MultiDataset):
                return data

            violation = self._check_bounds(data, dataset, local_idx, dataset_name)
            if violation is not None:
                idx, attempts = self._next_after_failure(
                    idx, dataset_name, attempts, reason=violation
                )
                continue

            # Bounds passed: normalize and return.
            if self.norm_stats and data.get("embodiment") in self.norm_stats:
                data = self.normalize(data, data["embodiment"])
            return data

    def __getitems__(self, idxs: list) -> list:
        """A DataLoader batch. Leaves with ``batched_fetch`` set (Lance streaming) do
        their reads for the whole batch at once, which is what makes random access from
        object storage affordable; every other sample, and any batched one that fails
        or trips the bounds check, goes through ``__getitem__`` and its fallback."""
        out = [None] * len(idxs)
        groups: dict[type, list] = {}
        for pos, idx in enumerate(idxs):
            name, local = self.index_map[idx]
            leaf = self.datasets[name]
            if getattr(leaf, "batched_fetch", False):
                groups.setdefault(type(leaf), []).append((pos, name, leaf, local))
        for cls, members in groups.items():
            planned = []
            for pos, name, leaf, local in members:
                try:
                    planned.append((pos, name, leaf, local, leaf.plan_sample(local)))
                except Exception:
                    continue
            if not planned:
                continue
            step = getattr(planned[0][2], "fetch_split", None) or len(planned)
            parts = [planned[i : i + step] for i in range(0, len(planned), step)]

            def fetch(part, cls=cls):
                return cls.fetch_planned([(m[2], m[4]) for m in part])

            if len(parts) > 1:
                # Parts download concurrently; each is assembled as soon as it lands,
                # so CPU work overlaps the remaining reads.
                futures = {_fetch_pool().submit(fetch, part): part for part in parts}
                done = ((futures[f], f.result) for f in cf.as_completed(futures))
            else:
                done = [(parts[0], functools.partial(fetch, parts[0]))]
            for part, result in done:
                try:
                    fetched = result()
                except Exception:
                    logger.warning(
                        "[MultiDataset] batched fetch failed; reading samples one by one",
                        exc_info=True,
                    )
                    continue
                for (pos, name, leaf, local, plan), got in zip(part, fetched):
                    try:
                        data = leaf.assemble_sample(plan, got)
                    except Exception:
                        continue
                    if self._check_bounds(data, leaf, local, name) is not None:
                        continue
                    if self.norm_stats and data.get("embodiment") in self.norm_stats:
                        data = self.normalize(data, data["embodiment"])
                    out[pos] = data
        return [self[idx] if d is None else d for d, idx in zip(out, idxs)]

    def _next_after_failure(
        self, idx: int, dataset_name: str, attempts: int | None, *, reason: str
    ) -> tuple[int | None, int]:
        """Pick the next index to try after a bad sample at ``idx``.

        Raises once ``MAX_FALLBACK_ATTEMPTS`` consecutive samples failed."""
        attempts = (attempts or 0) + 1
        if attempts >= self.MAX_FALLBACK_ATTEMPTS:
            raise RuntimeError(
                f"[MultiDataset] {attempts} consecutive bad samples, the last "
                f"{attempts - self.GLOBAL_FALLBACK_ATTEMPTS} drawn uniformly "
                "from the whole index space: the store, the keymap or the norm "
                f"stats are broken, not one episode. Last: {reason}"
            )
        if attempts <= self.GLOBAL_FALLBACK_ATTEMPTS:
            candidates = [
                c for c in self._global_indices_by_dataset[dataset_name] if c != idx
            ]
        else:
            candidates = None
        if candidates:
            next_idx = random.choice(candidates)
        else:
            next_idx = random.randrange(len(self.index_map))
        next_dataset_name, next_local_idx = self.index_map[next_idx]
        # Every retry used to log at WARNING: tens of thousands of lines per
        # run. Keep the first few per process, then one in every 500.
        self._retry_log_count = getattr(self, "_retry_log_count", 0) + 1
        if self._retry_log_count <= 20 or self._retry_log_count % 500 == 0:
            logger.warning(
                f"{reason} | attempt {attempts}, "
                f"trying {next_dataset_name}[{next_local_idx}] "
                f"(retries so far in this worker: {self._retry_log_count})"
            )
        return next_idx, attempts

    @classmethod
    def _from_resolver(cls, resolver, dataset_name: str | None = None, **kwargs):
        """create a MultiDataset from a resolver: anything with
        ``resolve(filters=, expected_embodiment=) -> {name: leaf}``.

        dataset_name: the config key this dataset is built under (trainHydra passes
        it). When it names an embodiment, pinned episodes must carry that
        embodiment (PinError otherwise) and other mismatches are logged.

        Every pinned episode must also end up in the dataset: a pin that
        filter_lambdas or the debug limit exclude, or that fails to load or is
        skipped by the resolver, is a PinError rather than a silent drop."""
        sync_from_s3 = kwargs.pop("sync_from_s3", False)
        filters = kwargs.pop("filters", None)
        expected = None
        if dataset_name is not None:
            try:
                get_embodiment_id(dataset_name)
                expected = dataset_name
            except (KeyError, AttributeError):
                expected = None

        # Only LocalEpisodeResolver can sync; other resolvers reject the kwarg.
        extra = {"sync_from_s3": True} if sync_from_s3 else {}
        resolved = resolver.resolve(
            filters=filters, expected_embodiment=expected, **extra
        )

        pins = filters.episode_hashes if isinstance(filters, DatasetFilter) else ()
        dropped = sorted(set(pins) - set(resolved))
        if dropped:
            raise PinError(
                f"{len(dropped)} pinned episode(s) not in dataset "
                f"'{dataset_name or '<unnamed>'}': {dropped}. They were excluded by "
                "filter_lambdas or the resolver's debug limit, or failed to load / "
                "were skipped by the resolver (see the log above)."
            )

        return cls(datasets=resolved, **kwargs)

    # =====================================================================
    # Stats / normalization
    # =====================================================================

    @staticmethod
    def _iter_leaves(ds):
        """Yield non-MultiDataset leaves from possibly nested wrappers."""
        if isinstance(ds, MultiDataset):
            for child in ds.datasets.values():
                yield from MultiDataset._iter_leaves(child)
        else:
            yield ds

    def populate_from_datasets(self, datasets: dict | None = None) -> None:
        """
        Populate per-embodiment key inventory by walking leaves and probing
        one post-transform sample per (embodiment, key_map, transform list).
        ``datasets`` defaults to ``self.datasets`` so the typical call is just
        ``mds.populate_from_datasets()``.
        """
        graph = datasets if datasets is not None else self.datasets
        groups: dict[tuple, list] = {}
        for ds in graph.values():
            for leaf in self._iter_leaves(ds):
                emb = getattr(leaf, "embodiment", None)
                key_map = getattr(leaf, "key_map", None)
                if emb is None or key_map is None:
                    continue
                emb_id = emb if isinstance(emb, int) else get_embodiment_id(emb)
                # A resolver hands every episode the same key_map and transform
                # objects, so one probe covers them all: probing each leaf read a
                # full sample per episode (~20 min on 41k mecka episodes).
                key = (emb_id, id(key_map), id(getattr(leaf, "transform", None)))
                groups.setdefault(key, [emb_id, key_map, []])[2].append(leaf)

        for emb_id, key_map, leaves in groups.values():
            self.embodiments.add(emb_id)
            self.key_types.setdefault(emb_id, {})
            self.zarr_keys.setdefault(emb_id, {})
            self.shapes.setdefault(emb_id, {})
            self.norm_stats.setdefault(emb_id, {})

            sample_keys: set | None = None
            for leaf in leaves[: self.POPULATE_PROBES]:
                try:
                    sample = leaf[0]
                except Exception as e:
                    logger.warning(
                        f"[MultiDataset] Could not probe leaf for post-transform "
                        f"keys (emb={emb_id}): {e}. Trying the next episode."
                    )
                    continue
                if isinstance(sample, dict):
                    sample_keys = set(sample.keys())
                    break
            if sample_keys is None:
                logger.warning(
                    f"[MultiDataset] No probe succeeded (emb={emb_id}). "
                    "Falling back to raw key_map."
                )
                for key_name, info in key_map.items():
                    self.key_types[emb_id][key_name] = info.get(
                        "key_type", "metadata_keys"
                    )
                    self.zarr_keys[emb_id][key_name] = info["zarr_key"]
                continue

            # Identity zarr_keys map (data_key is the algo-side name).
            for data_key in sample_keys:
                if data_key in key_map:
                    info = key_map[data_key]
                    self.key_types[emb_id][data_key] = info.get(
                        "key_type", "metadata_keys"
                    )
                else:
                    inferred = _infer_key_type(data_key)
                    if inferred is None:
                        continue
                    self.key_types[emb_id][data_key] = inferred
                self.zarr_keys[emb_id][data_key] = data_key

    # ---- key lookups ----

    def keys_of_type(self, key_type: str, embodiment_id: int) -> list[str]:
        return [
            k for k, t in self.key_types.get(embodiment_id, {}).items() if t == key_type
        ]

    def is_key_with_embodiment(self, key_name: str, embodiment_id: int) -> bool:
        return key_name in self.key_types.get(embodiment_id, {})

    def keyname_to_zarr_key(self, key_name: str, embodiment_id: int) -> str | None:
        return self.zarr_keys.get(embodiment_id, {}).get(key_name)

    def zarr_key_to_keyname(self, zarr_key: str, embodiment_id: int) -> str | None:
        for k, v in self.zarr_keys.get(embodiment_id, {}).items():
            if v == zarr_key:
                return k
        return None

    def key_shape(self, key_name: str, embodiment_id: int) -> tuple:
        if key_name not in self.shapes.get(embodiment_id, {}):
            raise ValueError(
                f"Shape for key {key_name!r} on embodiment {embodiment_id} not inferred yet."
            )
        return self.shapes[embodiment_id][key_name]

    # ---- shape & norm inference ----

    def infer_shapes_from_batch(self, batch: dict) -> None:
        for emb_id, per_emb in self.zarr_keys.items():
            for key_name, zarr_key in per_emb.items():
                if zarr_key in batch:
                    val = batch[zarr_key]
                    if hasattr(val, "shape"):
                        self.shapes.setdefault(emb_id, {})[key_name] = tuple(val.shape)
                    elif isinstance(val, int):
                        self.shapes.setdefault(emb_id, {})[key_name] = (1,)

    def infer_norm_from_dataset(
        self,
        dataset,
        dataset_name,
        sample_frac: float = 0.10,
        seed: int = 42,
        max_samples: int | None = None,
        batch_size: int = 512,
        num_workers: int = 4,
        precomputed_norm_path: str | None = None,
        pool_horizon: bool = False,
        episode_cache: str | Path | None = None,
    ):
        """Fill ``self.norm_stats[embodiment]`` from a precomputed file or by
        sampling ``dataset``.

        ``episode_cache``: a per-episode sample store
        (``episode_norm_samples.cache_root``). Rows are read from, and new
        episodes written to, it instead of sampling the whole dataset.

        ``pool_horizon``: compute each chunked key's stats over every timestep
        of the chunk (one set per channel, like openpi) instead of per
        (timestep, channel) cell. The pooled stats are tiled back to the
        ``(T, D)`` shape, so normalize / bounds check / checkpoints see the
        same layout either way.
        """
        embodiment = dataset_name
        if isinstance(embodiment, str):
            embodiment = get_embodiment_id(embodiment)

        norm_keys = list(self.keys_of_type("proprio_keys", embodiment))
        norm_keys.extend(self.keys_of_type("action_keys", embodiment))
        if not norm_keys:
            logger.warning(
                f"[MultiDataset] No proprio/action keys for embodiment={embodiment}"
            )
            return

        self.norm_stats.setdefault(embodiment, {})

        if precomputed_norm_path is not None:
            if os.path.isdir(precomputed_norm_path):
                precomputed_file = os.path.join(
                    precomputed_norm_path, "norm_stats.json"
                )
            elif os.path.isfile(precomputed_norm_path):
                precomputed_file = precomputed_norm_path
            else:
                logger.warning(
                    f"[MultiDataset] precomputed_norm_path={precomputed_norm_path} is not valid"
                )
                return
            if os.path.isfile(precomputed_file):
                with open(precomputed_file, "r") as f:
                    payload = json.load(f)
                if str(embodiment) not in payload["stats"]:
                    raise ValueError(
                        f"norm_stats file {precomputed_file} has no entry for "
                        f"embodiment id {embodiment} (available: "
                        f"{sorted(payload['stats'])}). Stats are keyed by numeric "
                        "EMBODIMENT id, and ids were renumbered by the human/eva "
                        "embodiment collapse — recompute norm stats instead of "
                        "reusing a pre-collapse norm_stats.json."
                    )
                # JSON gives back nested lists; a fresh computation stores
                # float32 arrays, so convert to keep both paths identical.
                self.norm_stats[embodiment] = {
                    k: {n: np.asarray(a, dtype=np.float32) for n, a in d.items()}
                    for k, d in payload["stats"][str(embodiment)].items()
                }
                self._norm_run_metadata = payload.get("norm_run_metadata", None)
                logger.info(
                    f"[MultiDataset] Loaded precomputed stats for embodiment={embodiment}"
                )
                return

        if episode_cache is not None and not all(
            not isinstance(ds, MultiDataset)
            and getattr(ds, "episode_path", None) is not None
            for ds in dataset.datasets.values()
        ):
            logger.warning(
                "[MultiDataset] per-episode norm cache needs a flat dataset of "
                "zarr episodes; sampling the whole dataset instead"
            )
            episode_cache = None

        N = len(dataset)
        if N <= 0:
            raise ValueError("Dataset is empty")
        n_samples = int(math.ceil(sample_frac * N))
        n_samples = max(1, min(n_samples, N))
        if max_samples is not None:
            n_samples = min(n_samples, max_samples)

        logger.info(f"[MultiDataset] embodiment={embodiment} norm_keys={norm_keys}")
        logger.info(
            f"[MultiDataset] sampling {n_samples}/{N} (~{100 * sample_frac:.1f}%)"
        )

        loading_start = time.time()
        episode_meta = {}
        if episode_cache is not None:
            from egomimic.rldb.norm import episode_norm_samples

            zarr_keys = {
                k: z
                for k in norm_keys
                if (z := self.keyname_to_zarr_key(k, embodiment)) is not None
            }
            collected, episode_meta = episode_norm_samples.collect(
                Path(episode_cache),
                dataset.datasets,
                zarr_keys,
                n_samples,
                seed=seed,
                num_workers=num_workers,
            )
            for k in norm_keys:
                collected.setdefault(k, [])
        else:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
                worker_init_fn=die_with_parent,
                multiprocessing_context=(
                    worker_context([dataset]) if num_workers > 0 else None
                ),
            )
            collected = self._collect_norm_samples(
                loader, norm_keys, embodiment, n_samples, batch_size, num_workers
            )
        for k in [k for k, v in collected.items() if not v]:
            del collected[k]
            norm_keys.remove(k)
        loading_time = time.time() - loading_start

        computing_start = time.time()
        for k in norm_keys:
            collected[k] = self._drop_nonfinite_rows(
                np.concatenate(collected[k], axis=0), k
            )
            X = collected[k]
            if pool_horizon and X.ndim > 2:
                stats_np = self._compute_stats_for_array(X.reshape(-1, X.shape[-1]))
                stats_np = {
                    name: np.broadcast_to(arr, X.shape[1:]).copy()
                    for name, arr in stats_np.items()
                }
            else:
                stats_np = self._compute_stats_for_array(X)
            self.norm_stats[embodiment][k] = {
                name: np.asarray(arr, dtype=np.float32)
                for name, arr in stats_np.items()
            }
            logger.info(
                f"[MultiDataset] key={k} samples={collected[k].shape[0]} stat_shape={stats_np['mean'].shape}"
            )
        computing_time = time.time() - computing_start

        self._norm_run_metadata = {
            "loading_time": loading_time,
            "computing_time": computing_time,
            "frames": n_samples,
            **episode_meta,
        }
        logger.info(
            f"[MultiDataset] Finished norm inference, loading={loading_time:.2f}s, computing={computing_time:.2f}s"
        )

    def _collect_norm_samples(
        self, loader, norm_keys, embodiment, n_samples, batch_size, num_workers
    ):
        collected = {k: [] for k in norm_keys}
        cur = 0
        with tqdm(total=n_samples, unit="sample") as pbar:
            for batch in loader:
                remaining = n_samples - cur
                if remaining <= 0:
                    break
                batch_len = None
                for value in batch.values():
                    if hasattr(value, "shape") and len(value.shape) > 0:
                        batch_len = int(value.shape[0])
                        break
                if batch_len is None:
                    raise ValueError(
                        "[MultiDataset] Could not infer batch size from DataLoader batch"
                    )
                take = min(remaining, batch_len)
                for k in norm_keys:
                    zarr_key = self.keyname_to_zarr_key(k, embodiment)
                    if zarr_key is None or zarr_key not in batch:
                        continue
                    x = batch[zarr_key][:take]
                    if hasattr(x, "detach"):
                        x = x.detach().cpu().numpy()
                    collected[k].append(x)
                cur += take
                pbar.update(take)
        return collected

    @staticmethod
    def _drop_nonfinite_rows(X: np.ndarray, key: str) -> np.ndarray:
        """Drop norm samples (rows of X, shape (N, ...)) containing NaN/Inf.

        np.percentile/mean propagate NaN, and a NaN quantile then poisons every
        normalized sample of that key (the per-sample bounds check cannot flag
        NaN bounds). Episodes with a non-finite feature (e.g. an all-NaN gripper
        command) are still rejected sample-by-sample at train time; they must
        not shape the statistics.
        """
        finite = np.isfinite(X).reshape(X.shape[0], -1).all(axis=1)
        n_bad = int((~finite).sum())
        if n_bad == 0:
            return X
        if n_bad == X.shape[0]:
            raise ValueError(
                f"[MultiDataset] key={key}: every collected norm sample is non-finite"
            )
        logger.warning(
            f"[MultiDataset] key={key}: dropping {n_bad}/{X.shape[0]} norm samples "
            "with non-finite values"
        )
        return X[finite]

    @staticmethod
    def _compute_stats_for_array(X):
        return {
            "mean": np.mean(X, axis=0),
            "std": np.std(X, axis=0),
            "min": np.min(X, axis=0),
            "max": np.max(X, axis=0),
            "median": np.median(X, axis=0),
            "quantile_1": np.percentile(X, 1, axis=0),
            "quantile_99": np.percentile(X, 99, axis=0),
            # Bounds-check quantiles snap to observed samples ("lower" /
            # "higher") instead of interpolating: with fewer than 10k samples
            # the default linear percentile lands strictly inside the sample
            # range, so the extreme frames of the very episodes the stats were
            # computed on were rejected at train time (3-16 % of a single
            # episode in the overfit runs).
            "quantile_0_01": np.percentile(X, 0.01, axis=0, method="lower"),
            "quantile_99_99": np.percentile(X, 99.99, axis=0, method="higher"),
        }

    def cache_stats(self, save_cache_dir: str):
        cache_dir = os.path.join(save_cache_dir, "norm_stats")
        os.makedirs(cache_dir, exist_ok=True)
        out_path = os.path.join(cache_dir, "norm_stats.json")

        stats_out: dict[str, dict[str, dict[str, list]]] = {}
        for emb, keys_dict in self.norm_stats.items():
            stats_out[str(emb)] = {
                k: {name: np.asarray(arr).tolist() for name, arr in stat_dict.items()}
                for k, stat_dict in keys_dict.items()
            }
        payload = {
            "stats": stats_out,
            "loading_time": None,
            "computing_time": None,
            "frames": None,
        }
        if self._norm_run_metadata is not None:
            for k in ("loading_time", "computing_time", "frames"):
                if k in self._norm_run_metadata:
                    payload[k] = self._norm_run_metadata[k]
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=4)
        logger.info(f"[MultiDataset] Cached stats to {out_path}")

    # ---- normalize / unnormalize ----

    def _apply_norm_one(self, tensor, stats, zarr_key=None):
        return _apply_norm_one(
            tensor, stats, self.norm_mode, self._identity_channels(zarr_key, tensor)
        )

    def _apply_unnorm_one(self, tensor, stats, zarr_key=None):
        return _apply_unnorm_one(
            tensor, stats, self.norm_mode, self._identity_channels(zarr_key, tensor)
        )

    def _identity_channels(self, zarr_key, tensor):
        """Channels this key leaves unnormalized: the rot6d columns.

        A 6D rotation is two columns of a rotation matrix, already bounded in
        [-1, 1], and per-channel scaling and shifting pulls the pair off the
        manifold for no gain in resolution -- which is why the two projects
        that anchor an action chunk at near-identity, TRI's LBM and UMI, both
        exclude rotation from normalization outright. It also removes the t=0
        identity-rotation degeneracy at its source instead of catching it with
        NORM_MIN_RANGE.
        """
        if zarr_key is None or not self.rot6d_passthrough:
            return None
        return rot6d_channels(zarr_key, tensor.shape[-1])

    def normalize(self, data: dict, embodiment_id: int) -> dict:
        if not self.norm_stats.get(embodiment_id):
            return data
        out = dict(data)
        for key_name, key_type in self.key_types.get(embodiment_id, {}).items():
            if key_type not in self.NORMALIZE_KEY_TYPES:
                continue
            stats = self.norm_stats[embodiment_id].get(key_name)
            if stats is None:
                continue
            zarr_key = self.zarr_keys[embodiment_id][key_name]
            if zarr_key not in out:
                continue
            tensor = out[zarr_key]
            if not isinstance(tensor, torch.Tensor):
                if isinstance(tensor, np.ndarray):
                    tensor = torch.from_numpy(tensor).float()
                else:
                    continue
            out[zarr_key] = self._apply_norm_one(tensor, stats, zarr_key)
        return out

    def unnormalize(self, data: dict, embodiment_id: int) -> dict:
        if not self.norm_stats.get(embodiment_id):
            return data
        out = dict(data)
        zk_to_kn = {v: k for k, v in self.zarr_keys.get(embodiment_id, {}).items()}
        for data_key, value in list(data.items()):
            key_name = (
                data_key
                if data_key in self.norm_stats[embodiment_id]
                else zk_to_kn.get(data_key)
            )
            if key_name is None:
                continue
            stats = self.norm_stats[embodiment_id].get(key_name)
            if stats is None:
                continue
            if not isinstance(value, torch.Tensor):
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value).float()
                else:
                    continue
            out[data_key] = self._apply_unnorm_one(
                value, stats, self.zarr_keys.get(embodiment_id, {}).get(key_name)
            )
        return out

    # ---- transform attachment ----

    def attach_normalize_transforms(
        self, datasets: dict | None = None, reject_outliers: bool = True
    ) -> None:
        """Deprecated. Use ``set_norm_stats_from`` on each training/valid
        MultiDataset instead. Bounds-check + normalize now run at the
        MultiDataset level in ``__getitem__``, not as per-leaf transforms.

        Kept as a thin shim that calls ``set_norm_stats_from(self)`` on each
        MultiDataset in ``datasets`` so existing callers keep working. The
        ``reject_outliers`` flag is no longer honored — bounds checking is
        always on when stats are populated. To disable, clear ``norm_stats``.
        """
        del reject_outliers  # unused
        graph = datasets if datasets is not None else self.datasets
        for ds in graph.values():
            if isinstance(ds, MultiDataset):
                ds.set_norm_stats_from(self)

    # ---- serialization (checkpoint roundtrip) ----

    @staticmethod
    def _clone_norm_stats(norm_stats):
        out = {}
        for emb, per_emb in (norm_stats or {}).items():
            out[emb] = {
                key: {
                    name: (
                        v.detach().cpu().clone()
                        if torch.is_tensor(v)
                        else copy.deepcopy(v)
                    )
                    for name, v in stats.items()
                }
                for key, stats in per_emb.items()
            }
        return out

    def to_state(self) -> dict:
        """Serialize stats only (not the dataset graph). Suitable for checkpoint."""
        return {
            "norm_mode": self.norm_mode,
            "embodiments": sorted(self.embodiments),
            "key_types": copy.deepcopy(self.key_types),
            "zarr_keys": copy.deepcopy(self.zarr_keys),
            "shapes": copy.deepcopy(self.shapes),
            "norm_stats": self._clone_norm_stats(self.norm_stats),
            "rot6d_passthrough": self.rot6d_passthrough,
        }

    @classmethod
    def from_state(cls, state: dict) -> "MultiDataset":
        """Reconstruct a stats-only MultiDataset (no dataset graph) from state."""
        if state is None:
            raise ValueError("MultiDataset state must be provided for reconstruction.")
        return cls(state=state)

    def _load_state(self, state: dict) -> None:
        """Populate stats fields from a state dict. Used by state-only construction."""
        self.norm_mode = state.get("norm_mode", self.norm_mode)
        self.embodiments = set(state.get("embodiments", []))
        self.key_types = copy.deepcopy(state.get("key_types", {}))
        self.zarr_keys = copy.deepcopy(state.get("zarr_keys", {}))
        self.shapes = copy.deepcopy(state.get("shapes", {}))
        self.norm_stats = self._clone_norm_stats(state.get("norm_stats", {}))
        # A checkpoint state saved before the flag existed normalized the rot6d
        # channels; an empty state (fresh stats, e.g. trainHydra) keeps the default.
        if "rot6d_passthrough" in state:
            self.rot6d_passthrough = bool(state["rot6d_passthrough"])
        elif state.get("norm_stats"):
            self.rot6d_passthrough = False
        for emb in self.embodiments:
            self.key_types.setdefault(emb, {})
            self.zarr_keys.setdefault(emb, {})
            self.shapes.setdefault(emb, {})
            self.norm_stats.setdefault(emb, {})


# ---------------------------------------------------------------------------
# Per-episode subsampling wrapper around MultiDataset (K evenly-spaced
# frames OR every Sth frame within each episode).
# ---------------------------------------------------------------------------
def _evenly_spaced_indices(n: int, k: int) -> list[int]:
    if n <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n // 2]
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def _strided_indices(n: int, stride: int) -> list[int]:
    if n <= 0 or stride <= 0:
        return []
    return list(range(0, n, stride))


class EvenStrideDataset(MultiDataset):
    """Wraps a `MultiDataset` to subsample frames per underlying episode.
    Pass exactly one of `frames_per_episode` (K evenly-spaced) or
    `stride` (every Sth frame). Subclasses `MultiDataset` so trainHydra's
    isinstance check passes; we skip the parent `__init__` and adopt its
    class identity only, delegating to `self.base`."""

    def __init__(
        self, base, frames_per_episode: int | None = None, stride: int | None = None
    ):
        if (frames_per_episode is None) == (stride is None):
            raise ValueError(
                "EvenStrideDataset requires exactly ONE of "
                "`frames_per_episode` or `stride` to be set "
                f"(got frames_per_episode={frames_per_episode}, stride={stride})."
            )
        gibd = getattr(base, "_global_indices_by_dataset", None)
        if gibd is None:
            raise ValueError(
                f"EvenStrideDataset requires a MultiDataset exposing "
                f"_global_indices_by_dataset, got {type(base).__name__}"
            )
        torch.utils.data.Dataset.__init__(self)
        self.base = base
        self.frames_per_episode = frames_per_episode
        self.stride = stride
        self.datasets = base.datasets

        keep: list[int] = []
        for ep_name, idxs in gibd.items():
            n = len(idxs)
            if stride is not None:
                picks = _strided_indices(n, stride)
                rule = f"stride={stride}"
            else:
                picks = _evenly_spaced_indices(n, frames_per_episode)
                rule = f"k={frames_per_episode}"
            chosen = [idxs[i] for i in picks]
            actual_stride = n / max(1, len(chosen))
            logger.info(
                "EvenStrideDataset: %s -> %d / %d frames (%s, actual stride ~%.1f)",
                ep_name,
                len(chosen),
                n,
                rule,
                actual_stride,
            )
            keep.extend(chosen)
        self.indices = sorted(keep)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]

    def set_data_schematic(self, data_schematic) -> None:
        self.base.set_data_schematic(data_schematic)
        self.data_schematic = data_schematic

    def set_norm_stats_from(self, source: "MultiDataset") -> None:
        """Delegate to ``self.base`` as well: ``__getitem__`` here forwards to
        ``self.base[...]``, so the bounds-check + normalize pass reads the
        *base*'s stats. Setting them on the wrapper alone (the inherited
        implementation) would leave the subsampled val loader unnormalized."""
        self.base.set_norm_stats_from(source)
        MultiDataset.set_norm_stats_from(self, source)

    def __getattr__(self, item):
        if item == "base":
            raise AttributeError(item)
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(item)
        return getattr(base, item)


def pinned_episode_subset(
    base: MultiDataset, episode_hashes, dataset_name: str | None = None
) -> MultiDataset:
    """A contiguous, unsubsampled ``MultiDataset`` over just ``episode_hashes``
    of an already-built ``base``.

    Used for the video-only val loaders: the overlay video must be watchable, so
    it renders whole episodes in frame order rather than the metric loader's
    per-episode subsample.

    The episodes are taken from ``base.datasets`` (the same leaf
    objects), NOT re-resolved: a second ``MultiDataset._from_resolver`` with
    ``filters.episode_hashes`` set would be a different ``resolve_once`` memo key
    and so a second path resolution / zarr open per head. A pin that is not in
    ``base`` is a ``PinError``, matching ``_from_resolver``'s pin contract.

    ``episode_hashes`` order is preserved, so the video plays the pinned
    episodes in config order.
    """
    hashes = [str(h) for h in (episode_hashes or [])]
    missing = [h for h in hashes if h not in base.datasets]
    if missing:
        raise PinError(
            f"{len(missing)} pinned video episode(s) not in dataset "
            f"'{dataset_name or '<unnamed>'}': {sorted(missing)}. The pin must "
            "name an episode of THIS head's split (e.g. a seen-val episode for "
            "`valid`, a held-out-operator episode for `unseen_op_valid`)."
        )
    if not hashes:
        raise PinError(
            f"pinned_episode_subset called with no episode hashes for "
            f"'{dataset_name or '<unnamed>'}'"
        )
    subset = MultiDataset(
        datasets={h: base.datasets[h] for h in hashes},
        mode="total",
        norm_mode=base.norm_mode,
    )
    subset.set_norm_stats_from(base)
    return subset
