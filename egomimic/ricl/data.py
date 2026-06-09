"""RICL data layer: attach k retrieved (image, state, action) blocks per query. (P3)

Three pieces (kept import-light: torch + numpy; zarr / pl imported lazily):

- ``RiclQueryDataset``: thin wrapper over a query ``MultiDataset`` that surfaces
  ``frame_idx`` on each sample (derived from ``MultiDataset.index_map``), since
  the stock ``ZarrDataset.__getitem__`` returns ``episode_hash`` but not the
  frame index — and the retrieval cache is keyed by ``(episode_hash, frame_idx)``.
- ``BankFrameProvider`` / ``ZarrBankFrameProvider``: load a single bank frame's
  (image, state, action) by ``(episode_hash, frame_idx)``, normalized into the
  query's 32-D convention (so retrieved bins are comparable to the query State).
- ``build_ricl_collate``: wrap the base collate; for each query sample look up its
  top-k neighbors (``RetrievalCache``) and gather their blocks via the provider
  into batch keys ``ricl_retrieved_{images,state,action,mask,dist}`` consumed by
  :class:`egomimic.algo.pi_ricl.PIRicl`.

Samples with no cache entry (or fewer than k neighbors) are zero-padded and
mask=False — i.e. the model sees k slots but the invalid ones are masked out and
omitted from the prompt, gracefully degrading toward the zero-context floor.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Callable, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bank norm stats: load + apply (mirrors MultiDataset's stats machinery)
# ---------------------------------------------------------------------------


def load_bank_norm_stats(path: str, embodiment_id: int) -> dict:
    """Load one embodiment's per-key stats from a ``norm_stats.json``.

    ``path`` may be the json file itself or a directory containing
    ``norm_stats.json`` (possibly under a ``norm_stats/`` subdir — the layout
    ``MultiDataset.cache_stats`` writes). Returns
    ``{key_name: {stat_name: np.ndarray}}`` for ``payload["stats"][str(id)]``,
    matching what ``MultiDataset.infer_norm_from_dataset(precomputed_norm_path=...)``
    loads for the query path — point both at the SAME file so retrieved bins
    are comparable to the query State block.
    """
    p = path
    if os.path.isdir(p):
        for cand in ("norm_stats.json", os.path.join("norm_stats", "norm_stats.json")):
            c = os.path.join(p, cand)
            if os.path.isfile(c):
                p = c
                break
    if not os.path.isfile(p):
        raise FileNotFoundError(f"bank norm stats not found at {path!r}")
    with open(p) as f:
        payload = json.load(f)
    stats = payload.get("stats", {}).get(str(int(embodiment_id)))
    if not stats:
        raise ValueError(
            f"{p} has no stats for embodiment {embodiment_id} "
            f"(available: {sorted(payload.get('stats', {}).keys())})"
        )
    return {
        key: {name: np.asarray(arr, dtype=np.float32) for name, arr in d.items()}
        for key, d in stats.items()
    }


def normalize_with_stats(arr, stats: dict, norm_mode: str = "quantile") -> np.ndarray:
    """Numpy mirror of ``MultiDataset._apply_norm_one`` (same formulas/epsilon).

    Stats arrays may carry more leading rows than ``arr``: action stats are
    computed per chunk position with shape ``(chunk, dim)`` while a retrieved
    action keeps only the first ``action_horizon`` steps — slice the leading
    axis so retrieved step ``t`` is normalized with stats row ``t``, exactly as
    the query path normalizes the full chunk.
    """
    a = np.asarray(arr, dtype=np.float32)

    def _stat(name: str) -> np.ndarray:
        s = np.asarray(stats[name], dtype=np.float32)
        # Slice extra leading (chunk) rows only for >=2-D values — for 1-D the
        # single axis is the feature axis and any mismatch is an error.
        if a.ndim >= 2 and s.ndim == a.ndim and s.shape[0] > a.shape[0]:
            s = s[: a.shape[0]]
        if s.shape != a.shape and s.shape != a.shape[-s.ndim :]:
            raise ValueError(
                f"bank norm stats shape {s.shape} incompatible with value shape "
                f"{a.shape} — stats were computed over a different representation"
            )
        return s

    if norm_mode == "zscore":
        return (a - _stat("mean")) / (_stat("std") + 1e-6)
    if norm_mode == "minmax":
        mn, mx = _stat("min"), _stat("max")
        return 2.0 * ((a - mn) / (mx - mn + 1e-6)) - 1.0
    if norm_mode == "quantile":
        q1, q99 = _stat("quantile_1"), _stat("quantile_99")
        return 2.0 * ((a - q1) / (q99 - q1 + 1e-6)) - 1.0
    raise ValueError(f"Invalid normalization mode: {norm_mode}")


# ---------------------------------------------------------------------------
# Query dataset wrapper: surface frame_idx
# ---------------------------------------------------------------------------


class RiclQueryDataset(Dataset):
    """Wrap a query ``MultiDataset`` so each sample carries ``frame_idx``.

    ``MultiDataset.index_map[global_idx] -> (dataset_name, local_idx)`` where
    ``local_idx`` is the start frame within the episode (== the value passed to
    the per-episode ``ZarrDataset.__getitem__``). We read it for the requested
    global index and stamp it on the sample. ``episode_hash`` (authoritative) is
    already set by the base dataset; ``frame_idx`` is best-effort (a rare
    bad-frame fallback inside the base may shift the actual frame, in which case
    the cache lookup is clamped — acceptable for per-observation retrieval).
    """

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        if "frame_idx" not in sample:
            try:
                _, local_idx = self.base.index_map[idx]
                sample["frame_idx"] = int(local_idx)
            except Exception:
                sample["frame_idx"] = 0
        return sample

    def __getattr__(self, name: str):
        # Delegate everything else (e.g. set_norm_stats_from and other
        # MultiDataset hooks trainHydra calls) to the wrapped base dataset.
        # Guard ``base`` to avoid infinite recursion before __init__ sets it.
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)


# ---------------------------------------------------------------------------
# Bank frame provider
# ---------------------------------------------------------------------------


class BankFrameProvider(Protocol):
    """Loads one bank frame's retrieved block, normalized to the query convention."""

    def __call__(self, episode_hash: str, frame_idx: int) -> dict:
        """Return ``{"image": (C,H,W) or (H,W,C), "state": (Ds,), "action": (Ha,Da)}``."""
        ...


class ZarrBankFrameProvider:
    """Load (image, state, action) for a bank frame via the embodiment's own
    keymap + transform_list (the same per-episode pipeline the query path uses),
    so retrieved frames live in the identical post-transform representation.

    Mirrors the query pipeline's value processing exactly (MultiDataset
    normalizes in ``__getitem__``, then ``PI._robomimic_to_pi_data`` applies
    ``to32`` — see ``pi.py``): state and action are quantile-normalized with the
    *bank embodiment's* stats (``norm_stats``), then the action goes through the
    embodiment's converter ``to32`` into the shared 32-D space (aria's missing
    gripper -> slot 0; see ``HumanBimanualCartesianEuler``). The state is never
    converted — the query State block uses normalized raw dims too.

    Without ``norm_stats`` the retrieved values stay in raw physical units,
    whose discretized bins do NOT match the query State block — acceptable only
    for smoke tests. Cluster-side; unit tests use a mock provider.
    """

    def __init__(
        self,
        resolve_store: Callable[[str], str],
        converter=None,
        keymap: dict | None = None,
        transform_list: list | None = None,
        image_key: str = "base_0_rgb",
        state_key: str = "observations.state.ee_pose",
        action_key: str = "actions_cartesian",
        action_horizon: int = 15,
        norm_stats: dict | None = None,
        norm_mode: str = "quantile",
    ):
        self.resolve_store = resolve_store
        self.converter = converter
        self.keymap = keymap
        self.transform_list = transform_list
        self.image_key = image_key
        self.state_key = state_key
        self.action_key = action_key
        self.action_horizon = action_horizon
        self.norm_stats = norm_stats
        self.norm_mode = norm_mode
        if norm_stats is not None:
            missing = [k for k in (state_key, action_key) if k not in norm_stats]
            if missing:
                raise ValueError(
                    f"bank norm_stats missing keys {missing} "
                    f"(available: {sorted(norm_stats.keys())})"
                )
        else:
            logger.warning(
                "ZarrBankFrameProvider: no norm_stats — retrieved state/action "
                "stay in raw units and their bins will NOT match the query "
                "State block. Pass data.bank_norm_path for real training."
            )
        self._cache: dict = {}

    def _normalize(self, key: str, arr: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            return arr
        return normalize_with_stats(arr, self.norm_stats[key], self.norm_mode)

    def _open(self, episode_hash: str):
        if episode_hash not in self._cache:
            # Build the bank embodiment's own per-episode dataset (keymap +
            # transform_list), so each retrieved frame goes through the SAME
            # SLAM->head-frame / chunk / concat pipeline as query samples. The
            # post-transform keys (``state_key`` / ``action_key`` / ``image_key``)
            # are produced on the fly -- they are never stored in the raw zarr.
            from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset

            self._cache[episode_hash] = ZarrDataset(
                self.resolve_store(episode_hash),
                key_map=self.keymap,
                transform_list=self.transform_list,
            )
        return self._cache[episode_hash]

    def __call__(self, episode_hash: str, frame_idx: int) -> dict:
        ds = self._open(episode_hash)
        fi = int(np.clip(frame_idx, 0, len(ds) - 1))
        sample = ds[fi]  # post-transform dict; images are CHW float in [0,1]

        image = _to_numpy(sample[self.image_key])
        state = _to_numpy(sample[self.state_key]).astype(np.float32).reshape(-1)
        state = self._normalize(self.state_key, state)
        # actions_cartesian is the full (chunk, dim) target; keep only the first
        # ``action_horizon`` retrieved steps, normalize (per chunk position,
        # like the query path), THEN map to the shared 32-D space — to32 runs on
        # normalized values in the query pipeline too.
        action = _to_numpy(sample[self.action_key]).astype(np.float32)
        action = action[: self.action_horizon]
        action = self._normalize(self.action_key, action)
        if self.converter is not None:
            action = self.converter.to32(
                torch.as_tensor(action, dtype=torch.float32)
            )  # (H, dim) -> (H, 1, 32)
        action = _to_numpy(action).astype(np.float32)
        action = action.reshape(action.shape[0], -1)  # -> (H, 32)
        return {"image": image, "state": state, "action": action}


# ---------------------------------------------------------------------------
# RICL collate
# ---------------------------------------------------------------------------


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_chw(img) -> torch.Tensor:
    t = torch.as_tensor(_to_numpy(img))
    if t.ndim == 3 and t.shape[-1] in (1, 3):  # HWC -> CHW
        t = t.permute(2, 0, 1)
    return t.contiguous()


def build_ricl_collate(
    cache,
    bank_provider: BankFrameProvider,
    k: int,
    *,
    base_collate: Callable | None = None,
    image_hw=(224, 224),
    state_dim: int = 32,
    action_dim: int = 32,
    action_horizon: int = 1,
):
    """Return a collate_fn that attaches k retrieved blocks to each query sample.

    ``cache`` is a :class:`egomimic.ricl.retrieval.RetrievalCache`. ``bank_provider``
    loads a single bank frame. Output adds these keys to the collated batch:
    ``ricl_retrieved_images`` (B,k,C,H,W), ``ricl_retrieved_state`` (B,k,state_dim),
    ``ricl_retrieved_action`` (B,k,action_horizon,action_dim),
    ``ricl_retrieved_mask`` (B,k) bool, ``ricl_retrieved_dist`` (B,k) float.
    """
    if base_collate is None:
        from egomimic.pl_utils.pl_data_utils import annotation_collate

        base_collate = annotation_collate

    H, W = image_hw
    zero_img = torch.zeros(3, H, W, dtype=torch.float32)
    zero_state = torch.zeros(state_dim, dtype=torch.float32)
    zero_action = torch.zeros(action_horizon, action_dim, dtype=torch.float32)

    def _gather_one(query_hash, frame_idx):
        imgs = [zero_img.clone() for _ in range(k)]
        states = [zero_state.clone() for _ in range(k)]
        acts = [zero_action.clone() for _ in range(k)]
        mask = torch.zeros(k, dtype=torch.bool)
        dist = torch.full((k,), float("inf"), dtype=torch.float32)

        if query_hash in getattr(cache, "_entries", {}):
            bh, bf, dd = cache.neighbors(query_hash, frame_idx)
            for i in range(min(k, len(bh))):
                h = str(bh[i])
                if not h:  # padded slot
                    continue
                blk = bank_provider(h, int(bf[i]))
                img = _as_chw(blk["image"]).float()
                if img.max() > 1.5:  # uint8 -> [0,1] (PIRicl maps to [-1,1])
                    img = img / 255.0
                imgs[i] = _resize_chw(img, (H, W))
                st = torch.as_tensor(
                    _to_numpy(blk["state"]), dtype=torch.float32
                ).reshape(-1)
                states[i] = _fit(st, state_dim)
                ac = torch.as_tensor(_to_numpy(blk["action"]), dtype=torch.float32)
                acts[i] = _fit_action(ac, action_horizon, action_dim)
                mask[i] = True
                dist[i] = float(dd[i])
        return (
            torch.stack(imgs),
            torch.stack(states),
            torch.stack(acts),
            mask,
            dist,
        )

    def _collate(batch):
        imgs, states, acts, masks, dists = [], [], [], [], []
        for sample in batch:
            qh = sample.get("episode_hash")
            fi = int(sample.get("frame_idx", 0))
            ri, rs, ra, rm, rd = _gather_one(qh, fi)
            imgs.append(ri)
            states.append(rs)
            acts.append(ra)
            masks.append(rm)
            dists.append(rd)
        collated = base_collate(batch)
        collated["ricl_retrieved_images"] = torch.stack(imgs)
        collated["ricl_retrieved_state"] = torch.stack(states)
        collated["ricl_retrieved_action"] = torch.stack(acts)
        collated["ricl_retrieved_mask"] = torch.stack(masks)
        collated["ricl_retrieved_dist"] = torch.stack(dists)
        return collated

    return _collate


def _fit(v: torch.Tensor, dim: int) -> torch.Tensor:
    """Pad/truncate a 1-D tensor to length ``dim``."""
    if v.numel() == dim:
        return v
    out = torch.zeros(dim, dtype=torch.float32)
    n = min(dim, v.numel())
    out[:n] = v.reshape(-1)[:n]
    return out


def _fit_action(a: torch.Tensor, horizon: int, dim: int) -> torch.Tensor:
    if a.ndim == 1:
        a = a[None, :]
    h = min(horizon, a.shape[0])
    out = torch.zeros(horizon, dim, dtype=torch.float32)
    out[:h, : min(dim, a.shape[1])] = a[:h, : min(dim, a.shape[1])]
    return out


def _resize_chw(img: torch.Tensor, hw) -> torch.Tensor:
    if tuple(img.shape[-2:]) == tuple(hw):
        return img
    return torch.nn.functional.interpolate(
        img[None], size=tuple(hw), mode="bilinear", align_corners=False
    )[0]
