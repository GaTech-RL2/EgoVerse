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

from typing import Callable, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

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


# ---------------------------------------------------------------------------
# Bank frame provider
# ---------------------------------------------------------------------------


class BankFrameProvider(Protocol):
    """Loads one bank frame's retrieved block, normalized to the query convention."""

    def __call__(self, episode_hash: str, frame_idx: int) -> dict:
        """Return ``{"image": (C,H,W) or (H,W,C), "state": (Ds,), "action": (Ha,Da)}``."""
        ...


class ZarrBankFrameProvider:
    """Load (image, state, action) for a bank frame from its zarr store (lazy zarr).

    Applies the embodiment's action converter ``to32`` so retrieved actions live
    in the same 32-D space as eva queries (aria's missing gripper -> slot 0; see
    ``HumanBimanualCartesianEuler``). Cluster-side; unit tests use a mock provider.
    """

    def __init__(
        self,
        resolve_store: Callable[[str], str],
        converter=None,
        image_zarr_key: str = "images.front_1",
        state_zarr_key: str = "observations.state.ee_pose",
        action_zarr_key: str = "actions_cartesian",
        action_horizon: int = 1,
    ):
        self.resolve_store = resolve_store
        self.converter = converter
        self.image_zarr_key = image_zarr_key
        self.state_zarr_key = state_zarr_key
        self.action_zarr_key = action_zarr_key
        self.action_horizon = action_horizon
        self._cache: dict = {}

    def _open(self, episode_hash: str):
        if episode_hash not in self._cache:
            import zarr  # lazy

            self._cache[episode_hash] = zarr.open(
                self.resolve_store(episode_hash), mode="r"
            )
        return self._cache[episode_hash]

    def __call__(self, episode_hash: str, frame_idx: int) -> dict:
        import simplejpeg  # lazy

        store = self._open(episode_hash)
        T = store[self.state_zarr_key].shape[0]
        fi = int(np.clip(frame_idx, 0, T - 1))

        jpeg = store[self.image_zarr_key][fi]
        image = simplejpeg.decode_jpeg(jpeg, colorspace="RGB")  # (H,W,3) uint8
        state = np.asarray(store[self.state_zarr_key][fi], dtype=np.float32)

        a0, a1 = fi, min(fi + self.action_horizon, T)
        action = np.asarray(store[self.action_zarr_key][a0:a1], dtype=np.float32)
        if self.converter is not None:
            action = _to_numpy(self.converter.to32(torch.from_numpy(action))).astype(
                np.float32
            )
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
