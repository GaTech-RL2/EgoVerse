"""Pure-Python glue for the RoboTwin closed-loop RICL eval adapter.

The closed-loop benchmark runs a trained EgoVerse PyTorch ``PIRicl`` checkpoint
inside RoboTwin's SAPIEN sim. This module holds the embodiment/observation/action
**glue** that is independent of both SAPIEN and the model, so it is directly
unit-testable:

- ``obs_to_state``        : RoboTwin obs dict -> ([head, right, left] rgb, qpos vector)
  (mirrors ``policy/pi05/deploy_policy.encode_obs``).
- ``quantile_norm`` / ``quantile_unnorm`` : the [-1,1] map used by the shim and its
  inverse (for turning a model action back into RoboTwin qpos).
- ``state_to_model_input`` : raw qpos -> normalized, slot-filled 32-D proprio.
- ``unnormalize_action``  : model action (32-D, normalized) -> RoboTwin qpos chunk.
- ``OnlineRetriever``     : embed the live head frame -> kNN against a prebuilt demo
  bank -> assemble ``ricl_retrieved_*`` blocks (the same keys ``build_ricl_collate``
  produces) for the model prefix.

The RoboTwin-side ``policy/pi_ricl_egoverse/{deploy_policy,pi_model}.py`` is a thin
wrapper that loads the checkpoint + a DINOv2 embedder and drives these primitives.
Quantiles come from the training corpus
(``robotwin_data.RoboTwinCorpus.save_quantiles`` / ``load_quantiles``); the action
space is RoboTwin's native qpos so the adapter returns a ``(steps, state_dim)`` chunk
executed with ``take_action(action)`` in the default ``qpos`` control mode.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from egomimic.ricl.robotwin_data import slot32

# RoboTwin obs camera -> RICL image key (head exterior is the retrieval/base view).
_RGB_ORDER = ("head_camera", "right_camera", "left_camera")  # matches pi05 encode_obs


def obs_to_state(observation: dict) -> tuple[list[np.ndarray], np.ndarray]:
    """RoboTwin ``get_obs()`` dict -> ``([head, right, left] rgb, qpos vector)``.

    Mirrors ``external/RoboTwin/policy/pi05/deploy_policy.encode_obs`` exactly so the
    same eval driver works."""
    obs = observation["observation"]
    rgb = [np.asarray(obs[cam]["rgb"]) for cam in _RGB_ORDER]
    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    return rgb, state


def quantile_norm(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Map raw qpos to [-1, 1] (same formula as ``RoboTwinCorpus.quantile_norm``)."""
    x = np.asarray(x, dtype=np.float32)
    return 2.0 * ((x - q01) / (q99 - q01 + 1e-6)) - 1.0


def quantile_unnorm(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quantile_norm`: [-1, 1] -> raw qpos units."""
    x = np.asarray(x, dtype=np.float32)
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def state_to_model_input(
    state: np.ndarray, q_state: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Raw qpos -> normalized, slot-filled 32-D proprio (the model's state input)."""
    q01, q99 = q_state
    return slot32(quantile_norm(state, q01, q99))


def unnormalize_action(
    pred: np.ndarray,
    q_actions: tuple[np.ndarray, np.ndarray],
    state_dim: int,
) -> np.ndarray:
    """Model action -> RoboTwin qpos chunk.

    ``pred`` is ``(H, 32)`` or ``(32,)`` normalized (the 32-D shared space, slot-filled);
    keep the first ``state_dim`` dims and invert the quantile normalization with the
    training corpus's action quantiles. Returns ``(H, state_dim)`` (or ``(state_dim,)``)."""
    pred = np.asarray(pred, dtype=np.float32)
    q01, q99 = q_actions
    sliced = pred[..., :state_dim]
    return quantile_unnorm(sliced, q01, q99)


def _to_chw01(img_hwc: np.ndarray, hw: tuple[int, int] = (224, 224)) -> torch.Tensor:
    """HWC uint8 (or float) -> CHW float in [0, 1], resized to ``hw`` (mirrors the
    collate's image handling so retrieved frames match the trained representation)."""
    t = torch.as_tensor(np.asarray(img_hwc))
    if t.ndim == 3 and t.shape[-1] in (1, 3):
        t = t.permute(2, 0, 1)
    t = t.float()
    if t.max() > 1.5:  # uint8 -> [0, 1]
        t = t / 255.0
    if tuple(t.shape[-2:]) != tuple(hw):
        t = torch.nn.functional.interpolate(
            t[None], size=tuple(hw), mode="bilinear", align_corners=False
        )[0]
    return t.contiguous()


class OnlineRetriever:
    """Build ``ricl_retrieved_*`` blocks for a live query frame at eval time.

    Embeds the query head frame, finds the k nearest demos in a prebuilt bank
    (``egomimic.ricl.retrieval.RetrievalIndex`` over the same 64-patch DINOv2
    descriptors used in training), and gathers each neighbor's (image, state, action)
    via ``bank_provider`` into the batch keys the model prefix consumes — the online
    analog of ``build_ricl_collate``. The embedder and index are injected so this is
    unit-testable with stubs (no DINOv2 needed).

    Args:
        index: a ``RetrievalIndex`` (``.query(vecs, k) -> (hashes, frames, dists)``).
        bank_provider: ``(episode_hash, frame_idx) -> {"image": HWC, "state": (D,),
            "action": (H, D)}`` (e.g. ``robotwin_data.make_robotwin_bank_provider``).
        embedder: object with ``.embed(images_thwc) -> (T, EMBED_DIM)`` (e.g.
            ``retrieval.DinoV2Embedder``); only called on the single query frame.
        k, action_horizon, state_dim: block dimensions; image_hw the resize target.
    """

    def __init__(
        self,
        index,
        bank_provider: Callable[[str, int], dict],
        embedder,
        *,
        k: int = 4,
        action_horizon: int = 15,
        state_dim: int = 14,
        image_hw: tuple[int, int] = (224, 224),
    ):
        self.index = index
        self.bank_provider = bank_provider
        self.embedder = embedder
        self.k = int(k)
        self.action_horizon = int(action_horizon)
        self.state_dim = int(state_dim)
        self.image_hw = image_hw

    def retrieve(self, query_image_hwc: np.ndarray) -> dict:
        """Return ``ricl_retrieved_*`` tensors (batch dim 1) for one query frame."""
        H, W = self.image_hw
        k, Ha, D = self.k, self.action_horizon, self.state_dim
        qvec = self.embedder.embed(np.asarray(query_image_hwc)[None])  # (1, EMBED_DIM)
        hashes, frames, dists = self.index.query(qvec, k=k)  # each (1, k)
        hashes, frames, dists = hashes[0], frames[0], dists[0]

        imgs = [torch.zeros(3, H, W) for _ in range(k)]
        states = [torch.zeros(D) for _ in range(k)]
        acts = [torch.zeros(Ha, D) for _ in range(k)]
        mask = torch.zeros(k, dtype=torch.bool)
        dist = torch.full((k,), float("inf"))
        for i in range(k):
            h = str(hashes[i])
            if not h:  # padded slot (bank smaller than k)
                continue
            blk = self.bank_provider(h, int(frames[i]))
            imgs[i] = _to_chw01(blk["image"], self.image_hw)
            st = torch.as_tensor(np.asarray(blk["state"]), dtype=torch.float32).reshape(
                -1
            )
            states[i] = _fit(st, D)
            ac = torch.as_tensor(np.asarray(blk["action"]), dtype=torch.float32)
            acts[i] = _fit_action(ac, Ha, D)
            mask[i] = True
            dist[i] = float(dists[i])
        return {
            "ricl_retrieved_images": torch.stack(imgs)[None],  # (1, k, 3, H, W)
            "ricl_retrieved_state": torch.stack(states)[None],  # (1, k, D)
            "ricl_retrieved_action": torch.stack(acts)[None],  # (1, k, Ha, D)
            "ricl_retrieved_mask": mask[None],  # (1, k)
            "ricl_retrieved_dist": dist[None],  # (1, k)
        }


def _fit(v: torch.Tensor, dim: int) -> torch.Tensor:
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
