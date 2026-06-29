"""Episode-level batch transforms applied during training in PackedAlgoBase.

Unlike the per-frame ``transform_list`` (dataset ``_read_span``, length-preserving)
and ``train_obs_transforms`` (post-norm, length-preserving), these operate on ONE
episode at a time and MAY change its length. They run PRE-normalization at the top
of ``PackedAlgoBase.process_batch_for_training`` so cross-key raw values (e.g. the
agent position in ``observations.state``) are in the same world-coord space as
``actions``.
"""

from __future__ import annotations

import torch

_EPT_LOGGED = False

_PACKED_META = (
    "cu_seqlens",
    "max_seq_len",
    "seq_lens",
    "batch_size",
    "embodiment",
    "episode_idx",
    "chunk_offset",
    "_packed",
)


class EpisodeLevelTransform:
    """Base: operate on one episode's packed tensors (each ``(L, ...)``, time
    leading). May change ``L``. Return the transformed episode dict."""

    def __call__(self, episode: dict) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


def apply_episode_level_transforms(batch: dict, transforms: list) -> dict:
    """Split a packed batch by ``cu_seqlens``, apply each transform per episode
    (length may change), re-pack, and update ``cu_seqlens`` / ``seq_lens`` /
    ``max_seq_len``. Meta + non-time-series keys are preserved. Returns a new dict.
    """
    if not transforms or "cu_seqlens" not in batch:
        return batch
    cu = batch["cu_seqlens"]
    cu_list = [int(x) for x in cu.tolist()]
    B = len(cu_list) - 1
    T_total = cu_list[-1]
    seq_keys = [
        k
        for k, v in batch.items()
        if torch.is_tensor(v)
        and k not in _PACKED_META
        and v.dim() >= 1
        and v.shape[0] == T_total
    ]
    if not seq_keys:
        return batch
    eps = []
    for b in range(B):
        s, e = cu_list[b], cu_list[b + 1]
        ep = {k: batch[k][s:e] for k in seq_keys}
        for t in transforms:
            ep = t(ep)
        eps.append(ep)
    out = dict(batch)
    new_lens = [int(ep[seq_keys[0]].shape[0]) for ep in eps]
    new_cu = [0]
    for L in new_lens:
        new_cu.append(new_cu[-1] + L)
    global _EPT_LOGGED
    if not _EPT_LOGGED:
        print(
            "[EpisodeLevelTransform] fired %s: T_total %d -> %d over %d episodes"
            % ([type(t).__name__ for t in transforms], T_total, new_cu[-1], B)
        )
        _EPT_LOGGED = True
    out["cu_seqlens"] = torch.tensor(new_cu, device=cu.device, dtype=cu.dtype)
    for k in seq_keys:
        out[k] = torch.cat([ep[k] for ep in eps], dim=0)
    if "seq_lens" in batch and torch.is_tensor(batch["seq_lens"]):
        out["seq_lens"] = torch.tensor(
            new_lens, device=batch["seq_lens"].device, dtype=batch["seq_lens"].dtype
        )
    if "max_seq_len" in batch:
        mv, old = max(new_lens), batch["max_seq_len"]
        out["max_seq_len"] = (
            torch.tensor(mv, device=old.device, dtype=old.dtype)
            if torch.is_tensor(old)
            else type(old)(mv)
        )
    return out


class PadHoldStill(EpisodeLevelTransform):
    """Append ``n_pad`` hold-still frames to each episode so the policy learns to
    STAY STILL after it solves the task. obs / state / image repeat their final
    frame; ``actions`` are held at the **actual agent position**
    (``observations.state[:agent_dim][-1]``) — NOT the last action/cursor (which
    is ``pusher_cmd_pose`` == ``actions``). Pre-norm: action and state[:2] share
    the world-coord (0-512) space.
    """

    def __init__(
        self,
        n_pad: int = 50,
        action_key: str = "actions",
        state_key: str = "observations.state",
        agent_dim: int = 2,
    ):
        self.n_pad = int(n_pad)
        self.action_key = action_key
        self.state_key = state_key
        self.agent_dim = int(agent_dim)

    def __call__(self, ep: dict) -> dict:
        if self.n_pad <= 0:
            return ep
        hold = None
        if self.state_key in ep and torch.is_tensor(ep[self.state_key]):
            hold = ep[self.state_key][-1, : self.agent_dim]
        out = {}
        for k, v in ep.items():
            if not torch.is_tensor(v) or v.dim() == 0:
                out[k] = v
                continue
            if k == self.action_key and hold is not None:
                step = v[-1].clone()
                d = min(step.shape[-1], hold.shape[-1])
                step[..., :d] = hold[:d].to(step.dtype)
                pad = step.unsqueeze(0).repeat(self.n_pad, *([1] * step.dim()))
            else:
                pad = v[-1:].repeat(self.n_pad, *([1] * (v.dim() - 1)))
            out[k] = torch.cat([v, pad], dim=0)
        return out
