"""Real-Time Chunking (RTC) serving wrapper — Black/Galliker/Levine,
"Real-Time Execution of Action Chunking Flow Policies" (Physical Intelligence).

Inference-time only, no retraining, and NO edits to the existing stack:
``RTCEgoVersePolicy`` subclasses ``EgoVersePolicy`` and wraps the FM head's
``inference`` bound method at construction. A request without ``rtc_*`` keys
takes the ORIGINAL code path bit-for-bit; a request with them generates the
next chunk as an inpainting problem against the chunk currently executing:

  * rows [0, d)  — FROZEN: they are guaranteed to execute while this inference
    runs, so they must equal the previous chunk (hard constraint);
  * rows [d, E)  — SOFT: pulled toward the previous chunk with weight
    w_j = ((E-j)/(E-d))**gamma, decaying 1 -> ~0 (consistency vs reactivity);
  * rows [E, 64) — FREE: replanned from the new observation.

Flow matching makes the frozen rows EXACT: with the straight path
x_t = t*eps + (1-t)*Y, blending x_t toward that path each Euler step keeps the
constrained rows on the trajectory whose endpoint is Y (we also hard-set them
after the loop, so they are bit-equal after unnormalize round-off ~1e-7).

Client contract (extra obs keys, popped before normal processing):
  rtc_prev_actions  (n<=64, 49) float — REMAINING rows of the executing chunk,
                    raw action space, time-aligned so new row i covers the same
                    control tick as prev row i;
  rtc_freeze_steps  int d   — >= ceil(inference latency / control dt);
  rtc_soft_horizon  int E   — soft-consistency end row (default d+16);
  rtc_soft_power    float   — decay exponent gamma (default 1 = linear).
"""
from __future__ import annotations

import logging
import types

import numpy as np
import torch

from egomimic.models.fm_policy import FMPolicy
from egomimic.serving.egoverse_policy import EgoVersePolicy

RTC_OBS_KEYS = ("rtc_prev_actions", "rtc_freeze_steps",
                "rtc_soft_horizon", "rtc_soft_power")


def _rtc_inference(head, noise, global_cond, generator=None):
    """Replaces the FM head's Euler loop ONLY while a ctx is armed."""
    ctx = getattr(head, "_rtc_ctx", None)
    if ctx is None:
        return FMPolicy.inference(head, noise, global_cond, generator)
    Y = ctx["target"].to(device=noise.device, dtype=noise.dtype)     # (1,H,D) normalized
    w = ctx["weights"].to(device=noise.device, dtype=noise.dtype)    # (1,H,1)
    d = int(ctx["freeze"])
    # identical schedule to FMPolicy.inference; eps0 = the initial noise keeps
    # the constrained rows exactly on the straight path t*eps0 + (1-t)*Y.
    head.dt = -1.0 / head.num_inference_steps
    eps0 = noise
    x_t = noise
    time = torch.ones((len(global_cond)), device=global_cond.device)
    while time[0] >= -head.dt / 2:
        t = time[0]
        x_known = t * eps0 + (1.0 - t) * Y
        x_t = w * x_known + (1.0 - w) * x_t
        x_t, time = head.step(x_t, time, global_cond)
    if d > 0:
        x_t = x_t.clone()
        x_t[:, :d] = Y[:, :d]
    return x_t


class RTCEgoVersePolicy(EgoVersePolicy):
    """EgoVersePolicy + optional per-request RTC inpainting (see module doc)."""

    def __init__(self, model_wrapper, device: str | None = None):
        super().__init__(model_wrapper, device)
        heads = self._model.model.nets["policy"].heads
        head = heads[self._embodiment_name]
        if type(head) is not FMPolicy:
            raise TypeError(
                f"RTC supports plain FMPolicy heads only; {self._embodiment_name} "
                f"head is {type(head).__name__} (Hierarchical/DDPM: not wired).")
        head._rtc_ctx = None
        head.inference = types.MethodType(_rtc_inference, head)
        self._rtc_head = head
        logging.info("[rtc] wrapped '%s' FMPolicy.inference "
                     "(requests without rtc_* keys take the original path)",
                     self._embodiment_name)

    def infer(self, obs: dict) -> dict:
        obs = dict(obs)
        prev = obs.pop("rtc_prev_actions", None)
        d = int(obs.pop("rtc_freeze_steps", 0) or 0)
        E = obs.pop("rtc_soft_horizon", None)
        gamma = float(obs.pop("rtc_soft_power", 1.0) or 1.0)
        if prev is None:
            return super().infer(obs)
        if int(obs.pop("num_samples", 1) or 1) > 1:
            raise ValueError("RTC does not support num_samples > 1")

        H, D = int(self._action_horizon), int(self._action_dim)
        prev = np.asarray(prev, dtype=np.float32).reshape(-1, D)
        n = min(len(prev), H)
        if n == 0:
            raise ValueError("rtc_prev_actions is empty")
        Y = np.zeros((H, D), dtype=np.float32)
        Y[:n] = prev[:n]
        d = max(0, min(d, n))
        E = int(E) if E is not None else d + 16
        E = max(d, min(E, n))
        w = np.zeros(H, dtype=np.float32)
        w[:d] = 1.0
        if E > d:
            j = np.arange(d, E, dtype=np.float32)
            w[d:E] = ((E - j) / float(E - d)) ** gamma

        Yt = torch.from_numpy(Y).float().unsqueeze(0)
        Yn = self._data_schematic.normalize_data(
            {self._ac_key: Yt}, self._embodiment_id)[self._ac_key]
        self._rtc_head._rtc_ctx = {
            "target": Yn,
            "weights": torch.from_numpy(w).reshape(1, H, 1),
            "freeze": d,
        }
        try:
            out = super().infer(obs)
        finally:
            self._rtc_head._rtc_ctx = None
        out["rtc"] = {"freeze": d, "soft_end": E, "tail_rows": n, "gamma": gamma}
        return out
