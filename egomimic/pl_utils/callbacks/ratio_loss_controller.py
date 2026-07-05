"""Closed-loop feedback controller for the ChunkerStage ratio-loss weight.

Motivation: the ratio loss regularizes each chunker's boundary rate (br)
toward the setpoint ``1 / target_compression_ratio`` (N=3 -> 0.333). The task
gradient strengthens over training (GMM sigma shrinks), so a FIXED
``ratio_loss_weight`` lets br creep upward. This callback holds br at the
setpoint by adapting each stage's weight with a multiplicative ratchet +
deadband (the adaptive-KL-penalty pattern): every ``every_n_epochs`` epochs,
per controlled stage,

    if |br_mean - setpoint| > tol:  w <- min(w * rho,   w_max)   # ratchet up
    else:                           w <- max(w * decay, w_min)   # relax

Metric convention (discovered from ``algo/hnet/algo.py`` + ``pl_model.py``):
``HNetAlgo.forward_training`` stores per-chunker stats as
``{emb_id}_boundary_rate_{i}`` (``emb_id`` = numeric ``get_embodiment_id`` of
the embodiment/domain NAME), ``compute_losses`` renames to
``emb{emb_id}_boundary_rate_{i}``, and ``ModelWrapper.training_step`` logs
``"Train/" + k`` with ``on_epoch=True, sync_dist=True`` — so the epoch-mean is
available in ``trainer.callback_metrics`` at ``on_train_epoch_end`` as e.g.
``Train/emb15_boundary_rate_0``. The per-level index ``i`` is the position in
``ctx.aux``; ``ChunkerStage`` calls ``ctx.register_aux`` AFTER its
``inner_stage`` forward returns, so aux is registered inner-first:
**index 0 = the INNERMOST chunker**, and a chunker's index equals the number
of chunkers reachable strictly below it via the ``inner_stage`` chain. This
callback reproduces that mapping by walking ``.inner_stage`` pointers
(descending into ``PerEmbodimentStage.sub_stages[emb]`` where needed).

Stage <-> metric association:
  * A ``ChunkerStage`` found inside ``PerEmbodimentStage.sub_stages[name]`` is
    controlled from its own embodiment's metric
    ``{metric_prefix}emb{get_embodiment_id(name)}_boundary_rate_{level}``.
  * A SHARED ``ChunkerStage`` (a plain chain element) appears in EVERY
    embodiment's metrics at its level; its br_mean is the mean over all
    ``{metric_prefix}emb(\\d+)_boundary_rate_{level}`` entries present in
    ``trainer.callback_metrics``. Its weight is logged under
    ``train/ratio_w_shared_l{level}``.

Coexistence with ``RatioLossScheduler`` (which writes ``ratio_loss_weight``
from an epoch schedule at ``on_train_epoch_start``): Lightning runs callbacks
in REGISTRATION ORDER within each hook, so ordering-based fixes are fragile.
This controller instead uses the robust REASSERT design: it keeps its own
per-stage weight state ``self._w`` and re-writes ``stage.ratio_loss_weight``
at EVERY ``on_train_epoch_end`` — overwriting whatever the scheduler wrote
that epoch, regardless of registration order. When enabling the controller,
set the scheduler's ``on_value`` / ``off_value`` EQUAL to the controller's
starting weight (the controller seeds ``self._w`` from each stage's current
``ratio_loss_weight`` at its first epoch end) so the scheduler's writes are
no-ops; or drop the scheduler entirely.

DDP: the boundary-rate metrics are logged with ``sync_dist=True`` (see
``pl_model.py``), so ``callback_metrics`` epoch means are already identical
across ranks and every rank computes the same decision. As a hard guarantee
(and to stay correct if a metric is ever logged unsynced), rank 0's decided
weights are additionally broadcast to all ranks via
``torch.distributed.broadcast`` whenever a process group is initialized, and
only then applied. Prints are rank-zero only; the weight WRITE happens on all
ranks (it must — the loss is computed on every rank).

Checkpoint resume: the controller persists ``self._w`` through the Lightning
callback ``state_dict`` / ``load_state_dict`` mechanism.

Enable via Hydra (callbacks compose as a dict)::

    +callbacks.ratio_ctrl._target_=egomimic.pl_utils.callbacks.ratio_loss_controller.RatioLossController \\
    +callbacks.ratio_ctrl.setpoint=null \\
    +callbacks.ratio_ctrl.tol=0.03 \\
    +callbacks.ratio_ctrl.rho=1.2 \\
    +callbacks.ratio_ctrl.decay=0.995 \\
    +callbacks.ratio_ctrl.every_n_epochs=10 \\
    +callbacks.ratio_ctrl.w_min=1.0 \\
    +callbacks.ratio_ctrl.w_max=50.0 \\
    +callbacks.ratio_ctrl.metric_prefix=Train/
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import lightning.pytorch as pl
import torch

from egomimic.models.hnet.per_embodiment_stage import PerEmbodimentStage
from egomimic.models.hnet.stages import ChunkerStage

try:  # canonical embodiment-name -> numeric-id mapping (same import the algo uses)
    from egomimic.rldb.embodiment.embodiment import get_embodiment_id
except Exception:  # pragma: no cover - allows use in stripped-down envs/tests
    get_embodiment_id = None


class _Controlled:
    """One controlled ChunkerStage: identity + how to read its br metric."""

    __slots__ = ("key", "stage", "emb_num", "level", "shared")

    def __init__(
        self, key: str, stage, emb_num: Optional[int], level: int, shared: bool
    ):
        self.key = key  # e.g. "emb15_l1" or "shared_l0" (stable, sorted for DDP order)
        self.stage = stage  # the ChunkerStage module
        self.emb_num = emb_num  # numeric embodiment id (None for shared stages)
        self.level = level  # aux index in the metric key (0 = innermost chunker)
        self.shared = shared


class RatioLossController(pl.Callback):
    def __init__(
        self,
        setpoint: Optional[float] = None,
        tol: float = 0.03,
        rho: float = 1.2,
        decay: float = 0.995,
        every_n_epochs: int = 10,
        w_min: float = 1.0,
        w_max: float = 50.0,
        metric_prefix: str = "Train/",
    ):
        super().__init__()
        # None -> derive per-stage as 1 / stage.target_compression_ratio.
        self.setpoint = None if setpoint is None else float(setpoint)
        self.tol = float(tol)
        self.rho = float(rho)
        self.decay = float(decay)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.metric_prefix = str(metric_prefix)

        self._records: List[_Controlled] = []
        self._w: Dict[
            str, float
        ] = {}  # controller-owned weight state (reasserted every epoch)
        self._warned: set = (
            set()
        )  # warn-once registry for missing metrics / unmapped embs

    # ------------------------------------------------------------------ #
    # Discovery: associate each ChunkerStage with (embodiment, level).
    # ------------------------------------------------------------------ #

    @staticmethod
    def _count_inner_chunkers(stage, emb: Optional[str]) -> int:
        """Number of ChunkerStages strictly below ``stage`` on the
        ``inner_stage`` chain == its ``boundary_rate_{i}`` index (aux is
        registered inner-first; see module docstring)."""
        n = 0
        node = getattr(stage, "inner_stage", None)
        seen: set = set()
        while node is not None and id(node) not in seen:
            seen.add(id(node))
            if isinstance(node, PerEmbodimentStage):
                subs = node.sub_stages
                sub = (
                    subs[emb]
                    if (emb is not None and emb in subs)
                    else next(iter(subs.values()))
                )
                if isinstance(sub, ChunkerStage):
                    n += 1
                node = getattr(sub, "inner_stage", None)
                continue
            if isinstance(node, ChunkerStage):
                n += 1
            node = getattr(node, "inner_stage", None)
        return n

    def _emb_number(self, name: str) -> Optional[int]:
        """Embodiment NAME (PerEmbodimentStage.sub_stages key, e.g.
        'pushshapes_sim') -> the NUMBER in the metric key (e.g. 15)."""
        if get_embodiment_id is not None:
            try:
                return int(get_embodiment_id(name))
            except Exception:
                pass
        # Fallbacks for trees keyed by id-like strings ("15" / "emb15").
        m = re.fullmatch(r"(?:emb)?(\d+)", str(name))
        if m:
            return int(m.group(1))
        return None

    def _discover(self, trainer, pl_module) -> List[_Controlled]:
        per_emb: List[tuple] = []  # (emb_name, chunker)
        for m in pl_module.modules():
            if isinstance(m, PerEmbodimentStage):
                for name, sub in m.sub_stages.items():
                    if isinstance(sub, ChunkerStage):
                        per_emb.append((str(name), sub))
        per_emb_ids = {id(c) for _, c in per_emb}

        records: List[_Controlled] = []
        for name, c in per_emb:
            level = self._count_inner_chunkers(c, emb=name)
            emb_num = self._emb_number(name)
            if emb_num is None:
                self._warn_once(
                    trainer,
                    f"emb-name:{name}",
                    f"cannot map embodiment name {name!r} to a metric number; "
                    f"stage at level {level} will be skipped.",
                )
                continue
            records.append(
                _Controlled(f"emb{emb_num}_l{level}", c, emb_num, level, shared=False)
            )
        for m in pl_module.modules():
            if isinstance(m, ChunkerStage) and id(m) not in per_emb_ids:
                level = self._count_inner_chunkers(m, emb=None)
                records.append(
                    _Controlled(f"shared_l{level}", m, None, level, shared=True)
                )

        records.sort(
            key=lambda r: r.key
        )  # deterministic order (DDP broadcast relies on it)
        return records

    # ------------------------------------------------------------------ #
    # Metric reading
    # ------------------------------------------------------------------ #

    def _read_br(self, trainer, rec: _Controlled) -> Optional[float]:
        """Epoch-mean boundary rate for a controlled stage, from
        ``trainer.callback_metrics``. None when unavailable this epoch."""
        cm = trainer.callback_metrics
        if not rec.shared:
            v = cm.get(
                f"{self.metric_prefix}emb{rec.emb_num}_boundary_rate_{rec.level}"
            )
            return None if v is None else float(v)
        # Shared stage: mean over every embodiment's metric at this level.
        pat = re.compile(
            re.escape(self.metric_prefix)
            + r"emb(\d+)_boundary_rate_"
            + str(rec.level)
            + r"$"
        )
        vals = [float(v) for k, v in cm.items() if pat.fullmatch(k)]
        return (sum(vals) / len(vals)) if vals else None

    def _warn_once(self, trainer, token: str, msg: str) -> None:
        if token in self._warned:
            return
        self._warned.add(token)
        if getattr(trainer, "is_global_zero", True):
            print(f"[RatioLossController] WARNING: {msg}", flush=True)

    # ------------------------------------------------------------------ #
    # DDP determinism: broadcast rank 0's decided weights.
    # ------------------------------------------------------------------ #

    def _sync_weights(self, trainer, pl_module, records: List[_Controlled]) -> None:
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return
        if int(getattr(trainer, "world_size", 1)) <= 1:
            return
        device = getattr(pl_module, "device", torch.device("cpu"))
        t = torch.tensor(
            [self._w[r.key] for r in records], dtype=torch.float64, device=device
        )
        torch.distributed.broadcast(t, src=0)  # overwrites on non-src ranks
        for r, v in zip(records, t.tolist()):
            self._w[r.key] = float(v)

    # ------------------------------------------------------------------ #
    # Lightning hook
    # ------------------------------------------------------------------ #

    def on_train_epoch_end(self, trainer, pl_module):
        if not self._records:
            self._records = self._discover(trainer, pl_module)
            if self._records and getattr(trainer, "is_global_zero", True):
                desc = ", ".join(
                    f"{r.key}(N={getattr(r.stage, 'target_compression_ratio', '?')})"
                    for r in self._records
                )
                print(
                    f"[RatioLossController] controlling {len(self._records)} ChunkerStage(s): {desc}",
                    flush=True,
                )
        records = self._records
        if not records:
            return
        # Seed controller state from the CURRENT stage weight (the controller's
        # starting weight; equal to the scheduler's on/off value when composed).
        for r in records:
            if r.key not in self._w:
                self._w[r.key] = float(r.stage.ratio_loss_weight)

        errs: Dict[str, float] = {}
        is_update = (int(trainer.current_epoch) + 1) % self.every_n_epochs == 0
        if is_update:
            for r in records:
                br = self._read_br(trainer, r)
                if br is None:
                    self._warn_once(
                        trainer,
                        r.key,
                        f"metric "
                        f"{self.metric_prefix}emb{r.emb_num if not r.shared else '*'}_boundary_rate_{r.level} "
                        f"missing from trainer.callback_metrics; skipping {r.key} this cycle.",
                    )
                    continue
                sp = (
                    self.setpoint
                    if self.setpoint is not None
                    else 1.0 / max(float(r.stage.target_compression_ratio), 1e-8)
                )
                err = br - sp
                errs[r.key] = err
                w = self._w[r.key]
                if abs(err) > self.tol:
                    w = min(w * self.rho, self.w_max)
                else:
                    w = max(w * self.decay, self.w_min)
                self._w[r.key] = w
            # Guarantee rank-identical weights before applying (see docstring).
            self._sync_weights(trainer, pl_module, records)
            if getattr(trainer, "is_global_zero", True):
                parts = [
                    f"{r.key}: br_err={errs[r.key]:+.4f} -> w={self._w[r.key]:.4g}"
                    for r in records
                    if r.key in errs
                ]
                if parts:
                    print(
                        f"[RatioLossController] epoch {int(trainer.current_epoch)}: "
                        + "; ".join(parts),
                        flush=True,
                    )

        # REASSERT the controller-owned weight on EVERY epoch end — this
        # overwrites anything RatioLossScheduler wrote this epoch, making the
        # composition robust to callback registration order.
        for r in records:
            r.stage.ratio_loss_weight = float(self._w[r.key])

        for r in records:
            pl_module.log(
                f"train/ratio_w_{r.key}",
                float(self._w[r.key]),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            if r.key in errs:
                pl_module.log(
                    f"train/ratio_br_err_{r.key}",
                    float(errs[r.key]),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

    # ------------------------------------------------------------------ #
    # Checkpoint state (so resumed runs keep the ratcheted weights)
    # ------------------------------------------------------------------ #

    def state_dict(self):
        return {"w": dict(self._w), "warned": sorted(self._warned)}

    def load_state_dict(self, state_dict):
        self._w = dict(state_dict.get("w", {}))
        self._warned = set(state_dict.get("warned", []))
