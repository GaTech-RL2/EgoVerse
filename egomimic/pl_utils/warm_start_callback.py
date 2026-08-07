"""Warm-start: partial-load a donor checkpoint into named param GROUPS at fit
start. Groups not listed keep their fresh (random) init.

Used for the "transplant the aligned structure, train the compute fresh"
experiment: `load_groups=[structure]` copies the donor's router+chunker+apex,
leaving encoder+trunk+head at fresh init. Runs in `setup` (before
configure_optimizers), on every DDP rank identically.
"""
from __future__ import annotations

import os
import torch
from lightning.pytorch.callbacks import Callback

from egomimic.pl_utils.param_groups import resolve_param_groups


class WarmStartCallback(Callback):
    def __init__(self, donor_ckpt: str, param_groups: dict, load_groups: list):
        super().__init__()
        self.donor_ckpt = donor_ckpt
        self.specs = dict(param_groups)
        self.load_groups = list(load_groups)
        self._done = False

    def setup(self, trainer, pl_module, stage=None):
        if self._done or stage not in (None, "fit"):
            return
        # RESUME GUARD: on a requeue (fit(ckpt_path=...)), the resumed checkpoint
        # already holds the co-adapted structure — do NOT re-transplant the donor
        # over it. Only transplant on the fresh initial launch (no ckpt_path).
        if getattr(trainer, "ckpt_path", None) or os.environ.get("WARM_START_RESUME"):
            print(f"[warm-start] SKIP (resume from {trainer.ckpt_path}); "
                  f"keeping checkpoint weights, not re-transplanting donor")
            self._done = True
            return
        donor = torch.load(self.donor_ckpt, map_location="cpu", weights_only=False)
        dsd = donor.get("state_dict", donor)
        groups = resolve_param_groups(pl_module.named_parameters(), self.specs)
        want = set()
        for g in self.load_groups:
            want |= {n for n, _ in groups[g]}
        msd = pl_module.state_dict()
        partial, miss, mism = {}, [], []
        for n in want:
            if n not in dsd:
                miss.append(n)
            elif dsd[n].shape != msd[n].shape:
                mism.append(n)
            else:
                partial[n] = dsd[n]
        pl_module.load_state_dict(partial, strict=False)
        loaded_M = sum(v.numel() for v in partial.values()) / 1e6
        print(f"[warm-start] loaded {len(partial)}/{len(want)} params "
              f"({loaded_M:.1f}M) of groups {self.load_groups} from "
              f"{self.donor_ckpt}  (missing={len(miss)}, shape_mismatch={len(mism)})")
        if miss or mism:
            print(f"[warm-start]   e.g. missing={miss[:3]} mism={mism[:3]}")
        self._done = True
