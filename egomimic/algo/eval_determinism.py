"""Deterministic evaluation shared by the flow-matching algos (HPT, QwenVLA).

Every RNG used while an algo is in eval mode is seeded from
``EVAL_BASE_SEED + rank * EVAL_RANK_STRIDE + pass_counter``, where the pass
counter counts ``forward_eval`` calls since the last
``reset_eval_pass_counter()`` (called from ``ModelWrapper.on_validation_start``).
Two validation passes over the same batches therefore replay the same prompts
and the same sampling noise, while different ranks and different batches
within a pass still see different draws.

The mixin reads ``self.nets`` (an ``nn.ModuleDict``, for ``.training``),
``self.device``, ``self.annotation_key``, ``self.annotation_sampling_mode``
and ``self.default_prompt`` from the algo it is mixed into.
"""

from __future__ import annotations

import os
import random

import torch

EVAL_BASE_SEED = 0
EVAL_RANK_STRIDE = 1_000_003


class DeterministicEvalMixin:
    annotation_key: str | None = None
    annotation_sampling_mode: str = "random"
    default_prompt: str = ""

    def reset_eval_pass_counter(self) -> None:
        """Start a new validation pass: the next ``forward_eval`` call is pass 0.

        Called from ``ModelWrapper.on_validation_start``. Two passes over the
        same loader (same order, same number of batches) then draw the same
        prompts and the same sampling noise.
        """
        self._eval_pass_counter = 0

    @staticmethod
    def _eval_rank() -> int:
        """Distributed rank, or 0 outside a process group."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        for var in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
            value = os.environ.get(var)
            if value is not None and value.lstrip("-").isdigit():
                return int(value)
        return 0

    def _eval_pass_seed(self) -> int:
        """Seed for the current eval pass: base + rank stride + pass counter."""
        return (
            int(getattr(self, "eval_base_seed", EVAL_BASE_SEED))
            + self._eval_rank() * EVAL_RANK_STRIDE
            + int(getattr(self, "_eval_pass_counter", 0))
        )

    def _eval_device(self) -> torch.device:
        return (
            torch.device(self.device)
            if self.device is not None
            else torch.device("cpu")
        )

    def _eval_generator(self) -> torch.Generator:
        """Seeded generator on the device the sampling noise is drawn on."""
        device = self._eval_device()
        generator = torch.Generator(device=device)
        generator.manual_seed(self._eval_pass_seed())
        return generator

    def _eval_prompt_rng(self) -> random.Random:
        """Seeded Python RNG for annotation choice during eval."""
        return random.Random(self._eval_pass_seed())

    def _eval_fork_devices(self) -> list:
        """Devices ``torch.random.fork_rng`` must save/restore (CPU is implicit)."""
        device = self._eval_device()
        if device.type != "cuda":
            return []
        return [
            device.index if device.index is not None else torch.cuda.current_device()
        ]

    def _build_prompts(self, _batch, batch_size: int) -> list[str]:
        """Sample one annotation per batch item, falling back to default_prompt
        on empty / missing annotations. Mirrors the Pi algo flow.

        In eval the choice comes from a per-pass seeded RNG regardless of
        ``annotation_sampling_mode``: that keeps prompt variety (unlike always
        taking ``sample[0]``) while making two validation passes over the same
        batches produce identical prompt lists. Training is unchanged.
        """
        if self.annotation_key is None or self.annotation_key not in _batch:
            return [self.default_prompt] * batch_size
        eval_rng = None if self.nets.training else self._eval_prompt_rng()
        prompts = []
        for sample in _batch[self.annotation_key]:
            if not sample:
                prompts.append(self.default_prompt)
            elif eval_rng is not None:
                prompts.append(sample[eval_rng.randint(0, len(sample) - 1)])
            elif self.annotation_sampling_mode == "random":
                prompts.append(sample[random.randint(0, len(sample) - 1)])
            else:  # "first"
                prompts.append(sample[0])
        return prompts
