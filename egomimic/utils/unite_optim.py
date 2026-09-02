"""Released UNITE Muon+AdamW grouping and two-stage schedule.

The released UNITE implementation at commit
``2389f2c65fedbd43ada851e69b2b810efd23af7d`` delegates its matrix optimizer
to ``torch.optim.Muon``. EgoVerse is pinned to Torch 2.7.1, which predates that
class, so this module vendors the single-tensor algorithm from PyTorch 2.10.0
commit ``449b1768410104d3ed79d3bcfe4ba1d65c7f22c0``:

https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/torch/optim/_muon.py

Only imports and the class name were adapted. Newton--Schulz coefficients,
BF16 orthogonalization, Nesterov momentum, decoupled weight decay, and
``match_rms_adamw`` scaling retain the upstream implementation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, MutableMapping

import torch
from torch import Tensor

_PYTORCH_MUON_COMMIT = "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"
_MUON_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
_MUON_NS_STEPS = 5

# The first four entries are the released ImageNet exceptions. The remaining
# names are their explicit robot-policy counterparts. Canonical VisualCore
# parameters live below ``...encoder.img_encoders.<camera>...``; matching only
# ``visual_encoder`` misses every real path.
_ADAMW_NAME_MARKERS = (
    "patch_embed",
    "final_layer",
    "in_context_posemb",
    "latent_tokens",
    "img_encoders",
    "visual_encoder",
    "visual_token_projection",
    "proprio_token_projection",
    "action_context_projections",
    "condition_projection",
    "null_observation_condition",
    "token_identity",
    "pos_emb",
    "proj_d",
    "action_projection",
)


def _zeropower_via_newton_schulz(
    gradient: Tensor,
    coefficients: tuple[float, float, float] = _MUON_COEFFICIENTS,
    steps: int = _MUON_NS_STEPS,
    eps: float = 1.0e-7,
) -> Tensor:
    """PyTorch 2.10's BF16 quintic Newton--Schulz orthogonalization."""

    if steps >= 100:
        raise ValueError("Number of steps must be less than 100")
    if gradient.ndim != 2:
        raise ValueError("Muon gradient must be a 2D matrix")
    if len(coefficients) != 3:
        raise ValueError("Muon coefficients must contain exactly three values")
    a, b, c = coefficients
    orthogonal = gradient.bfloat16()
    transposed = gradient.shape[0] > gradient.shape[1]
    if transposed:
        orthogonal = orthogonal.T
    orthogonal.div_(orthogonal.norm().clamp(min=eps))
    for _ in range(steps):
        gram = orthogonal @ orthogonal.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        orthogonal = torch.addmm(orthogonal, gram_update, orthogonal, beta=a)
    return orthogonal.T if transposed else orthogonal


def _adjust_muon_lr(
    lr: float, adjust_lr_fn: str | None, parameter_shape: torch.Size
) -> float:
    rows, columns = parameter_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        ratio = math.sqrt(max(1.0, rows / columns))
    elif adjust_lr_fn == "match_rms_adamw":
        ratio = 0.2 * math.sqrt(max(rows, columns))
    else:
        ratio = 1.0
    return float(lr) * ratio


class VendoredTorchMuon(torch.optim.Optimizer):
    """Torch 2.10 Muon backport for the repository's pinned Torch 2.7.1."""

    source_commit = _PYTORCH_MUON_COMMIT

    def __init__(
        self,
        params,
        lr: float = 1.0e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = _MUON_COEFFICIENTS,
        eps: float = 1.0e-7,
        ns_steps: int = _MUON_NS_STEPS,
        adjust_lr_fn: str | None = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Learning rate must be non-negative, got {lr}")
        if momentum < 0.0:
            raise ValueError(f"Momentum must be non-negative, got {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Weight decay must be non-negative, got {weight_decay}")
        if adjust_lr_fn not in {None, "original", "match_rms_adamw"}:
            raise ValueError(f"Unsupported Muon LR adjustment {adjust_lr_fn!r}")
        defaults = {
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "momentum": float(momentum),
            "nesterov": bool(nesterov),
            "ns_coefficients": tuple(float(value) for value in ns_coefficients),
            "eps": float(eps),
            "ns_steps": int(ns_steps),
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError(
                        "Muon supports only 2D parameters; got "
                        f"shape {tuple(parameter.shape)}"
                    )

    def _init_group(
        self, group: MutableMapping
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        parameters = []
        gradients = []
        momentum_buffers = []
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            if torch.is_complex(parameter):
                raise RuntimeError("Muon does not support complex parameters")
            if parameter.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")
            parameters.append(parameter)
            gradients.append(parameter.grad)
            state = self.state[parameter]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    parameter.grad, memory_format=torch.preserve_format
                )
            momentum_buffers.append(state["momentum_buffer"])
        return parameters, gradients, momentum_buffers

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            parameters, gradients, momentum_buffers = self._init_group(group)
            for parameter, gradient, buffer in zip(
                parameters, gradients, momentum_buffers
            ):
                momentum = group["momentum"]
                buffer.lerp_(gradient, 1.0 - momentum)
                update = (
                    gradient.lerp(buffer, momentum) if group["nesterov"] else buffer
                )
                update = _zeropower_via_newton_schulz(
                    update,
                    coefficients=group["ns_coefficients"],
                    steps=group["ns_steps"],
                    eps=group["eps"],
                )
                lr = group["lr"]
                parameter.mul_(1.0 - lr * group["weight_decay"])
                parameter.add_(
                    update,
                    alpha=-_adjust_muon_lr(lr, group["adjust_lr_fn"], parameter.shape),
                )
        return loss


def partition_released_unite_parameters(
    named_params: Iterable[tuple[str, torch.nn.Parameter]],
) -> tuple[
    tuple[tuple[str, torch.nn.Parameter], ...],
    tuple[tuple[str, torch.nn.Parameter], ...],
]:
    """Return complete, disjoint ``(AdamW, Muon)`` named parameter groups."""

    trainable = tuple(
        (str(name), parameter)
        for name, parameter in named_params
        if parameter.requires_grad
    )
    if not trainable:
        raise ValueError("Released UNITE optimizer received no trainable parameters")
    identities = [id(parameter) for _, parameter in trainable]
    if len(set(identities)) != len(identities):
        raise RuntimeError("Released UNITE named parameters contain aliases")

    adamw = []
    muon = []
    for name, parameter in trainable:
        special = any(marker in name for marker in _ADAMW_NAME_MARKERS)
        # PyTorch's released Muon accepts exactly 2D matrices. VisualCore is
        # special by name even for its 2D projection; conv kernels and every
        # other non-matrix tensor stay in AdamW by dimensionality.
        destination = muon if parameter.ndim == 2 and not special else adamw
        destination.append((name, parameter))
    if not adamw or not muon:
        raise RuntimeError("Released UNITE requires non-empty AdamW and Muon sets")
    return tuple(adamw), tuple(muon)


class ReleasedUniteCompositeOptimizer(torch.optim.Optimizer):
    """One Lightning-compatible facade over released AdamW and Muon groups."""

    def __init__(
        self,
        named_params: Iterable[tuple[str, torch.nn.Parameter]],
        lr: float = 1.0e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-6,
        adamw_weight_decay: float = 0.0,
        muon_weight_decay: float = 0.0,
        muon_momentum: float = 0.95,
        muon_adjust_lr_fn: str = "match_rms_adamw",
    ):
        if float(muon_momentum) != 0.95:
            raise ValueError("Released UNITE requires Muon momentum=0.95")
        if muon_adjust_lr_fn != "match_rms_adamw":
            raise ValueError("Released UNITE requires match_rms_adamw")
        adamw_named, muon_named = partition_released_unite_parameters(named_params)
        trainable = (*adamw_named, *muon_named)
        super().__init__(
            (parameter for _, parameter in trainable),
            defaults={"lr": float(lr)},
        )

        adamw_parameters = [parameter for _, parameter in adamw_named]
        self.adamw = torch.optim.AdamW(
            adamw_parameters,
            lr=float(lr),
            betas=tuple(float(value) for value in betas),
            eps=float(eps),
            weight_decay=float(adamw_weight_decay),
            fused=all(parameter.is_cuda for parameter in adamw_parameters),
        )
        self.muon = VendoredTorchMuon(
            [parameter for _, parameter in muon_named],
            lr=float(lr),
            momentum=float(muon_momentum),
            weight_decay=float(muon_weight_decay),
            eps=float(eps),
            adjust_lr_fn=muon_adjust_lr_fn,
        )
        self.param_groups = self.adamw.param_groups + self.muon.param_groups
        self.group_manifest = {
            "adamw_parameter_names": tuple(name for name, _ in adamw_named),
            "muon_parameter_names": tuple(name for name, _ in muon_named),
            "muon_source_commit": _PYTORCH_MUON_COMMIT,
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.adamw.step()
        self.muon.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.adamw.zero_grad(set_to_none=set_to_none)
        self.muon.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "adamw": self.adamw.state_dict(),
            "muon": self.muon.state_dict(),
            "group_manifest": self.group_manifest,
        }

    def load_state_dict(self, state_dict):
        if state_dict.get("group_manifest") != self.group_manifest:
            raise RuntimeError("Released UNITE optimizer parameter manifest changed")
        self.adamw.load_state_dict(state_dict["adamw"])
        self.muon.load_state_dict(state_dict["muon"])
        self.param_groups = self.adamw.param_groups + self.muon.param_groups


def released_unite_two_stage_scheduler(
    optimizer,
    warmup_steps: int,
    decay_start_1_steps: int,
    decay_end_1_steps: int,
    decay_start_2_steps: int,
    decay_end_2_steps: int,
    base_lr_1: float,
    base_lr_2: float,
    final_lr: float,
):
    """Step-domain form of the released two-stage cosine schedule."""

    warmup = int(warmup_steps)
    start1 = int(decay_start_1_steps)
    end1 = max(int(decay_end_1_steps), start1)
    start2 = max(int(decay_start_2_steps), end1)
    end2 = max(int(decay_end_2_steps), start2)
    base1 = float(base_lr_1)
    base2 = float(base_lr_2)
    final = float(final_lr)
    if min(warmup, start1, end1, start2, end2) < 0 or base1 <= 0.0:
        raise ValueError("Invalid released UNITE scheduler contract")

    def cosine(step: int, begin: int, end: int, high: float, low: float) -> float:
        if end <= begin:
            return low
        progress = min(max((step - begin) / (end - begin), 0.0), 1.0)
        return low + (high - low) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def multiplier(step: int) -> float:
        if warmup > 0 and step < warmup:
            value = base1 * step / warmup
        elif step < start1:
            value = base1
        elif step < end1:
            value = cosine(step, start1, end1, base1, base2)
        elif step < start2:
            value = base2
        elif step < end2:
            value = cosine(step, start2, end2, base2, final)
        else:
            value = final
        return value / base1

    for group in optimizer.param_groups:
        group["lr"] = base1
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)
