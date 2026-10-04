"""Masked endpoint guidance in LeRobot's native noise=1, action=0 convention.

For s = 1 - t, the proposal's forward velocity is -v_native. Consequently
A_hat = x - t*v_native and v_guided = v_native + lambda(t)*grad_x E.
Euler's negative dt turns the added term into descent on endpoint error.
This changes action samples, never parameters. It does not invert a target.
"""

import math
import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GuidanceConfig:
    steps: int = 10
    strength: float = 1.0
    max_gradient_norm: float = 5.0
    schedule: str = "taper_to_zero"
    project_gradient: bool = True
    max_guidance_weight: float = 100.0

    def __post_init__(self):
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("Positive integer solver steps required")
        if not math.isfinite(self.strength) or not 0 <= self.strength <= 10:
            raise ValueError("Guidance strength must be in [0,10]")
        if not math.isfinite(self.max_gradient_norm) or self.max_gradient_norm <= 0:
            raise ValueError("Positive finite gradient bound required")
        if self.schedule not in ("taper_to_zero", "rtc_pigdm"):
            raise ValueError("Unknown guidance schedule")
        if type(self.project_gradient) is not bool:
            raise ValueError("Gradient projection must be boolean")
        if (
            not math.isfinite(self.max_guidance_weight)
            or not 0 < self.max_guidance_weight <= 100
        ):
            raise ValueError("Guidance coefficient cap must be in (0,100]")

    def coefficient(self, t):
        if not math.isfinite(t) or not 0 <= t <= 1:
            raise ValueError("Native flow time must be in [0,1]")
        if self.schedule == "taper_to_zero":
            return self.strength * t
        # RTC equations 2--4, tau=1-t: ((1-t)^2+t^2)/(t*(1-t)).
        # The limiting coefficient is capped at either endpoint.
        if self.strength == 0:
            return 0.0
        ratio = math.inf if t in (0, 1) else ((1 - t) ** 2 + t**2) / (t * (1 - t))
        return min(self.max_guidance_weight, self.strength * ratio)


def _validate(x, target, mask):
    if x.ndim != 3 or min(x.shape) < 1 or not x.is_floating_point():
        raise ValueError("Expected floating [B,H,D] actions")
    if target.shape != x.shape or mask.shape != x.shape:
        raise ValueError("Target/mask must cover the exact native action shape")
    if target.device != x.device or mask.device != x.device:
        raise ValueError("Target, mask and actions must share a device")
    if any(not torch.isfinite(v).all() for v in (x, target, mask)):
        raise ValueError("Finite guidance inputs required")
    if torch.any((mask < 0) | (mask > 1)):
        raise ValueError("Mask weights must be in [0,1]")


def endpoint_gradient(velocity, x, t, target, mask):
    """Full VJP through the denoiser; M appears twice in ||M(Ahat-A*)||²."""
    _validate(x, target, mask)
    if not math.isfinite(t) or not 0 <= t <= 1:
        raise ValueError("Native flow time must be in [0,1]")
    with torch.enable_grad():
        leaf = x.detach().requires_grad_(True)
        v = velocity(leaf, t)
        if v.shape != x.shape or not torch.isfinite(v).all():
            raise ValueError("Invalid native velocity")
        estimate = leaf - t * v
        energy = 0.5 * ((estimate - target.detach()) * mask.detach()).square().sum()
        gradient = torch.autograd.grad(energy, leaf, create_graph=False)[0]
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Non-finite guidance gradient")
    return v.detach(), gradient.detach(), float(energy.detach())


def generate(velocity, noise, target, mask, config=GuidanceConfig()):
    """Generate from the supplied *independent* noise, without action inversion.

    The original taper/projection is retained for reproducibility. The optional
    RTC schedule uses the paper's clipped coefficient in native time; disabling
    the extra projection preserves the full VJP. The mask always gates endpoint
    error. Off-mask latent changes can help correct a coupled output channel.
    """
    _validate(noise, target, mask)
    clock = time.perf_counter()
    x = noise.detach().clone()
    trace = []
    enabled = config.strength > 0 and bool(torch.any(mask != 0))
    for i in range(config.steps):
        t = 1.0 - i / config.steps
        if enabled:
            v, gradient, energy = endpoint_gradient(velocity, x, t, target, mask)
            if config.project_gradient:
                gradient *= mask != 0
            norm = gradient.flatten(1).norm(dim=1).reshape(-1, 1, 1)
            scale = (config.max_gradient_norm / norm.clamp_min(1e-12)).clamp(max=1)
            coefficient = config.coefficient(t)
            guidance = coefficient * gradient * scale
            trace.append(
                {
                    "t": t,
                    "energy": energy,
                    "gradient_norm": norm.flatten().tolist(),
                    "clipped": bool(torch.any(scale < 1)),
                    "guidance_coefficient": coefficient,
                }
            )
        else:
            with torch.no_grad():
                v = velocity(x, t)
            if v.shape != x.shape:
                raise ValueError("Native velocity changed shape")
            guidance = 0
        x = (x + (-1.0 / config.steps) * (v + guidance)).detach()
        if not torch.isfinite(x).all():
            raise FloatingPointError("Non-finite guided sample")
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    return x, {
        "operator": "masked_endpoint_gradient",
        "sampling_direction": "1_to_0",
        "schedule": config.schedule,
        "project_gradient": config.project_gradient,
        "max_guidance_weight": config.max_guidance_weight,
        "strength": config.strength,
        "steps": config.steps,
        "velocity_evaluations": config.steps,
        "gradient_evaluations": config.steps if enabled else 0,
        "latency_seconds": time.perf_counter() - clock,
        "trace": trace,
    }
