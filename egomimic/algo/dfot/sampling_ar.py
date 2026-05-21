"""Causal-AR (staircase) and schedule-matrix samplers for DFoT inference.

The flexible scheduler API: a sampler is parameterized by a
``schedule_matrix`` of shape ``(n_steps + 1, T)`` where ``schedule_matrix[s, i]``
is the noise level of token ``i`` at denoising step ``s``. The sampler walks
``s = 0 -> n_steps``, running the backbone at each step with per-token noise
levels read directly from the matrix, then DDIM-stepping from
``schedule_matrix[s]`` to ``schedule_matrix[s + 1]``.

For continuous diffusion, schedule entries are floats in ``[0, 1]`` (the
``t`` parameter consumed by ``ContinuousDiffusion.schedule``). For discrete
diffusion, they are integer step indices in ``[0, timesteps)``.

Three named patterns are provided as schedule constructors; users can
build any other pattern by emitting a custom matrix.
"""

from typing import Optional, Union

import torch

from .continuous_diffusion import ContinuousDiffusion
from .discrete_diffusion import DiscreteDiffusion


# --------------------------------------------------------------------------- #
# Schedule constructors. All return a ``(n_steps + 1, T)`` matrix of noise
# levels in the convention of the diffusion variant being used (float [0, 1]
# for continuous, int [0, timesteps) for discrete).
# --------------------------------------------------------------------------- #


def vanilla_schedule(n_steps: int, T: int, *, discrete_timesteps: Optional[int] = None) -> torch.Tensor:
    """Standard uniform schedule (all tokens denoise together each step).

    Equivalent to DDIM with a global timestep. Useful as a baseline /
    sanity check that the matrix sampler reproduces existing behavior.
    """
    if discrete_timesteps is None:
        # Continuous: linearly anneal 1 -> 0 over n_steps + 1 points.
        levels = torch.linspace(1.0, 0.0, n_steps + 1)
    else:
        # Discrete: evenly pick n_steps + 1 timestep indices in (T-1, ..., 0).
        levels = torch.linspace(
            float(discrete_timesteps - 1), -1.0, n_steps + 1
        ).long().clamp_min(0)
    return levels[:, None].expand(n_steps + 1, T).contiguous()


def causal_ar_schedule(
    T: int,
    *,
    n_steps_per_token: int = 1,
    discrete_timesteps: Optional[int] = None,
) -> torch.Tensor:
    """Causal-AR staircase schedule.

    Token ``i`` is fully noisy at step ``i * n_steps_per_token`` and fully
    clean by step ``(i + 1) * n_steps_per_token``. The diagonal "rolls"
    one staircase step forward at each denoising step.

    At any given step, the model sees tokens at all noise levels
    simultaneously: token ``0`` is most-denoised, token ``T-1`` is least-
    denoised. This matches the DFoT-paper causal-AR sampler.

    Total steps = ``T * n_steps_per_token``. Output shape:
    ``(T * n_steps_per_token + 1, T)``.

    For ``n_steps_per_token == 1``: each step denoises each token by exactly
    one schedule unit, then a new fully-noisy token effectively rolls in.
    """
    n_total = T * n_steps_per_token
    # noise[s, i] = clamp(1 - (s - i * n_steps_per_token) / n_steps_per_token, 0, 1)
    # i.e. for token i, noise stays at 1 until step i*K, then linearly drops
    # to 0 at step (i+1)*K, then stays at 0.
    step_idx = torch.arange(n_total + 1).float()  # (n_total + 1,)
    tok_idx = torch.arange(T).float()  # (T,)
    raw = 1.0 - (step_idx[:, None] - tok_idx[None, :] * n_steps_per_token) / float(
        n_steps_per_token
    )
    levels = raw.clamp(0.0, 1.0)
    if discrete_timesteps is not None:
        # Map continuous [0, 1] -> discrete [0, timesteps).
        levels = (levels * (discrete_timesteps - 1)).round().long().clamp(
            0, discrete_timesteps - 1
        )
    return levels.contiguous()


def staircase_ar_schedule(
    T: int,
    *,
    chunk_size: int = 1,
    step_size: int = 1,
    discrete_timesteps: Optional[int] = None,
) -> torch.Tensor:
    """Configurable causal-AR staircase schedule.

    The two knobs control staircase geometry:
      - ``chunk_size`` (staircase "width"): how many tokens share the same
        noise level at any given step. 1 = one token per rung (vanilla
        causal AR). Larger = wider staircase rungs (more tokens denoise
        together).
      - ``step_size`` (staircase "height"): how many denoising steps each
        rung takes before sliding forward. 1 = each step advances every
        token by one schedule unit. Larger = slower denoise per rung.

    With ``chunk_size=1, step_size=1`` this reduces to ``causal_ar_schedule``.

    Noise level for token ``i`` at step ``s``:

        rung      = i // chunk_size                  # 0..n_chunks-1
        n_chunks  = ceil(T / chunk_size)
        level     = clamp(1 - (s - rung*step_size) / step_size, 0, 1)

    Total schedule length = ``n_chunks * step_size + 1``.
    """
    n_chunks = (T + chunk_size - 1) // chunk_size
    n_total = n_chunks * step_size
    step_idx = torch.arange(n_total + 1).float()  # (n_total + 1,)
    tok_idx = torch.arange(T).float()  # (T,)
    rung = (tok_idx // chunk_size)  # (T,) integer rung index
    raw = 1.0 - (step_idx[:, None] - rung[None, :] * step_size) / float(step_size)
    levels = raw.clamp(0.0, 1.0)
    if discrete_timesteps is not None:
        levels = (levels * (discrete_timesteps - 1)).round().long().clamp(
            0, discrete_timesteps - 1
        )
    return levels.contiguous()


def chunk_schedule(
    T: int,
    *,
    chunk_size: int = 8,
    n_steps_per_chunk: int = 10,
    discrete_timesteps: Optional[int] = None,
) -> torch.Tensor:
    """Chunked denoising: tokens in groups of ``chunk_size`` denoise together.

    Chunks denoise sequentially — chunk 0 first, then chunk 1, etc. Within a
    chunk, all tokens share the same noise level at each step.

    Total steps = ``ceil(T / chunk_size) * n_steps_per_chunk``.
    """
    n_chunks = (T + chunk_size - 1) // chunk_size
    n_total = n_chunks * n_steps_per_chunk
    step_idx = torch.arange(n_total + 1).float()
    tok_idx = torch.arange(T).float()
    chunk_of = (tok_idx // chunk_size)  # (T,)
    raw = 1.0 - (step_idx[:, None] - chunk_of[None, :] * n_steps_per_chunk) / float(
        n_steps_per_chunk
    )
    levels = raw.clamp(0.0, 1.0)
    if discrete_timesteps is not None:
        levels = (levels * (discrete_timesteps - 1)).round().long().clamp(
            0, discrete_timesteps - 1
        )
    return levels.contiguous()


# --------------------------------------------------------------------------- #
# Matrix sampler. Backbone agnostic to schedule shape — does one step at a
# time using whatever noise levels the matrix dictates per token.
# --------------------------------------------------------------------------- #


@torch.no_grad()
def matrix_sample(
    diffusion: Union[ContinuousDiffusion, DiscreteDiffusion],
    backbone,
    *,
    schedule_matrix: torch.Tensor,  # (n_steps + 1, T) — see module docstring
    action_dim: int,
    batch_size: int = 1,
    external_cond: Optional[torch.Tensor] = None,
    eta: float = 0.0,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Denoise an action chunk under an arbitrary per-token-per-step schedule.

    Args:
        diffusion: ``ContinuousDiffusion`` or ``DiscreteDiffusion``.
        backbone: callable ``(x, noise_levels, external_cond) -> pred``. The
            ``noise_levels`` shape matches the matrix row at each step
            (continuous floats or discrete longs).
        schedule_matrix: ``(n_steps + 1, T)``. Row 0 is the initial noise
            level for each token (typically all 1.0 / max); row -1 is the
            final (typically 0). Floats for continuous, longs for discrete.
        action_dim: feature dim of the action.
        batch_size: batch dim for the output. The schedule itself is broadcast
            across the batch.
        external_cond: optional obs cond, ``(B, T, cond_dim)`` or
            ``(B, cond_dim)``.
        eta: DDIM stochasticity (0.0 = deterministic). Used by both variants.

    Returns:
        ``(B, T, action_dim)`` denoised actions.
    """
    n_steps_plus_one, T = schedule_matrix.shape
    n_steps = n_steps_plus_one - 1
    device = device or (
        external_cond.device if external_cond is not None else schedule_matrix.device
    )
    schedule_matrix = schedule_matrix.to(device)
    B = batch_size

    # Initialize x at the level dictated by row 0 of the schedule. For tokens
    # that start at full noise this is pure Gaussian; for tokens that start
    # clean (level=0) we still seed with noise — the first denoising step at
    # level=0 will recover them via the diffusion math (in practice these
    # tokens should be provided externally for true AR, but for v1 we treat
    # the schedule as authoritative and let the model see noise at level=0).
    x = torch.randn(B, T, action_dim, device=device, dtype=dtype)

    if isinstance(diffusion, ContinuousDiffusion):
        for s in range(n_steps):
            t_cur = schedule_matrix[s].to(dtype).unsqueeze(0).expand(B, T)
            t_next = schedule_matrix[s + 1].to(dtype).unsqueeze(0).expand(B, T)
            v = diffusion.model_v(backbone, x, t_cur, external_cond)
            x0, eps = diffusion.v_to_x0_and_noise(x, t_cur, v)
            logsnr_next = diffusion.schedule(t_next)
            alpha_next = torch.sigmoid(logsnr_next).sqrt().unsqueeze(-1)
            sigma_next = torch.sigmoid(-logsnr_next).sqrt().unsqueeze(-1)
            if eta > 0:
                noise = torch.randn_like(x)
                x = alpha_next * x0 + sigma_next * (
                    (1 - eta ** 2).clamp_min(0.0).sqrt() * eps + eta * noise
                )
            else:
                x = alpha_next * x0 + sigma_next * eps
        return x

    if isinstance(diffusion, DiscreteDiffusion):
        for s in range(n_steps):
            k_cur = schedule_matrix[s].long().unsqueeze(0).expand(B, T)
            k_next = schedule_matrix[s + 1].long().unsqueeze(0).expand(B, T)
            pred = diffusion.model_predictions(backbone, x, k_cur, external_cond)
            x0 = pred.pred_x_start
            eps = pred.pred_noise
            alpha = diffusion.alphas_cumprod[k_cur].unsqueeze(-1)
            # k_next can be < 0 to mean "fully denoised". Clamp to 0 and
            # alpha_next = 1 in that case.
            k_next_clamp = k_next.clamp_min(0)
            alpha_next = diffusion.alphas_cumprod[k_next_clamp].unsqueeze(-1)
            fully_clean = (k_next < 0).unsqueeze(-1)
            alpha_next = torch.where(fully_clean, torch.ones_like(alpha_next), alpha_next)
            sigma_sq = (
                eta ** 2
                * (1 - alpha / alpha_next).clamp_min(0.0)
                * (1 - alpha_next).clamp_min(0.0)
                / (1 - alpha).clamp_min(1e-8)
            )
            sigma = sigma_sq.clamp_min(0.0).sqrt()
            c = (1 - alpha_next - sigma ** 2).clamp_min(0.0).sqrt()
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            x = x0 * alpha_next.sqrt() + eps * c + sigma * noise
        return x

    raise TypeError(f"unsupported diffusion type {type(diffusion).__name__}")


# --------------------------------------------------------------------------- #
# Online AR rollout helper for inference_step. Maintains a buffer of in-flight
# tokens at staircase noise levels; one call advances every token by one
# schedule unit, pops the front (now clean) token, and pushes a new fully-
# noisy token at the back. Designed to be called once per env step.
# --------------------------------------------------------------------------- #


class CausalARRollout:
    """Online causal-AR sampler. One call per env step.

    Maintains a token buffer of length ``buffer_size``. On each ``step()``:
      1. The model sees the buffer with per-token noise levels
         ``[0, 1, 2, ..., buffer_size - 1]`` (in DFoT schedule units of
         ``1 / n_steps_per_token`` for continuous or ``1`` for discrete).
      2. Each token advances by one schedule unit.
      3. The front token (now at level 0) is committed and returned.
      4. A new fully-noisy token (level = buffer_size - 1) is pushed at the
         back of the buffer.

    Args:
        diffusion: continuous or discrete diffusion module.
        backbone: ``(x, noise_levels, cond) -> pred`` callable. For closed-
            loop AR the cond is the *current* per-step obs encoding (the
            caller supplies it fresh each ``step()``).
        action_dim: action feature dim.
        buffer_size: number of in-flight tokens. For DFoT-paper causal-AR
            this is the "k" in ``[0, 1, ..., k]`` noise levels — i.e. each
            token takes ``k`` env steps from full noise to clean.
        device: torch device for buffer state.
        dtype: float dtype.
    """

    def __init__(
        self,
        diffusion: Union[ContinuousDiffusion, DiscreteDiffusion],
        backbone,
        *,
        action_dim: int,
        buffer_size: int,
        device,
        dtype: torch.dtype = torch.float32,
    ):
        self.diffusion = diffusion
        self.backbone = backbone
        self.action_dim = int(action_dim)
        self.buffer_size = int(buffer_size)
        self.device = device
        self.dtype = dtype
        if isinstance(diffusion, ContinuousDiffusion):
            # Continuous schedule units of (1 / buffer_size). Token i sits at
            # level (i + 1) / buffer_size, descending by 1 / buffer_size per step.
            self._unit = 1.0 / float(self.buffer_size)
            self._discrete = False
        elif isinstance(diffusion, DiscreteDiffusion):
            # Discrete: each token spans 1 timestep of the diffusion schedule.
            # Buffer length = K means tokens at integer levels [1, 2, ..., K].
            self._unit = max(1, diffusion.timesteps // self.buffer_size)
            self._discrete = True
        else:
            raise TypeError(f"unsupported diffusion type {type(diffusion).__name__}")

        # Buffer of in-flight tokens, all initialized to fully-noisy.
        self.x = torch.randn(
            1, self.buffer_size, self.action_dim, device=device, dtype=dtype
        )

    def _current_levels(self) -> torch.Tensor:
        """Per-token noise levels for the current buffer state.

        Token at index ``i`` (0-based from front) holds level ``i + 1`` in
        unit steps. The front-most token's level is ``self._unit`` (one step
        away from clean); the back-most token's level is ``buffer_size *
        self._unit`` (~1.0 continuous / ``timesteps`` discrete).
        """
        idx = torch.arange(self.buffer_size, device=self.device) + 1
        if self._discrete:
            levels = (idx * self._unit).long().clamp(
                1, self.diffusion.timesteps - 1
            )
        else:
            levels = (idx.float() * self._unit).clamp(0.0, 1.0)
        return levels.unsqueeze(0)  # (1, buffer_size)

    def _next_levels(self) -> torch.Tensor:
        """Per-token noise levels after advancing one schedule unit. Front
        token reaches 0 (clean); a new fully-noisy slot will appear at the
        back AFTER popping the front. Here we just return the post-advance
        levels for the EXISTING buffer (front -> 0, others -> level - unit)."""
        idx = torch.arange(self.buffer_size, device=self.device)  # (buffer_size,)
        if self._discrete:
            levels = (idx * self._unit).long().clamp(
                0, self.diffusion.timesteps - 1
            )
        else:
            levels = (idx.float() * self._unit).clamp(0.0, 1.0)
        return levels.unsqueeze(0)  # (1, buffer_size)

    @torch.no_grad()
    def step(self, external_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Advance the schedule by one unit, commit the front token, push a
        fresh noisy token at the back. Returns the committed action of shape
        ``(action_dim,)``.

        Args:
            external_cond: obs cond for THIS env step, shape ``(1, cond_dim)``
                or ``(1, buffer_size, cond_dim)``. For canonical causal-AR
                the cond should be broadcast across all buffer tokens (the
                model conditions every in-flight token on the latest obs).
        """
        cur = self._current_levels()  # (1, buffer_size)
        nxt = self._next_levels()  # (1, buffer_size)

        if isinstance(self.diffusion, ContinuousDiffusion):
            v = self.diffusion.model_v(self.backbone, self.x, cur, external_cond)
            x0, eps = self.diffusion.v_to_x0_and_noise(self.x, cur, v)
            logsnr_next = self.diffusion.schedule(nxt)
            alpha_next = torch.sigmoid(logsnr_next).sqrt().unsqueeze(-1)
            sigma_next = torch.sigmoid(-logsnr_next).sqrt().unsqueeze(-1)
            self.x = alpha_next * x0 + sigma_next * eps
        else:  # DiscreteDiffusion
            pred = self.diffusion.model_predictions(self.backbone, self.x, cur, external_cond)
            x0 = pred.pred_x_start
            eps = pred.pred_noise
            alpha = self.diffusion.alphas_cumprod[cur].unsqueeze(-1)
            nxt_clamp = nxt.clamp_min(0)
            alpha_next = self.diffusion.alphas_cumprod[nxt_clamp].unsqueeze(-1)
            c = (1 - alpha_next).clamp_min(0.0).sqrt()
            self.x = x0 * alpha_next.sqrt() + eps * c

        # Pop the front (committed clean) and push a fresh fully-noisy token.
        front = self.x[:, 0, :].clone()  # (1, action_dim)
        new_back = torch.randn(
            1, 1, self.action_dim, device=self.device, dtype=self.dtype
        )
        self.x = torch.cat([self.x[:, 1:, :], new_back], dim=1)
        return front.squeeze(0)  # (action_dim,)
