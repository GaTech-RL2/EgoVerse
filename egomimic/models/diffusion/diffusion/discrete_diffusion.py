"""
Discrete-time DDPM for action-chunk denoising (per-token noise levels).

Slimmed from the DFoT video implementation: drops video-specific shape
machinery (channels/HxW), DDIM logic is provided separately in
``sampling.py``. ``forward(x, k, external_cond)`` runs the standard q-sample
+ model-prediction + weighted-MSE loss path with per-token ``k`` of shape
``(B, T)`` and action tensor ``x`` of shape ``(B, T, action_dim)``.
"""

import warnings
from collections import namedtuple
from typing import Literal, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .noise_schedule import make_beta_schedule


ModelPrediction = namedtuple(
    "ModelPrediction", ["pred_noise", "pred_x_start", "model_out"]
)

# Once-per-process guard for the fused_min_snr 1-D-k fallback warning.
_FUSED_MIN_SNR_1D_K_WARNED = False


def _extract(a: torch.Tensor, k: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Gather schedule values ``a[k]`` and reshape to broadcast over ``x``.

    ``a`` is (timesteps,); ``k`` is (B,) or (B, T) (per-token). ``x_shape``
    is the action tensor shape, e.g. ``(B, T, action_dim)``.
    """
    shape = k.shape
    out = a[k]
    return out.reshape(*shape, *((1,) * (len(x_shape) - len(shape))))


class DiscreteDiffusion(nn.Module):
    """Per-token discrete-time DDPM training+loss core.

    Args:
        action_dim: dim of the trailing feature axis on ``x``.
        timesteps: number of discrete noise levels.
        beta_schedule: 'cosine' | 'linear' | 'alphas_cumprod_linear'.
        schedule_fn_kwargs: extra kwargs to ``make_beta_schedule``.
        objective: 'pred_noise' | 'pred_x0' | 'pred_v'.
        loss_weighting_strategy: 'uniform' | 'min_snr' | 'sigmoid' | 'constant'.
        snr_clip: SNR clip used by 'min_snr' weighting.
        sigmoid_bias: bias used by 'sigmoid' weighting.
        clip_noise: clip ε samples to ``[-clip_noise, clip_noise]``.
        constant_weight: weight returned at every ``k`` by the 'constant'
            strategy (no objective-dependent factor; upstream parity = 2.5).
        max_beta: when set, clamp betas to ``<= max_beta`` before deriving
            alphas (upstream parity = 0.999, keeps terminal SNR > 0);
            ``None`` keeps the schedule untouched.
    """

    def __init__(
        self,
        action_dim: int,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        schedule_fn_kwargs: Optional[dict] = None,
        objective: Literal["pred_noise", "pred_x0", "pred_v"] = "pred_v",
        loss_weighting_strategy: Literal["uniform", "min_snr", "fused_min_snr", "sigmoid", "constant"] = "min_snr",
        snr_clip: float = 5.0,
        cum_snr_decay: float = 0.96,
        use_causal_mask: bool = False,
        sigmoid_bias: float = -1.0,
        clip_noise: float = 20.0,
        constant_weight: float = 1.0,
        max_beta: Optional[float] = None,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.timesteps = int(timesteps)
        self.objective = objective
        self.loss_weighting_strategy = loss_weighting_strategy
        self.snr_clip = float(snr_clip)
        self.cum_snr_decay = float(cum_snr_decay)
        self.use_causal_mask = bool(use_causal_mask)
        self.sigmoid_bias = float(sigmoid_bias)
        self.clip_noise = float(clip_noise)
        self.constant_weight = float(constant_weight)
        self.max_beta = None if max_beta is None else float(max_beta)

        betas = make_beta_schedule(
            schedule=beta_schedule,
            timesteps=self.timesteps,
            zero_terminal_snr=self.objective != "pred_noise",
            **(schedule_fn_kwargs or {}),
        )
        if self.max_beta is not None:
            betas = betas.clamp(max=self.max_beta)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        def reg(name: str, val: torch.Tensor):
            self.register_buffer(name, val.to(torch.float32), persistent=False)

        reg("betas", betas)
        reg("alphas_cumprod", alphas_cumprod)
        reg("alphas_cumprod_prev", alphas_cumprod_prev)
        reg("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        reg("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        reg("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        reg("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        reg("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        reg("posterior_variance", posterior_variance)
        reg(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        reg(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        reg(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        snr = alphas_cumprod / (1 - alphas_cumprod)
        reg("snr", snr)
        # Registered unconditionally (persistent=False) so per-call strategy
        # overrides work regardless of the ctor strategy.
        reg("clipped_snr", snr.clone().clamp_(max=self.snr_clip))
        reg("logsnr", torch.log(snr))

    # ---- closed-form predictions ---- #

    def predict_start_from_noise(self, x_k, k, noise):
        return (
            _extract(self.sqrt_recip_alphas_cumprod, k, x_k.shape) * x_k
            - _extract(self.sqrt_recipm1_alphas_cumprod, k, x_k.shape) * noise
        )

    def predict_noise_from_start(self, x_k, k, x0):
        return (
            x_k - _extract(self.sqrt_alphas_cumprod, k, x_k.shape) * x0
        ) / _extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape)

    def predict_v(self, x_start, k, noise):
        return (
            _extract(self.sqrt_alphas_cumprod, k, x_start.shape) * noise
            - _extract(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_k, k, v):
        return (
            _extract(self.sqrt_alphas_cumprod, k, x_k.shape) * x_k
            - _extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape) * v
        )

    def predict_noise_from_v(self, x_k, k, v):
        return (
            _extract(self.sqrt_alphas_cumprod, k, x_k.shape) * v
            + _extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape) * x_k
        )

    def q_sample(self, x_start, k, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start).clamp_(-self.clip_noise, self.clip_noise)
        return (
            _extract(self.sqrt_alphas_cumprod, k, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * noise
        )

    def q_posterior(self, x_start, x_k, k):
        mean = (
            _extract(self.posterior_mean_coef1, k, x_k.shape) * x_start
            + _extract(self.posterior_mean_coef2, k, x_k.shape) * x_k
        )
        var = _extract(self.posterior_variance, k, x_k.shape)
        log_var = _extract(self.posterior_log_variance_clipped, k, x_k.shape)
        return mean, var, log_var

    # ---- model wrapper ---- #

    def model_predictions(
        self,
        backbone,
        x: torch.Tensor,
        k: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
    ) -> ModelPrediction:
        model_out = backbone(x, k, external_cond)
        if self.objective == "pred_noise":
            pred_noise = torch.clamp(model_out, -self.clip_noise, self.clip_noise)
            x_start = self.predict_start_from_noise(x, k, pred_noise)
        elif self.objective == "pred_x0":
            x_start = model_out
            pred_noise = self.predict_noise_from_start(x, k, x_start)
        elif self.objective == "pred_v":
            v = model_out
            x_start = self.predict_start_from_v(x, k, v)
            pred_noise = self.predict_noise_from_v(x, k, v)
        else:
            raise ValueError(f"unknown objective {self.objective}")
        return ModelPrediction(pred_noise, x_start, model_out)

    def compute_loss_weights(
        self, k: torch.Tensor, strategy: Optional[str] = None
    ) -> torch.Tensor:
        """Per-token loss weights at noise levels ``k``.

        Args:
            k: (B,) or (B, T) long tensor of noise levels.
            strategy: optional per-call override of the ctor
                ``loss_weighting_strategy`` (e.g. per modality group);
                ``None`` uses the ctor strategy on the identical code path.
                'fused_min_snr' is rejected as an override — it couples
                weights across the time axis and must never be applied
                per-modality.
        """
        if strategy == "fused_min_snr":
            raise ValueError(
                "fused_min_snr cannot be requested as a per-call override; "
                "it couples weights across the time axis"
            )
        if strategy is None:
            strategy = self.loss_weighting_strategy
        if strategy == "bivariate_fused":
            raise ValueError(
                "bivariate_fused needs two noise-level streams; call "
                "compute_bivariate_fused_weights(k_obs, k_act) directly"
            )
        if strategy == "uniform":
            return torch.ones_like(k, dtype=torch.float32)
        if strategy == "constant":
            # No objective-dependent factor — mirrors the upstream constant
            # weight (diffusion_transition.py: snr_clip * 0.5 at every t).
            return torch.full_like(k, self.constant_weight, dtype=torch.float32)
        snr = self.snr[k]
        if strategy == "min_snr":
            clipped_snr = self.clipped_snr[k]
            epsilon_weighting = clipped_snr / snr.clamp(min=1e-8)
        elif strategy == "sigmoid":
            logsnr = self.logsnr[k]
            epsilon_weighting = torch.sigmoid(self.sigmoid_bias - logsnr)
        elif strategy == "fused_min_snr":
            if k.dim() == 1:
                global _FUSED_MIN_SNR_1D_K_WARNED
                if not _FUSED_MIN_SNR_1D_K_WARNED:
                    _FUSED_MIN_SNR_1D_K_WARNED = True
                    warnings.warn(
                        "loss_weighting_strategy='fused_min_snr' received 1-D k "
                        "(packed/flat batch): falling back to plain min_snr — "
                        "cum_snr_decay is inactive.",
                        stacklevel=2,
                    )
                clipped_snr = self.clipped_snr[k]
                epsilon_weighting = clipped_snr / snr.clamp(min=1e-8)
            else:
                snr_clip = self.snr_clip
                cum_snr_decay = self.cum_snr_decay
                clipped_snr = self.clipped_snr[k]
                normalized_clipped_snr = clipped_snr / snr_clip
                normalized_snr = snr / snr_clip

                def compute_cum_snr(reverse=False):
                    x = normalized_clipped_snr.flip(1) if reverse else normalized_clipped_snr
                    cum = torch.zeros_like(x)
                    for t in range(k.shape[1]):
                        if t == 0:
                            cum[:, t] = x[:, t]
                        else:
                            cum[:, t] = cum_snr_decay * cum[:, t - 1] + (1 - cum_snr_decay) * x[:, t]
                    cum = torch.nn.functional.pad(cum[:, :-1], (1, 0, 0, 0), value=0.0)
                    return cum.flip(1) if reverse else cum

                if self.use_causal_mask:
                    cum_snr = compute_cum_snr()
                else:
                    cum_snr = (compute_cum_snr(reverse=True) + compute_cum_snr(reverse=False)) * 0.5

                clipped_fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (1 - normalized_clipped_snr)
                fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (1 - normalized_snr)
                snr = fused_snr * snr_clip
                clipped_snr = clipped_fused_snr * snr_clip
                epsilon_weighting = clipped_snr / snr.clamp(min=1e-8)
        else:
            raise ValueError(f"unknown loss weighting strategy {strategy}")
        if self.objective == "pred_noise":
            return epsilon_weighting
        if self.objective == "pred_x0":
            return epsilon_weighting * snr
        if self.objective == "pred_v":
            return epsilon_weighting * snr / (snr + 1)
        raise ValueError(f"unknown objective {self.objective}")

    def compute_bivariate_fused_weights(
        self,
        k_obs: torch.Tensor,
        k_act: torch.Tensor,
        *,
        fusion_mode: str = "blend",
        gamma_obs: Optional[float] = None,
        gamma_act: float = 0.90,
        lambda_blend: float = 0.9,
        mu: float = 0.5,
        kappa: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bivariate fused-SNR loss weights for decoupled obs/action noise.

        Generalizes 'fused_min_snr' to two per-frame noise-level streams
        (same-frame obs and action at independent levels). Per stream m, a
        cumulated history S̄_m is an EMA of the normalized clipped SNR with
        its own decay ``gamma_m`` (forward-only when ``use_causal_mask``,
        else forward/backward averaged — must match the trunk's attention).
        Histories are γ-transported and blended into one context
        ``S̄_h = λ·(γ_o·S̄_o) + (1−λ)·(γ_a·S̄_a)``; fusion then follows the
        upstream two-term noisy-OR shape (own-frame term undamped,
        diffusion_transition.py:254-255):

          obs:              S'_o = 1 − (1 − S_o)(1 − S̄_h)
          act ("blend"):    S'_a = 1 − (1 − S_a)(1 − [μ·S_o + (1−μ)·S̄_h])
          act ("product"):  S'_a = 1 − (1 − S_a)(1 − κ·S_o)(1 − S̄_h)

        "blend" keeps the fusion two-term with a convex context (current obs
        vs history budgeted by ``mu`` — total cross-signal cannot inflate);
        "product" treats current obs as a third independent source damped by
        ``kappa``. Returns per-token weights ``(w_obs, w_act)`` mapped through
        the same clipped/unclipped min-SNR tracks and objective factor as
        'fused_min_snr'.

        DORMANT: nothing selects this yet; intended caller is the decoupled
        two-stream outer stage once parity training lands. Both ``k`` tensors
        must be (B, T) — the time axis is required.
        """
        if fusion_mode not in ("blend", "product"):
            raise ValueError(f"unknown fusion_mode {fusion_mode!r}")
        if k_obs.dim() != 2 or k_act.dim() != 2:
            raise ValueError(
                "bivariate_fused requires (B, T) noise levels for both "
                f"streams; got {tuple(k_obs.shape)} / {tuple(k_act.shape)}"
            )
        g_o = self.cum_snr_decay if gamma_obs is None else float(gamma_obs)
        g_a = float(gamma_act)
        snr_clip = self.snr_clip

        def _ema(x: torch.Tensor, decay: float) -> torch.Tensor:
            def one_dir(reverse: bool) -> torch.Tensor:
                xx = x.flip(1) if reverse else x
                cum = torch.zeros_like(xx)
                for t in range(xx.shape[1]):
                    if t == 0:
                        cum[:, t] = xx[:, t]
                    else:
                        cum[:, t] = decay * cum[:, t - 1] + (1 - decay) * xx[:, t]
                cum = torch.nn.functional.pad(cum[:, :-1], (1, 0, 0, 0), value=0.0)
                return cum.flip(1) if reverse else cum

            if self.use_causal_mask:
                return one_dir(False)
            return (one_dir(False) + one_dir(True)) * 0.5

        def _tracks(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self.snr[k] / snr_clip, self.clipped_snr[k] / snr_clip

        ns_o, ncs_o = _tracks(k_obs)
        ns_a, ncs_a = _tracks(k_act)
        s_h = lambda_blend * (g_o * _ema(ncs_o, g_o)) \
            + (1 - lambda_blend) * (g_a * _ema(ncs_a, g_a))

        def _to_weights(fused: torch.Tensor, clipped_fused: torch.Tensor) -> torch.Tensor:
            snr = fused * snr_clip
            clipped = clipped_fused * snr_clip
            eps_w = clipped / snr.clamp(min=1e-8)
            if self.objective == "pred_noise":
                return eps_w
            if self.objective == "pred_x0":
                return eps_w * snr
            if self.objective == "pred_v":
                return eps_w * snr / (snr + 1)
            raise ValueError(f"unknown objective {self.objective}")

        w_obs = _to_weights(
            1 - (1 - ns_o) * (1 - s_h),
            1 - (1 - ncs_o) * (1 - s_h),
        )
        if fusion_mode == "blend":
            ctx = mu * ncs_o + (1 - mu) * s_h
            w_act = _to_weights(
                1 - (1 - ns_a) * (1 - ctx),
                1 - (1 - ncs_a) * (1 - ctx),
            )
        else:
            w_act = _to_weights(
                1 - (1 - ns_a) * (1 - kappa * ncs_o) * (1 - s_h),
                1 - (1 - ncs_a) * (1 - kappa * ncs_o) * (1 - s_h),
            )
        return w_obs, w_act

    def compute_loss(
        self,
        v_pred: torch.Tensor,
        q_state: dict,
        weighting_strategy: Optional[str] = None,
    ) -> torch.Tensor:
        """Compute weighted per-token MSE given the backbone's output and
        the q_state dict from encode. Matches ContinuousDiffusion.compute_loss
        interface so DFoTLoss works with both.

        Args:
            v_pred: model output (B, T, ...) — interpretation depends on objective.
            q_state: dict with 'x_t' (noised), 'k' (timesteps), 'noise' (eps),
                     'x_start' (clean target).
            weighting_strategy: optional per-call override of the ctor
                ``loss_weighting_strategy`` (e.g. per modality group);
                ``None`` uses the ctor strategy on the identical code path.
        """
        k = q_state["k"]
        noise = q_state["noise"]
        x_start = q_state["x_start"]
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            target = self.predict_v(x_start, k, noise)
        else:
            raise ValueError(f"unknown objective {self.objective}")
        per_element = (v_pred - target.detach()) ** 2
        # Reduce over all trailing dims (action_dim or C,H,W) to get per-token loss
        n_trailing = per_element.dim() - k.dim()
        for _ in range(n_trailing):
            per_element = per_element.mean(dim=-1)
        w = self.compute_loss_weights(k, strategy=weighting_strategy)
        return per_element * w

    def forward(
        self,
        backbone,
        x: torch.Tensor,
        k: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
    ):
        """Train-time forward.

        Args:
            backbone: callable ``(x_k, k, external_cond) -> model_out`` where
                ``x_k`` is (B, T, action_dim), ``k`` is (B, T).
            x: clean target ``(B, T, action_dim)``.
            k: per-token noise levels ``(B, T)`` (long, in ``[0, timesteps)``).
            external_cond: per-token obs cond ``(B, T, cond_dim)`` or
                ``(B, cond_dim)`` (broadcast inside ``backbone``).
        """
        noise = torch.randn_like(x).clamp_(-self.clip_noise, self.clip_noise)
        x_k = self.q_sample(x, k, noise=noise)
        pred = self.model_predictions(backbone, x_k, k, external_cond)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x
        elif self.objective == "pred_v":
            target = self.predict_v(x, k, noise)
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = F.mse_loss(pred.model_out, target.detach(), reduction="none")
        w = self.compute_loss_weights(k)
        # Broadcast (B, T) weights against (B, T, action_dim) loss.
        loss = loss * w.unsqueeze(-1)
        return pred.pred_x_start, loss
