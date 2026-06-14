"""Shared JEPA integration hooks for policy algos.

``JEPAMixin`` carries the *family-agnostic* orchestration: build the JEPA module,
compute the auxiliary loss for one embodiment's batch, and step the EMA target.
Each algo (HPT, BCRNN) overrides four small hooks that know its encoder and where
its current/future observations live:

    _jepa_online_encoder()  -> nn.Module    # the encoder to pressure + EMA-mirror
    _jepa_latent_dim()      -> int          # that encoder's pooled output width
    _jepa_encode(enc, obs)  -> (B, D)       # pooled latent for one obs tensor
    _jepa_extract(per_emb)  -> (obs_t, obs_future, action_chunk, valid_mask|None)

Nothing runs unless a model config sets ``jepa.enabled=true``; ``self.jepa`` stays
``None`` otherwise and every hook below is a cheap no-op.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from egomimic.models.jepa import JEPAModule


def _cfg_get(cfg, key, default):
    """Read a key from a dict or an OmegaConf DictConfig uniformly."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:
            pass
    return getattr(cfg, key, default)


class JEPAMixin:
    # Set by _jepa_setup; guards every hook.
    jepa: Optional[JEPAModule] = None
    jepa_lambda: float = 0.0
    _jepa_dumped_keys: bool = False

    # ---- setup ---------------------------------------------------------- #
    def _jepa_setup(self, jepa_cfg, action_dim: int, action_horizon: int) -> None:
        """Build the JEPA module if enabled. Call at the END of the algo __init__
        (after the obs encoder + ``self.nets['policy']`` exist)."""
        self.jepa = None
        self.jepa_lambda = 0.0
        if jepa_cfg is None or not bool(_cfg_get(jepa_cfg, "enabled", False)):
            return

        self.jepa_lambda = float(_cfg_get(jepa_cfg, "lambda", 1.0))
        module = JEPAModule(
            online_encoder=self._jepa_online_encoder(),
            latent_dim=int(self._jepa_latent_dim()),
            action_dim=int(action_dim),
            action_horizon=int(action_horizon),
            hidden_dim=int(_cfg_get(jepa_cfg, "hidden_dim", 512)),
            n_layers=int(_cfg_get(jepa_cfg, "n_layers", 2)),
            ema_decay=float(_cfg_get(jepa_cfg, "ema_decay", 0.996)),
            loss_type=str(_cfg_get(jepa_cfg, "loss_type", "smooth_l1")),
            normalize_targets=bool(_cfg_get(jepa_cfg, "normalize_targets", True)),
            var_coef=float(_cfg_get(jepa_cfg, "var_coef", 1.0)),
            cov_coef=float(_cfg_get(jepa_cfg, "cov_coef", 0.04)),
        )
        # Register under the policy so the predictor trains and the whole module
        # follows the policy's .to(device)/.float() moves. The EMA encoder's params
        # are requires_grad=False, so the optimizer skips them (grads stay None).
        self.nets["policy"].jepa = module
        self.jepa = module
        print(
            f"[JEPA] enabled lambda={self.jepa_lambda} latent_dim={self._jepa_latent_dim()} "
            f"action={action_dim}x{action_horizon} ema={module.ema_decay} loss={module.loss_type}"
        )

    # ---- per-batch loss + EMA (shared) ---------------------------------- #
    def _jepa_loss_for_batch(self, per_emb_batch) -> Optional[torch.Tensor]:
        if self.jepa is None:
            return None
        extracted = self._jepa_extract(per_emb_batch)
        if extracted is None:
            return None
        obs_t, obs_future, action_chunk, valid_mask = extracted
        enc = self._jepa_online_encoder()
        z_ctx = self._jepa_encode(enc, obs_t)  # grad ON -> pressures the encoder
        with torch.no_grad():
            z_tgt = self._jepa_encode(self.jepa.target_encoder, obs_future)
        return self.jepa.loss(z_ctx, z_tgt, action_chunk, valid_mask)

    def _jepa_ema_step(self) -> None:
        if self.jepa is None or not self.nets.training:
            return
        self.jepa.update_target(self._jepa_online_encoder())

    def _jepa_debug_keys_once(self, per_emb_batch) -> None:
        """Print available batch keys/shapes the first time, so the smoke test
        reveals exactly how the future-obs key is named in the real batch."""
        if self._jepa_dumped_keys:
            return
        self._jepa_dumped_keys = True
        try:
            items = {
                k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__)
                for k, v in per_emb_batch.items()
            }
            print(f"[JEPA][debug] batch keys -> {items}")
        except Exception as e:  # never let debug break training
            print(f"[JEPA][debug] could not dump keys: {e}")

    # ---- family hooks (override in each algo) ---------------------------- #
    def _jepa_online_encoder(self):
        raise NotImplementedError

    def _jepa_latent_dim(self) -> int:
        raise NotImplementedError

    def _jepa_encode(self, encoder, obs) -> torch.Tensor:
        raise NotImplementedError

    def _jepa_extract(self, per_emb_batch) -> Optional[Tuple]:
        raise NotImplementedError
