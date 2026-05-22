"""``ObsActionDFoTOuterStage`` — joint obs+action diffusion forcing.

Mirrors the pattern used in https://github.com/buoyancy99/diffusion-forcing
``df_planning.py``: per step, the diffusion target is a "bundle" tensor
formed by concatenating obs modalities with the action along the last
dim, ``bundle = concat([obs_0, ..., obs_K, action], dim=-1)``. The whole
bundle is noised together at training time with one independent noise
level per (sequence position) token, and the backbone is asked to denoise
it jointly.

Image obs (or any other high-dimensional modality) is NOT added to the
bundle — it stays in the AdaLN conditioning path via ``cond_encoder``, so
the backbone sees it as un-noised side information. The split between
"bundle modalities" and "cond-only modalities" is decided by the
``bundle_obs_keys`` / ``bundle_obs_dims`` config.

At inference time, the algo's AR-staircase / chunk samplers run on a
buffer of ``bundle_dim`` width; before the committed slice is sent to the
env, ``DFoT._inference_step_*`` applies ``outer_stage.action_slice`` to
pick the action portion out of the bundle and discards the obs portion.

Train-time loss is unchanged: ``DFoTLoss`` averages MSE over all bundle
dims, so both obs and action contribute to the gradient — that's the
"joint world-model + policy" objective from the diffusion-forcing paper.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

import torch

from egomimic.algo.dfot.outer_stage import DFoTOuterStage


class ObsActionDFoTOuterStage(DFoTOuterStage):
    """DFoT outer stage that diffuses concat([obs, action]) jointly.

    Args:
        action_dim: width of the ACTION portion only (e.g. 2 for pushshapes).
        cond_encoder: usually configured with image_encoders only — state
            modalities that are in ``bundle_obs_keys`` should NOT also be
            in cond_encoder.obs_specs (or they'd appear on both the noised
            input and the cond path, which double-counts).
        backbone: ``DFoTBackbone`` configured with ``action_dim = obs_dim +
            action_dim`` (the full bundle width). The class checks this.
        diffusion: ``ContinuousDiffusion`` or ``DiscreteDiffusion``.
        bundle_obs_keys: list of obs keys to include in the diffusion
            bundle, in the order they are concatenated. Action is always
            concatenated LAST (so ``action_slice`` is a single trailing
            range). Each key must exist on every training batch.
        bundle_obs_dims: per-key feature width matching ``bundle_obs_keys``.
            ``sum(bundle_obs_dims) + action_dim`` must equal the backbone's
            configured ``action_dim`` (= bundle width).
        cond_output_key: same as base.
    """

    def __init__(
        self,
        action_dim: int,
        cond_encoder,
        backbone,
        diffusion,
        bundle_obs_keys: List[str],
        bundle_obs_dims: List[int],
        cond_output_key: str = "fused_cond",
    ):
        super().__init__(
            action_dim=action_dim,
            cond_encoder=cond_encoder,
            backbone=backbone,
            diffusion=diffusion,
            cond_output_key=cond_output_key,
        )
        if len(bundle_obs_keys) != len(bundle_obs_dims):
            raise ValueError(
                f"bundle_obs_keys ({len(bundle_obs_keys)}) and "
                f"bundle_obs_dims ({len(bundle_obs_dims)}) must align."
            )
        self.bundle_obs_keys = list(bundle_obs_keys)
        self.bundle_obs_dims = [int(d) for d in bundle_obs_dims]
        self._obs_total = sum(self.bundle_obs_dims)
        self._bundle_dim = self._obs_total + int(action_dim)

        # Backbone width sanity check — its ``action_dim`` must match the
        # bundle width since it produces ``v_pred`` of the same shape.
        bb_dim = int(getattr(backbone, "action_dim", -1))
        if bb_dim != self._bundle_dim:
            raise ValueError(
                f"backbone.action_dim ({bb_dim}) must equal bundle width "
                f"obs_total({self._obs_total}) + action_dim({action_dim}) "
                f"= {self._bundle_dim}."
            )

    # ------------------------------------------------------------------
    # Override base properties so the algo's inference path slices and
    # sizes against the bundle correctly.
    # ------------------------------------------------------------------

    @property
    def bundle_dim(self) -> int:
        return self._bundle_dim

    @property
    def action_slice(self) -> slice:
        # action is concatenated last; slice picks it out of the bundle.
        return slice(self._obs_total, self._bundle_dim)

    # ------------------------------------------------------------------
    # Bundle construction. Reads obs values from ``ctx.obs`` (set by
    # algo.forward_training before calling self.outer_stage(...)) and the
    # action from ``batch[ctx.action_key]``. Concatenates along the
    # trailing dim, then runs ``diffusion.q_sample`` on the whole thing.
    # ------------------------------------------------------------------

    def _build_bundle(self, batch: dict, ctx: SimpleNamespace) -> torch.Tensor:
        actions = batch[ctx.action_key]
        pieces: List[torch.Tensor] = []
        for key, dim in zip(self.bundle_obs_keys, self.bundle_obs_dims):
            if key not in ctx.obs:
                raise KeyError(
                    f"obs key '{key}' required in the bundle but missing from "
                    f"ctx.obs (keys: {list(ctx.obs.keys())})."
                )
            v = ctx.obs[key]
            # Allow both packed (T_total, D) and padded (B, T, D); reject
            # anything else — image obs (4D) should be in cond_encoder, not
            # the bundle.
            if v.dim() not in (actions.dim() - 1, actions.dim()):
                # Permissive: only require trailing-dim alignment and same
                # rank as actions for cat. Anything else is a config bug.
                pass
            if v.shape[-1] != dim:
                raise ValueError(
                    f"obs '{key}' trailing dim {v.shape[-1]} != configured "
                    f"bundle_obs_dim {dim}."
                )
            pieces.append(v)
        pieces.append(actions)
        bundle = torch.cat(pieces, dim=-1)
        if bundle.shape[-1] != self._bundle_dim:
            raise RuntimeError(
                f"bundle width mismatch: built {bundle.shape[-1]}, expected "
                f"{self._bundle_dim}."
            )
        return bundle

    def encode(self, batch: dict, ctx: SimpleNamespace) -> torch.Tensor:
        bundle = self._build_bundle(batch, ctx)

        if ctx.is_packed:
            if bundle.dim() != 2:
                raise ValueError(
                    f"packed bundle must be (T_total, bundle_dim); "
                    f"got {tuple(bundle.shape)}"
                )
            T_total = bundle.shape[0]
            cond = self._encode_cond_packed(ctx.obs)
            t = self._sample_noise_levels((T_total,), bundle.device)
        else:
            if bundle.dim() != 3:
                raise ValueError(
                    f"padded bundle must be (B, T, bundle_dim); "
                    f"got {tuple(bundle.shape)}"
                )
            B, T, _ = bundle.shape
            cond = self._encode_cond_padded(ctx.obs, T)
            t = self._sample_noise_levels((B, T), bundle.device)

        q = self.diffusion.q_sample(bundle, t)
        ctx.q_state = q
        ctx.external_cond = cond
        # Stash the per-step ground-truth bundle so downstream logging /
        # diagnostics can compute per-modality losses if desired (the
        # default DFoTLoss reduces over all bundle dims uniformly).
        ctx.bundle_clean = bundle
        return q["x_t"]
