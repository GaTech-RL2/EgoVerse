"""Causal and bidirectional transformer decoders over arc-token rows.

The comparison these nodes exist to serve is *causal vs bidirectional
generation of one action token*, holding the observation encoder, the
optimizer, the data and the parameter budget fixed. Everything here therefore
sits in exactly the slot ``MultiJActionSampler`` occupies in the flow-matching
baseline: read ``condition``, write ``pred_action``, and let the pre-existing
``NativeActionMSELoss`` score it. Reusing that loss node — rather than scoring
internally — is what keeps the arms comparable: the loss, the padding mask and
the normalization are literally the same object.

WHAT A "STEP" IS HERE. One training sample is a single observation plus one
planar arc token of shape ``(H, A)``: ``H = M + 1`` rows for the ``append``
velocity layout (M waypoints, then one trailing velocity row that is NOT a
waypoint), ``A = 5`` channels ``[x, y, cos, sin, grip]``. So the sequence the
decoder is autoregressive OVER is the rows of that token, not time. Row m is
a point further along the same arc, which is why feeding row m-1 back is
meaningful: it is the model's own committed path so far.

    variant          attention        row m conditioned on        actions from
    ---------------  ---------------  --------------------------  ------------
    causal_bidir     bidirectional    the whole token at once     direct head
    state_action_ar  causal           rows < m (its own output)   direct head
    state_idm        causal           rows < m (its own output)   IDM(p_m,p_m+1)

``causal_bidir`` is the load-bearing control. It shares this file's backbone,
this file's head and this file's loss with the causal arms and differs ONLY in
the attention mask and in consuming learned queries instead of shifted rows.
Without it, a win by ``state_action_ar`` over the flow-matching baseline
confounds three things at once — causal generation, the MSE head, and this
backbone. With it, arm2-vs-arm3 isolates the mask.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from egomimic.pipeline.core import Stage

VARIANTS = ("causal_bidir", "state_action_ar", "state_idm")


class _Block(nn.Module):
    """Pre-LN transformer block. Deliberately plain: the study varies the
    attention MASK, so anything clever here would be a second uncontrolled
    difference between the arms."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        hidden = d_model * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None):
        h = self.norm1(x)
        # need_weights=False keeps SDPA on its fused path; the weights are
        # never read here and materializing them costs real memory at d=1024.
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.norm2(x)))


class InverseDynamics(nn.Module):
    """``(p_m, p_{m+1}) -> a_m``.

    Small and separate from the backbone on purpose: the ``state_idm`` claim is
    that predicting the PATH is the hard part and recovering the action between
    two consecutive points is nearly mechanical. Giving the IDM real capacity
    would blur exactly the comparison the arm exists to make.
    """

    def __init__(self, pose_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * pose_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, p_t, p_next):
        return self.net(torch.cat([p_t, p_next], dim=-1))


class ARActionDecoder(Stage):
    """Transformer decoder over the rows of one action token.

    Occupies the sampler slot: ``condition`` in, ``pred_action`` out, scored by
    the shared ``NativeActionMSELoss``.

    At train time the causal variants are teacher-forced from ``target``; at
    rollout ``target`` does not exist, so they free-run, feeding each predicted
    row back as the next input. The bidirectional variant never reads
    ``target`` at all and so runs one identical code path in both modes.
    """

    reads = ["condition", "embodiment", "target"]
    writes = ["pred_action"]
    # No target at rollout — the causal arms must free-run, which is precisely
    # the distribution shift this study is trying to measure.
    reads_by_mode = {"rollout": ["condition", "embodiment"]}
    writes_by_mode = {"rollout": ["pred_action"]}

    def __init__(
        self,
        condition_input_dim: int,
        action_horizon: int,
        action_dims: Dict[str, int],
        variant: str = "state_action_ar",
        d_model: int = 1024,
        n_layers: int = 24,
        n_heads: int = 16,
        ff_mult: int = 4,
        dropout: float = 0.1,
        n_waypoints: int | None = None,
        pose_channels: int = 4,
        idm_hidden: int = 256,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.variant = variant
        self.causal = variant != "causal_bidir"
        self.uses_idm = variant == "state_idm"
        # `state_idm` feeds back the POSE only. Two reasons, and the second is
        # the one that bites: (1) the arm's claim is that the path is the hard
        # part, so letting it read action history back would hand it the
        # shortcut the arm exists to deny; (2) its action channels are a
        # readout of the p_m -> p_{m+1} transition, so at row m they are still
        # provisional — feeding them back would make the fed-back sequence
        # differ from the final output, and training would optimize a function
        # the rollout never runs.
        self.feeds_back_actions = not self.uses_idm
        self.condition_input_dim = int(condition_input_dim)
        self.action_horizon = int(action_horizon)
        self.action_dims = {str(k): int(v) for k, v in dict(action_dims).items()}
        if not self.action_dims:
            raise ValueError("action_dims must configure at least one domain")
        self.d_model = int(d_model)
        self.pose_channels = int(pose_channels)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if self.action_horizon <= 0 or self.d_model <= 0:
            raise ValueError("action_horizon and d_model must be positive")

        # `append` layout parks a velocity row after the waypoints. It is not a
        # point on the path, so the IDM must not try to difference into it.
        self.n_waypoints = (
            self.action_horizon if n_waypoints is None else int(n_waypoints)
        )
        if not 0 < self.n_waypoints <= self.action_horizon:
            raise ValueError(
                f"n_waypoints must be in (0, {self.action_horizon}], "
                f"got {self.n_waypoints}"
            )
        if self.uses_idm and self.pose_channels >= min(self.action_dims.values()):
            raise ValueError(
                "state_idm needs at least one non-pose channel for the IDM to "
                f"produce; pose_channels={self.pose_channels} but the narrowest "
                f"domain has {min(self.action_dims.values())}"
            )

        self.cond_proj = nn.Linear(self.condition_input_dim, self.d_model)
        self.domain_embeddings = nn.ParameterDict(
            {
                d: nn.Parameter(torch.empty(self.d_model).normal_(std=0.02))
                for d in self.action_dims
            }
        )
        # One projection pair per domain, mirroring MultiJActionSampler's
        # per-domain decoders so a cotrain over mixed effector widths stays
        # possible without reshaping this node.
        self.row_in = nn.ModuleDict(
            {d: nn.Linear(a, self.d_model) for d, a in self.action_dims.items()}
        )
        self.row_out = nn.ModuleDict(
            {d: nn.Linear(self.d_model, a) for d, a in self.action_dims.items()}
        )
        # Bidirectional decoding has no shifted input to consume, so it reads
        # learned per-row queries instead. Allocated for every variant so the
        # parameter count does not move with the mask.
        self.queries = nn.Parameter(
            torch.empty(self.action_horizon, self.d_model).normal_(std=0.02)
        )
        self.pos = nn.Parameter(
            torch.empty(self.action_horizon + 1, self.d_model).normal_(std=0.02)
        )
        self.blocks = nn.ModuleList(
            [_Block(self.d_model, n_heads, ff_mult, dropout) for _ in range(n_layers)]
        )
        self.norm_out = nn.LayerNorm(self.d_model)
        # Channel gate on the DECODER INPUT, not on `row_in` itself: masking
        # keeps every arm's parameter count identical, which a narrower
        # projection would not.
        max_dim = max(self.action_dims.values())
        gate = torch.ones(max_dim)
        if not self.feeds_back_actions:
            gate[self.pose_channels :] = 0.0
        self.register_buffer("input_gate", gate, persistent=False)
        self.idm = (
            nn.ModuleDict(
                {
                    d: InverseDynamics(
                        self.pose_channels, a - self.pose_channels, idm_hidden
                    )
                    for d, a in self.action_dims.items()
                }
            )
            if self.uses_idm
            else None
        )

    # ------------------------------------------------------------------ core

    def _causal_mask(self, n: int, device) -> torch.Tensor:
        return torch.triu(
            torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1
        )

    def _backbone(self, x: torch.Tensor, mask: torch.Tensor | None):
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, mask, use_reentrant=False)
            else:
                x = block(x, mask)
        return self.norm_out(x)

    def _condition_token(self, batch: dict, domain: str) -> torch.Tensor:
        condition = batch["condition"]
        if condition.shape[-1] != self.condition_input_dim:
            raise ValueError(
                f"Expected {self.condition_input_dim}-D condition, got "
                f"{condition.shape[-1]}"
            )
        token = self.cond_proj(condition)
        return token + self.domain_embeddings[domain].to(token)

    def _embed_rows(self, rows: torch.Tensor, domain: str) -> torch.Tensor:
        """Project fed-back rows, gating off channels this variant must not see."""
        gate = self.input_gate[: rows.shape[-1]].to(rows)
        return self.row_in[domain](rows * gate)

    def _infer_actions(self, rows: torch.Tensor, domain: str) -> torch.Tensor:
        """Replace the non-pose channels of the waypoint rows with IDM output.

        The last waypoint has no successor, so its action is copied from the
        one before it — the same padding convention the token's own final row
        already carries.
        """
        w, p = self.n_waypoints, self.pose_channels
        pose = rows[:, :w, :p]
        nxt = torch.cat([pose[:, 1:], pose[:, -1:]], dim=1)
        waypoints = torch.cat([pose, self.idm[domain](pose, nxt)], dim=-1)
        if w == rows.shape[1]:
            return waypoints
        # Trailing velocity row is a summary of the path, not a point on it:
        # it is read straight off the head.
        return torch.cat([waypoints, rows[:, w:]], dim=1)

    def _decode_rows(self, feats: torch.Tensor, domain: str) -> torch.Tensor:
        """Map backbone features at row positions to native action channels."""
        rows = self.row_out[domain](feats)
        if not self.uses_idm:
            return rows
        return self._infer_actions(rows, domain)

    # -------------------------------------------------------------- variants

    def _forward_bidirectional(self, batch: dict, domain: str) -> torch.Tensor:
        cond = self._condition_token(batch, domain).unsqueeze(1)
        queries = self.queries.unsqueeze(0).expand(cond.shape[0], -1, -1)
        x = torch.cat([cond, queries], dim=1) + self.pos.unsqueeze(0)
        feats = self._backbone(x, None)[:, 1:]
        return self._decode_rows(feats, domain)

    def _forward_causal_teacher_forced(
        self, batch: dict, domain: str, target: torch.Tensor
    ) -> torch.Tensor:
        cond = self._condition_token(batch, domain).unsqueeze(1)
        # Shift right: position 0 is the condition and predicts row 0, position
        # i>0 consumes row i-1 and predicts row i.
        shifted = self._embed_rows(target[:, :-1], domain)
        x = torch.cat([cond, shifted], dim=1) + self.pos[: self.action_horizon]
        mask = self._causal_mask(x.shape[1], x.device)
        feats = self._backbone(x, mask)
        return self._decode_rows(feats, domain)

    def _forward_causal_free_running(self, batch: dict, domain: str) -> torch.Tensor:
        cond = self._condition_token(batch, domain).unsqueeze(1)
        x = cond + self.pos[:1]
        rows: list[torch.Tensor] = []
        for step in range(self.action_horizon):
            mask = self._causal_mask(x.shape[1], x.device)
            feats = self._backbone(x, mask)
            # Read the head directly rather than going through `_decode_rows`:
            # under IDM the action channels are a function of the NEXT pose,
            # which does not exist yet. They are filled in once below, when the
            # whole path is known.
            row = self.row_out[domain](feats[:, -1:])
            rows.append(row)
            if step + 1 < self.action_horizon:
                x = torch.cat(
                    [x, self._embed_rows(row, domain) + self.pos[step + 1]], dim=1
                )
        out = torch.cat(rows, dim=1)
        # Safe to defer: the IDM only ever touches channels this variant gates
        # out of its own input, so the fed-back sequence above is already final.
        return self._infer_actions(out, domain) if self.uses_idm else out

    # ------------------------------------------------------------------- api

    def forward(self, batch: dict) -> dict:
        domain = str(batch["embodiment"])
        if domain not in self.action_dims:
            raise KeyError(
                f"Unknown embodiment {domain!r}; configured="
                f"{sorted(self.action_dims)}"
            )
        if not self.causal:
            batch["pred_action"] = self._forward_bidirectional(batch, domain)
            return batch

        target = batch.get("target")
        if target is None:
            batch["pred_action"] = self._forward_causal_free_running(batch, domain)
            return batch
        if target.shape[1] != self.action_horizon:
            raise ValueError(
                f"ARActionDecoder expects target horizon {self.action_horizon}, "
                f"got {tuple(target.shape)}"
            )
        batch["pred_action"] = self._forward_causal_teacher_forced(
            batch, domain, target
        )
        return batch
