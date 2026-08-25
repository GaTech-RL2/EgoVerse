"""Four AR state-action variants over one causal backbone.

They exist to separate three things that are usually entangled in a single
"autoregressive policy":

  1. WHAT the model predicts   -- next state, next action, or both
  2. HOW actions are obtained  -- decoded directly, or inferred from a
                                  predicted state by an inverse dynamics model
  3. WHETHER predictions are fed back during TRAINING, or only at rollout

    variant             predicts        action from        fed back in training
    ------------------  --------------  -----------------  --------------------
    action_ar           action          direct head        yes (scheduled)
    state_idm           state           IDM(s_t, s_hat)    yes (scheduled)
    state_action_ar     state + action  direct head        yes (scheduled)
    joint_tf            state + action  direct head        NO (teacher forced)

`joint_tf` is the control: it is `state_action_ar` with feedback disabled, so
the pair isolates the effect of closed-loop training on its own. Without it,
any difference between the AR variants confounds the feedback question with
the representation question.

Feedback uses SCHEDULED SAMPLING (`feedback_prob` ramped over training) rather
than full closed-loop unrolling from step 0. Full closed loop from an
untrained model feeds garbage into itself and the loss is dominated by
compounding noise; the ramp lets the model first learn one-step prediction and
then be exposed to its own errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from egomimic.models.ar.sequence import (
    InverseDynamics,
    TokenSpec,
    Tokenizer,
    gather_positions,
)
from egomimic.models.cores.transformer_core import TransformerCore


@dataclass
class ARVariantConfig:
    image_dim: int = 512
    state_dim: int = 6
    action_dim: int = 4
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    horizon: int = 16
    #: Probability of substituting the model's own prediction for the ground
    #: truth at a step. Ramped 0 -> `feedback_prob_max` over training.
    feedback_prob_max: float = 0.5
    state_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    idm_hidden: int = 256
    extras: dict = field(default_factory=dict)


class _ARBase(nn.Module):
    """Causal backbone + heads. Subclasses declare tokens, losses, feedback."""

    spec: TokenSpec = TokenSpec()
    predicts_state: bool = False
    predicts_action: bool = True
    uses_idm: bool = False
    allow_feedback: bool = True

    def __init__(self, cfg: ARVariantConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = Tokenizer(cfg.image_dim, cfg.state_dim, cfg.action_dim,
                             cfg.d_model)
        self.backbone = TransformerCore(
            input_dim=cfg.d_model, d_model=cfg.d_model, n_layers=cfg.n_layers,
            n_heads=cfg.n_heads, ff_mult=cfg.ff_mult, dropout=cfg.dropout,
            # Positional table must span TOKENS, not timesteps: a variant with
            # three token types per step needs 3x the positions or the table
            # silently truncates the window.
            max_window=cfg.horizon * self.spec.per_step + 8,
        )
        self.state_head = (nn.Linear(cfg.d_model, cfg.state_dim)
                           if self.predicts_state else None)
        self.action_head = (nn.Linear(cfg.d_model, cfg.action_dim)
                            if self.predicts_action and not self.uses_idm
                            else None)
        self.idm = (InverseDynamics(cfg.state_dim, cfg.action_dim,
                                    cfg.idm_hidden) if self.uses_idm else None)

    # ---------------------------------------------------------------- utils

    def feedback_prob(self, progress: float) -> float:
        """Linear ramp of the scheduled-sampling rate over training."""
        if not self.allow_feedback:
            return 0.0
        return float(self.cfg.feedback_prob_max) * max(0.0, min(1.0, progress))

    def _encode(self, image, state, action):
        toks = self.tok(self.spec, image, state, action)
        # BufferedWindowCore.forward returns (features, hidden) so the same
        # core can serve the batched unroll and the step-by-step rollout.
        # Only the features are wanted here.
        feats, _hidden = self.backbone(toks)
        return feats

    # ---------------------------------------------------------------- api

    def forward(self, image, state, action, progress: float = 1.0):
        """Teacher-forced pass with optional scheduled-sampling feedback.

        image  (B, T, image_dim)   state (B, T, state_dim)
        action (B, T, action_dim)  -- the action TAKEN at each step

        Returns dict with `state_pred` (B, T, state_dim) and/or
        `action_pred` (B, T, action_dim), each aligned so index t is the
        prediction FOR step t+1 (state) or step t (action).
        """
        p = self.feedback_prob(progress)
        if p > 0.0 and self.training:
            return self._forward_with_feedback(image, state, action, p)
        return self._heads(self._encode(image, state, action), state)

    def _heads(self, feats, state):
        t = state.shape[1]
        out = {}
        if self.predicts_state:
            h = gather_positions(feats, self.spec, "state", t)
            out["state_pred"] = self.state_head(h)
        if self.uses_idm:
            s_hat = out["state_pred"]
            # a_t is inferred from (s_t, s_hat_{t+1}); the last step has no
            # successor so it is dropped and padded to keep the shape.
            a = self.idm(state[:, :-1], s_hat[:, :-1])
            out["action_pred"] = torch.cat([a, a[:, -1:]], dim=1)
        elif self.action_head is not None:
            key = "action" if self.spec.action else "state"
            h = gather_positions(feats, self.spec, key, t)
            out["action_pred"] = self.action_head(h)
        return out

    def _forward_with_feedback(self, image, state, action, p):
        """Scheduled sampling: replace some inputs with the model's own output.

        Done in ONE extra pass rather than a per-step python loop -- a loop
        over the horizon costs T backbone calls per batch and dominates
        training time for no benefit at this sequence length.
        """
        with torch.no_grad():
            base = self._heads(self._encode(image, state, action), state)
        st, ac = state, action
        if "state_pred" in base and self.spec.state:
            shift = torch.cat([state[:, :1], base["state_pred"][:, :-1]], 1)
            m = (torch.rand_like(state[..., :1]) < p).float()
            st = (1 - m) * state + m * shift.detach()
        if "action_pred" in base and self.spec.action:
            shift = torch.cat([action[:, :1], base["action_pred"][:, :-1]], 1)
            m = (torch.rand_like(action[..., :1]) < p).float()
            ac = (1 - m) * action + m * shift.detach()
        return self._heads(self._encode(image, st, ac), st)

    def losses(self, pred, state, action):
        cfg = self.cfg
        out = {}
        if "state_pred" in pred:
            # index t predicts state t+1
            out["state_loss"] = nn.functional.mse_loss(
                pred["state_pred"][:, :-1], state[:, 1:]) * cfg.state_loss_weight
        if "action_pred" in pred:
            out["action_loss"] = nn.functional.mse_loss(
                pred["action_pred"], action) * cfg.action_loss_weight
        out["loss"] = sum(v for k, v in out.items() if k != "loss")
        return out

    @torch.no_grad()
    def rollout(self, image, state, action, steps: int):
        """Closed-loop: feed each prediction back for `steps` steps."""
        img, st, ac = image, state, action
        preds = []
        for _ in range(steps):
            o = self._heads(self._encode(img, st, ac), st)
            a_next = o["action_pred"][:, -1:]
            preds.append(a_next)
            ac = torch.cat([ac[:, 1:], a_next], dim=1)
            if "state_pred" in o:
                st = torch.cat([st[:, 1:], o["state_pred"][:, -1:]], dim=1)
            img = torch.cat([img[:, 1:], img[:, -1:]], dim=1)
        return torch.cat(preds, dim=1)


class ActionAR(_ARBase):
    """Predict the next ACTION from image+state history; feed actions back."""

    spec = TokenSpec(image=True, state=True, action=True)
    predicts_state = False
    predicts_action = True


class StateIDM(_ARBase):
    """Predict the next STATE; recover the action with an inverse model.

    The bet: dynamics are easier to learn than a policy, and once you know
    where the object should go next, the action that gets it there is close to
    mechanical. No action tokens are fed in at all, so the model cannot lean
    on action history as a shortcut.
    """

    spec = TokenSpec(image=True, state=True, action=False)
    predicts_state = True
    predicts_action = True
    uses_idm = True


class StateActionAR(_ARBase):
    """Predict BOTH, and feed both back. The full autoregressive model."""

    spec = TokenSpec(image=True, state=True, action=True)
    predicts_state = True
    predicts_action = True


class JointTeacherForced(StateActionAR):
    """`StateActionAR` with feedback OFF -- the control for the ablation."""

    allow_feedback = False


VARIANTS = {
    "action_ar": ActionAR,
    "state_idm": StateIDM,
    "state_action_ar": StateActionAR,
    "joint_tf": JointTeacherForced,
}


def build_variant(name: str, cfg: ARVariantConfig) -> _ARBase:
    if name not in VARIANTS:
        raise ValueError(f"unknown AR variant {name!r}; have {sorted(VARIANTS)}")
    return VARIANTS[name](cfg)
