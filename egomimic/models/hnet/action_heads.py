"""Swappable action heads for the H-Net policies (continuous / discrete / gmm).

Both ``HNetPolicy`` and ``FlatFusedPolicy`` own an ``action_out`` projection
and decode it at three kinds of site:

* training forward  -> the RAW head output is the ``pred`` fed to the loss,
* rollout (generate/step) -> the head output must be DECODED to a continuous
  action before it is returned / fed back as the next input,
* teacher-forced eval -> RAW output is decoded before un-normalization.

To avoid duplicating that logic across two policy classes we keep it here as
free functions that operate on the policy module (which stores the head
metadata as plain attributes + a ``_act_bin_centers`` buffer). The
``continuous`` path is intentionally a no-op wrapper so existing runs are
numerically unchanged.

Discrete head: each of the ``action_dim`` continuous dims (already in
*normalized* action space) is binned into ``n_bins`` uniform bins over
``[bin_min, bin_max]``. The projection emits ``action_dim * n_bins`` logits;
training uses per-dim cross-entropy; rollout samples (temperature
``sample_temp``; ``<=0`` -> argmax) and de-bins to the bin center. This lets
each autoregressive step *commit* to one action mode instead of regressing to
the (multimodal) mean — the cause of the "shake at the T" failure.

GMM head (robomimic / ACT style): each predicted action (``action_dim`` D) is
modelled as an explicit ``n_modes`` (M)-component diagonal-Gaussian mixture.
The projection emits ``M * (2D + 1)`` numbers per predicted action -- M means
(D each), M log-stds (D each), and M mixture logits (1 each). Training
minimises the negative log-likelihood of the GT action under that mixture
(``logsumexp`` over modes for numerical stability; log-stds clamped to a sane
range). Decode *commits to one mode* -- by default the mean of the
highest-weight (argmax-logit) component, NEVER the weight-weighted mean -- so
each step picks a single coherent action instead of averaging across modes.
A ``gmm_sample`` flag switches to categorical-then-Gaussian sampling. Like the
other heads it composes with ``chunk_k > 1`` as an independent M-mode mixture
per chunk step.
"""

import math

import torch
import torch.nn as nn


def configure_action_head(module, d_model, action_dim, cfg=None):
    """Attach ``action_out`` + head metadata to ``module`` (a policy).

    ``cfg`` keys (all optional):
      * ``mode`` (``continuous`` | ``discrete`` | ``gmm``) -- head type.
      * discrete: ``n_bins``, ``bin_min``, ``bin_max``, ``sample_temp``.
      * gmm: ``gmm_num_modes`` (M, default 5), ``gmm_min_std`` /
        ``gmm_max_std`` (decode/loss std clamp; defaults exp(-5) / exp(2)),
        ``gmm_sample`` (bool, default False -> mode-committing decode).
      * all: ``chunk_k`` (one-shot K-action chunk; default 1).

    Called once from a policy ``__init__`` in place of
    ``self.action_out = nn.Linear(...)``. The default (``continuous``) path is
    byte-identical to the original single ``nn.Linear`` head.
    """
    cfg = cfg or {}
    mode = str(cfg.get("mode", "continuous"))
    if mode not in ("continuous", "discrete", "gmm"):
        raise ValueError(
            f"action head mode={mode!r} not in {{continuous, discrete, gmm}}"
        )
    module._act_dim = int(action_dim)
    module._act_head = mode
    # ``_act_disc`` retained so the unchanged discrete/continuous branches keep
    # reading the same boolean. ``_act_gmm`` gates the new mixture path.
    module._act_disc = mode == "discrete"
    module._act_gmm = mode == "gmm"
    # SLOT: one-shot action chunk. chunk_k=1 (default) -> single next-action
    # head (byte-identical to before). chunk_k=K>1 -> each timestep's hidden
    # predicts the next K actions in ONE shot (trajectory commitment); the head
    # widens to K*action_dim (or K*action_dim*n_bins for discrete, or
    # K*M*(2D+1) for gmm), and the loss builds K-step windowed targets with an
    # end-of-episode validity mask.
    module._act_chunk_k = int(cfg.get("chunk_k", 1) or 1)
    K = module._act_chunk_k
    if module._act_gmm:
        M = int(cfg["gmm_num_modes"]) if cfg.get("gmm_num_modes") is not None else 5
        if M < 1:
            raise ValueError(f"gmm_num_modes must be >= 1, got {M}")
        D = module._act_dim
        # Std clamp bounds are configurable; default log-std range [-5, 2]
        # (-> sigma in ~[6.7e-3, 7.39]) matches robomimic's GMM defaults and
        # prevents mode collapse (sigma->0) / explosion (sigma->inf).
        min_std = cfg.get("gmm_min_std", None)
        max_std = cfg.get("gmm_max_std", None)
        log_min = math.log(float(min_std)) if min_std is not None else -5.0
        log_max = math.log(float(max_std)) if max_std is not None else 2.0
        if not log_min < log_max:
            raise ValueError(
                f"gmm std clamp invalid: log_min={log_min} >= log_max={log_max}"
            )
        module._act_gmm_modes = M
        module._act_gmm_logstd_min = float(log_min)
        module._act_gmm_logstd_max = float(log_max)
        module._act_gmm_sample = bool(cfg.get("gmm_sample", False))
        # Per predicted action: M means (D) + M log-stds (D) + M logits (1).
        module._act_gmm_per_action = M * (2 * D + 1)
        module.action_out = nn.Linear(d_model, module._act_gmm_per_action * K)
        return
    if not module._act_disc:
        module.action_out = nn.Linear(d_model, action_dim * K)
        return
    n_bins = int(cfg.get("n_bins", 256) or 256)
    bin_min = float(cfg.get("bin_min", -4.0))
    bin_max = float(cfg.get("bin_max", 4.0))
    module._act_nbins = n_bins
    module._act_bmin = bin_min
    module._act_bmax = bin_max
    module._act_temp = float(cfg.get("sample_temp", 1.0))
    module.action_out = nn.Linear(d_model, action_dim * n_bins * K)
    edges = torch.linspace(bin_min, bin_max, n_bins + 1)
    module.register_buffer("_act_bin_centers", 0.5 * (edges[:-1] + edges[1:]))


def action_head_raw(module, h):
    """Project hidden state to the raw head output (the training ``pred``).

    chunk_k == 1 (default): continuous -> ``(..., action_dim)``; discrete ->
    ``(..., action_dim, n_bins)``; gmm -> ``(..., M*(2D+1))`` flat params
    (byte-identical to before for continuous/discrete).
    chunk_k == K > 1: an extra chunk axis is exposed -> continuous
    ``(..., K, action_dim)``; discrete ``(..., K, action_dim, n_bins)``;
    gmm ``(..., K, M*(2D+1))``. The flat gmm param vector is split into
    means / log-stds / logits by ``_gmm_split`` inside the loss/decode.
    """
    out = module.action_out(h)
    K = getattr(module, "_act_chunk_k", 1)
    if getattr(module, "_act_gmm", False):
        out = out.reshape(*out.shape[:-1], K, module._act_gmm_per_action)
        if K == 1:
            out = out.squeeze(-2)
    elif module._act_disc:
        out = out.reshape(*out.shape[:-1], K, module._act_dim, module._act_nbins)
        if K == 1:
            out = out.squeeze(-3)
    else:
        out = out.reshape(*out.shape[:-1], K, module._act_dim)
        if K == 1:
            out = out.squeeze(-2)
    return out


def _gmm_split(module, raw):
    """Flat gmm params ``(..., M*(2D+1))`` -> (means, log_stds, logits).

    means / log_stds: ``(..., M, D)``; logits: ``(..., M)``. log_stds are
    clamped to the configured ``[logstd_min, logstd_max]`` range here so every
    consumer (loss + decode) sees the same clamped value."""
    M = module._act_gmm_modes
    D = module._act_dim
    lead = raw.shape[:-1]
    means = raw[..., : M * D].reshape(*lead, M, D)
    log_stds = raw[..., M * D : 2 * M * D].reshape(*lead, M, D)
    logits = raw[..., 2 * M * D :].reshape(*lead, M)
    log_stds = log_stds.clamp(module._act_gmm_logstd_min, module._act_gmm_logstd_max)
    return means, log_stds, logits


def _gmm_log_prob(module, means, log_stds, logits, target):
    """Diagonal-Gaussian-mixture log-likelihood of ``target``.

    means / log_stds: ``(..., M, D)``; logits: ``(..., M)``; target:
    ``(..., D)``. Returns log p(target): ``(...)``.

    log p(a) = logsumexp_m [ log_softmax(logits)_m
                             + sum_d logN(a_d; mu_md, sigma_md) ],
      logN(a; mu, sigma) = -0.5*((a-mu)/sigma)^2 - log(sigma) - 0.5*log(2pi).
    """
    log_w = nn.functional.log_softmax(logits, dim=-1)  # (..., M)
    a = target.unsqueeze(-2)  # (..., 1, D) -> broadcast over modes
    inv_std = torch.exp(-log_stds)  # 1/sigma; sigma = exp(log_std)
    z = (a - means) * inv_std  # (..., M, D)
    # sum_d logN over the action dims.
    comp_log_prob = (
        -0.5 * (z ** 2) - log_stds - 0.5 * math.log(2.0 * math.pi)
    ).sum(dim=-1)  # (..., M)
    return torch.logsumexp(log_w + comp_log_prob, dim=-1)  # (...)


def _build_chunk_targets(actions, K, cu_seqlens=None):
    """Action stream -> (windowed target, validity mask) for the chunk head.

    Packed (``cu_seqlens`` given): ``actions`` is ``(T_total, A)`` ->
    target ``(T_total, K, A)``, mask ``(T_total, K)`` where
    ``target[i, k] = actions[i + k]`` if ``i + k`` is within frame i's episode
    (else 0); ``mask`` marks the valid (non-cross-episode) entries.
    Padded: ``actions`` is ``(B, T, A)`` -> target ``(B, T, K, A)``, mask
    ``(B, T, K)`` (clamped at the sequence end)."""
    if cu_seqlens is not None:
        T_total, A = actions.shape
        device = actions.device
        cu = cu_seqlens.to(device=device, dtype=torch.long)
        pos = torch.arange(T_total, device=device)
        seq_idx = (pos[:, None] >= cu[None, 1:]).sum(dim=-1)
        e_i = cu[seq_idx + 1]
        ks = torch.arange(K, device=device)
        idx = pos[:, None] + ks[None, :]
        valid = idx < e_i[:, None]
        idx_c = torch.minimum(idx, e_i[:, None] - 1)
        target = actions[idx_c] * valid[..., None].to(actions.dtype)
        return target, valid.to(actions.dtype)
    B, T, A = actions.shape
    device = actions.device
    ks = torch.arange(K, device=device)
    pos = torch.arange(T, device=device)
    idx = pos[:, None] + ks[None, :]
    idx_c = idx.clamp(max=T - 1)
    target = actions[:, idx_c] * (idx < T)[None, ..., None].to(actions.dtype)
    valid = (idx < T)[None].expand(B, -1, -1)
    return target, valid.to(actions.dtype)


def action_head_loss(module, raw, target, cu_seqlens=None):
    """Action-prediction loss. continuous -> MSE; discrete -> per-dim CE;
    gmm -> negative log-likelihood of the GT action under the mixture.

    ``target`` is the GT action stream in normalized space ``(..., action_dim)``.
    With chunk_k > 1, ``raw`` carries the extra K axis and the GT stream is
    expanded into K-step windowed targets (with an end-of-episode mask);
    ``cu_seqlens`` (packed mode) is needed to keep windows within an episode.
    """
    K = getattr(module, "_act_chunk_k", 1)
    if K > 1:
        tgt, mask = _build_chunk_targets(target, K, cu_seqlens)  # (...,K,A),(...,K)
        if getattr(module, "_act_gmm", False):
            # raw: (..., K, M*(2D+1)) ; tgt: (..., K, D) ; mask: (..., K).
            means, log_stds, logits = _gmm_split(module, raw)
            log_prob = _gmm_log_prob(module, means, log_stds, logits, tgt)  # (..., K)
            nll = -(log_prob * mask)
            return nll.sum() / (mask.sum() + 1e-8)
        if not module._act_disc:
            diff = (raw - tgt) ** 2 * mask[..., None]
            return diff.sum() / (mask.sum() * module._act_dim + 1e-8)
        idx = _to_bins(module, tgt)  # (..., K, action_dim)
        ce = nn.functional.cross_entropy(
            raw.reshape(-1, module._act_nbins), idx.reshape(-1), reduction="none"
        ).reshape(idx.shape)  # (..., K, action_dim)
        m = mask[..., None].expand_as(ce)
        return (ce * m).sum() / (m.sum() + 1e-8)
    if getattr(module, "_act_gmm", False):
        # raw: (..., M*(2D+1)) ; target: (..., D).
        means, log_stds, logits = _gmm_split(module, raw)
        log_prob = _gmm_log_prob(module, means, log_stds, logits, target)  # (...)
        return -log_prob.mean()
    if not module._act_disc:
        return nn.functional.mse_loss(raw, target)
    idx = _to_bins(module, target)  # (..., action_dim) long
    return nn.functional.cross_entropy(
        raw.reshape(-1, module._act_nbins), idx.reshape(-1)
    )


def action_head_decode(module, raw):
    """Raw head output -> continuous action ``(..., action_dim)``.

    continuous -> identity; discrete -> sample (or argmax) per dim + de-bin;
    gmm -> commit to ONE mode (mean of the highest-weight component by default,
    NEVER the weight-weighted mean) or, when ``gmm_sample`` is set, sample a
    component by the mixture weights then sample/return that Gaussian.
    With chunk_k > 1, the first chunk step (k=0) is used as the executed /
    fed-back action so AR rollout + teacher-forced eval are unchanged. (Full
    chunk + temporal-ensemble rollout is a separate follow-up knob.)
    Used only on no-grad rollout / eval paths.
    """
    K = getattr(module, "_act_chunk_k", 1)
    if getattr(module, "_act_gmm", False):
        if K > 1:
            raw = raw[..., 0, :]  # (..., M*(2D+1)) executed (k=0) chunk step
        return _gmm_decode(module, raw)
    if K > 1:
        raw = raw[..., 0, :, :] if module._act_disc else raw[..., 0, :]
    if not module._act_disc:
        return raw
    logits = raw  # (..., action_dim, n_bins)
    temp = module._act_temp
    if temp and temp > 0:
        probs = nn.functional.softmax(logits / temp, dim=-1)
        idx = torch.multinomial(
            probs.reshape(-1, module._act_nbins), num_samples=1
        ).squeeze(-1).reshape(logits.shape[:-1])
    else:
        idx = logits.argmax(dim=-1)
    return module._act_bin_centers.to(logits.device)[idx]


def _gmm_decode(module, raw):
    """Flat gmm params ``(..., M*(2D+1))`` -> committed action ``(..., D)``.

    Default (``gmm_sample`` False): pick the argmax-logit component and return
    its mean -- a deterministic single-mode commitment, never an average over
    modes. ``gmm_sample`` True: draw a component ~ Categorical(softmax(logits)),
    then draw a_d ~ N(mu_md, sigma_md) for that component (reparameterised).
    """
    means, log_stds, logits = _gmm_split(module, raw)  # (...,M,D),(...,M,D),(...,M)
    lead = logits.shape[:-1]
    M, D = module._act_gmm_modes, module._act_dim
    if not getattr(module, "_act_gmm_sample", False):
        comp = logits.argmax(dim=-1)  # (...,) highest-weight mode
    else:
        probs = nn.functional.softmax(logits, dim=-1)  # (...,M)
        comp = torch.multinomial(
            probs.reshape(-1, M), num_samples=1
        ).squeeze(-1).reshape(lead)  # (...,)
    sel = comp.reshape(*lead, 1, 1).expand(*lead, 1, D)  # gather index (...,1,D)
    mean = means.gather(-2, sel).squeeze(-2)  # (...,D)
    if not getattr(module, "_act_gmm_sample", False):
        return mean
    log_std = log_stds.gather(-2, sel).squeeze(-2)  # (...,D)
    eps = torch.randn_like(mean)
    return mean + eps * torch.exp(log_std)


def action_head_decode_chunk(module, raw):
    """Decode the FULL ``chunk_k`` plan to ``(..., K, action_dim)``.

    Used by the chunk_k>1 one-shot ``generate`` path (chunk + temporal
    ensemble), where ALL K steps are consumed, not just k=0. ``raw`` is the
    chunk head output carrying the K axis:
      * continuous -> ``(..., K, action_dim)`` (identity passthrough, matching
        the old explicit ``.reshape(B, K, action_dim)``),
      * discrete   -> ``(..., K, action_dim, n_bins)`` -> per-step de-bin,
      * gmm        -> ``(..., K, M*(2D+1))`` -> per-step mode-committed decode.
    Each chunk step is decoded independently with the same mode-committing
    rule as ``action_head_decode``.
    """
    if getattr(module, "_act_gmm", False):
        return _gmm_decode(module, raw)  # operates per-step over the K axis
    if module._act_disc:
        logits = raw  # (..., K, action_dim, n_bins)
        temp = module._act_temp
        if temp and temp > 0:
            probs = nn.functional.softmax(logits / temp, dim=-1)
            idx = torch.multinomial(
                probs.reshape(-1, module._act_nbins), num_samples=1
            ).squeeze(-1).reshape(logits.shape[:-1])
        else:
            idx = logits.argmax(dim=-1)
        return module._act_bin_centers.to(logits.device)[idx]
    return raw  # continuous: (..., K, action_dim) identity


def _to_bins(module, target):
    """Continuous normalized action ``(..., action_dim)`` -> bin index (long)."""
    t = target.clamp(module._act_bmin, module._act_bmax)
    frac = (t - module._act_bmin) / (module._act_bmax - module._act_bmin)
    return (frac * module._act_nbins).long().clamp(0, module._act_nbins - 1)
