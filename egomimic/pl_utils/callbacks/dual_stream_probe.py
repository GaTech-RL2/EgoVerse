"""Live training probes for the dual-stream H-Net (logged to ``Train/*``).

Two mechanism metrics, computed WITHOUT touching shared training code
(pl_model / algo.py stay byte-identical to EgoVerse-gmm):

  * **Per-trunk / per-stream grad norms** — raw L2 grad norm bucketed by the
    v2 TWO-TRUNK structure. Each trunk param name carries an ``_A`` (agnostic
    stream) or ``_S`` (specific stream) suffix on a module segment
    (``attn.qkv_A``, ``mlp_S``, ``norm1_A``, ``norm_f_S`` …), so we split the
    trunk into ``trunk_A`` vs ``trunk_S`` — the headline "per-trunk grad norm".
    Also bucketed: ``enc_A`` (shared agnostic encoder ``agnostic_input``) vs
    ``enc_S`` (per-emb encoders ``input_modules``), ``head_A``/``head_S``
    (partitioned-head ``proj_A``/``proj_S``), and stream totals ``A``/``S``.
    Taken in ``on_after_backward`` (callbacks run BEFORE the LightningModule
    hook → grads are pre-clip), mirroring pl_model's ``policy_grad_norms_raw``.
    Logged as ``Train/grad_norm_{trunk_A,trunk_S,enc_A,enc_S,head_A,head_S,A,S}``.

  * **Mode sharing** — the GMM gate's mixture mass on the agnostic (shared)
    modes (``w_A``) vs the specific modes (``w_S``), PER embodiment. The
    ``PartitionedGMMActionHead`` stashes ``_last_w_A`` / ``_last_w_S`` /
    ``_fired`` each forward; ``DualStreamOuterStage.decode`` tags the active
    head with ``_emb_id``. We read + clear them here. Logged as
    ``Train/wA_<emb>`` / ``Train/wS_<emb>``.

Wire in: ``+callbacks.ds_probe._target_=egomimic.pl_utils.callbacks.dual_stream_probe.DualStreamProbeCallback``
"""

from lightning.pytorch.callbacks import Callback


class DualStreamProbeCallback(Callback):
    @staticmethod
    def _stream(name: str):
        """'A'/'S' from an ``_A``/``_S`` suffix on any module segment, else None."""
        for seg in name.split("."):
            if seg.endswith("_A"):
                return "A"
            if seg.endswith("_S"):
                return "S"
        return None

    @classmethod
    def _classify(cls, name: str):
        """Return (component, stream) e.g. ('trunk','A'); (None,None) to skip.

        Component-first so encoders (which have no _A/_S suffix) map by module
        path; trunk/head streams come from the _A/_S suffix.
        """
        if "agnostic_input" in name:
            return "enc", "A"  # weight-shared agnostic obs encoder
        if "input_modules" in name:
            return "enc", "S"  # per-embodiment specific obs encoders
        if "inner_stage" in name or ".trunk." in name:
            return "trunk", cls._stream(name)  # trunk_A vs trunk_S (separate weights)
        if "action_out" in name:
            return "head", cls._stream(name)  # proj_A/proj_S; shared gate -> None
        return None, None

    def on_after_backward(self, trainer, pl_module):
        # Per-bucket raw grad norm = sqrt(sum of per-leaf grad-norm^2).
        sq: dict = {}
        for name, p in pl_module.named_parameters():
            if p.grad is None:
                continue
            comp, st = self._classify(name)
            if comp is None or st is None:
                continue  # skip unclassified / shared (e.g. gate) params
            g2 = float(p.grad.detach().norm(2)) ** 2
            sq[f"{comp}_{st}"] = sq.get(f"{comp}_{st}", 0.0) + g2  # trunk_A, enc_S, ...
            sq[st] = sq.get(st, 0.0) + g2  # stream total A / S
        for key, v in sq.items():
            pl_module.log(
                f"Train/grad_norm_{key}",
                v**0.5,
                on_step=False,
                on_epoch=True,
                sync_dist=False,  # per-rank diagnostic; no DDP allreduce
            )
        # Mode sharing from the head(s) that fired this step (per embodiment).
        for m in pl_module.modules():
            if getattr(m, "_fired", False):
                emb = getattr(m, "_emb_id", "x")
                pl_module.log(
                    f"Train/wA_{emb}",
                    float(getattr(m, "_last_w_A", 0.0)),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,  # dynamic per-emb key must NOT allreduce
                )
                pl_module.log(
                    f"Train/wS_{emb}",
                    float(getattr(m, "_last_w_S", 0.0)),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
                m._fired = False
