"""Live training probes for the dual-stream H-Net (logged to ``Train/*``).

Two mechanism metrics, computed WITHOUT touching shared training code
(pl_model / algo.py stay byte-identical to EgoVerse-gmm):

  * **Per-stream grad norms** — raw L2 grad norm of the AGNOSTIC-stream params
    (agnostic encoder + per-head ``proj_A``/``trunk_A``) vs the SPECIFIC-stream
    params (per-emb encoders + ``proj_S``/``trunk_S``) vs the shared TRUNK
    (``inner_stage``). Taken in ``on_after_backward`` (callbacks run BEFORE the
    LightningModule hook, so grads are pre-clip), mirroring pl_model's global
    ``policy_grad_norms_raw``. Logged as ``Train/grad_norm_{A,S,trunk}``.

  * **Mode weighting** — the GMM gate's mixture mass on the agnostic modes
    (``w_A``) vs the specific modes (``w_S``), PER embodiment. The
    ``PartitionedGMMActionHead`` stashes ``_last_w_A`` / ``_last_w_S`` / ``_fired``
    each forward; ``DualStreamOuterStage.decode`` tags the active head with
    ``_emb_id``. We read + clear them here. Logged as ``Train/wA_<emb>`` /
    ``Train/wS_<emb>``.

Wire in: ``+callbacks.ds_probe._target_=egomimic.pl_utils.callbacks.dual_stream_probe.DualStreamProbeCallback``
"""

from lightning.pytorch.callbacks import Callback


class DualStreamProbeCallback(Callback):
    @staticmethod
    def _group(name: str):
        if "agnostic_input" in name or "proj_A" in name or "trunk_A" in name:
            return "A"
        if "input_modules" in name or "proj_S" in name or "trunk_S" in name:
            return "S"
        if "inner_stage" in name:
            return "trunk"
        return None

    def on_after_backward(self, trainer, pl_module):
        # per-stream raw grad norm = sqrt(sum of per-leaf grad-norm^2 in the group)
        sq = {"A": 0.0, "S": 0.0, "trunk": 0.0}
        for name, p in pl_module.named_parameters():
            if p.grad is None:
                continue
            g = self._group(name)
            if g is None:
                continue
            sq[g] += float(p.grad.detach().norm(2)) ** 2
        for g, v in sq.items():
            pl_module.log(
                f"Train/grad_norm_{g}",
                v**0.5,
                on_step=False,
                on_epoch=True,
                sync_dist=False,  # per-rank diagnostic; no DDP allreduce (avoids key-mismatch hang)
            )
        # mode weighting from the head(s) that fired this step (per embodiment)
        for m in pl_module.modules():
            if getattr(m, "_fired", False):
                emb = getattr(m, "_emb_id", "x")
                pl_module.log(
                    f"Train/wA_{emb}",
                    float(getattr(m, "_last_w_A", 0.0)),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,  # per-rank; dynamic per-emb key must NOT allreduce
                )
                pl_module.log(
                    f"Train/wS_{emb}",
                    float(getattr(m, "_last_w_S", 0.0)),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
                m._fired = False
