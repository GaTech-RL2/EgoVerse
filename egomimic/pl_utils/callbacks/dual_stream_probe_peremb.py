"""Per-embodiment grad-attribution probe for the dual-stream H-Net.

Superset of ``DualStreamProbeCallback``. Adds PER-EMBODIMENT grad norms on
every module group -- including the SHARED agnostic/specific trunks -- to
diagnose why the agnostic stream is favored.

Mechanism (all callback-side; pl_model.py / algo.py stay byte-identical):
  * Every step (cheap): combined per-stream grad norms in ``on_after_backward``
    (``Train/grad_norm_{trunk_A,trunk_S,enc_A,enc_S,head_A,head_S,gate,A,S}``)
    + mode-sharing ``Train/wA_<emb>`` / ``Train/wS_<emb>``.
  * Every ``probe_every`` steps (in ``on_train_batch_end``): re-run
    ``forward_training`` ONCE on the same batch, then backprop EACH embodiment's
    action loss SEPARATELY (retain_graph). A single-emb backward isolates that
    emb's gradient contribution onto ALL params (shared trunks included), so we
    read per-(emb,group) grad norms. Logged as ``Train/gE_emb{id}/{group}``
    with group in {trunk_A,trunk_S,enc_A,enc_S,head_A,head_S,gate,A,S}.
    The GMM head is split into proj_A (agnostic modes) / proj_S (specific modes)
    / gate -- the routing knob behind agnostic dominance.

Wire: ``+callbacks.ds_probe._target_=egomimic.pl_utils.callbacks.dual_stream_probe_peremb.DualStreamPerEmbProbeCallback``
  optional ``+callbacks.ds_probe.probe_every=20``
"""

import torch
from lightning.pytorch.callbacks import Callback


class DualStreamPerEmbProbeCallback(Callback):
    def __init__(self, probe_every: int = 20):
        super().__init__()
        self.probe_every = max(1, int(probe_every))

    # ---------- classification ----------
    @staticmethod
    def _stream(name: str):
        for seg in name.split("."):
            if seg.endswith("_A"):
                return "A"
            if seg.endswith("_S"):
                return "S"
        return None

    @classmethod
    def _classify(cls, name: str):
        """(component, stream); component None => skip. GMM head split proj_A/proj_S/gate."""
        if "agnostic_input" in name:
            return "enc", "A"
        if "input_modules" in name:
            return "enc", "S"
        if "inner_stage" in name or ".trunk." in name:
            return "trunk", cls._stream(name)
        if "action_out" in name:
            if "gate" in name:
                return "gate", None  # the agnostic-vs-specific routing gate
            return "head", cls._stream(name)  # proj_A (agn modes) / proj_S (spec modes)
        return None, None

    @classmethod
    def _bucket_grads(cls, pl_module):
        """sqrt(sum gradnorm^2) per (component[_stream]) + per-stream totals, from CURRENT grads."""
        sq: dict = {}
        for name, p in pl_module.named_parameters():
            if p.grad is None:
                continue
            comp, st = cls._classify(name)
            if comp is None:
                continue
            g2 = float(p.grad.detach().norm(2)) ** 2
            key = comp if st is None else f"{comp}_{st}"
            sq[key] = sq.get(key, 0.0) + g2
            if st is not None:
                sq[st] = sq.get(st, 0.0) + g2  # stream total A / S
        return {k: v**0.5 for k, v in sq.items()}

    # ---------- combined per-stream grad norms (every step) ----------
    def on_after_backward(self, trainer, pl_module):
        for key, v in self._bucket_grads(pl_module).items():
            pl_module.log(
                f"Train/grad_norm_{key}",
                v,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
        for m in pl_module.modules():
            if getattr(m, "_fired", False):
                emb = getattr(m, "_emb_id", "x")
                pl_module.log(
                    f"Train/wA_{emb}",
                    float(getattr(m, "_last_w_A", 0.0)),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
                pl_module.log(
                    f"Train/wS_{emb}",
                    float(getattr(m, "_last_w_S", 0.0)),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
                m._fired = False

    # ---------- per-emb grad attribution (every probe_every steps) ----------
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (int(pl_module.global_step) % self.probe_every) != 0:
            return
        try:
            self._per_emb_probe(pl_module, batch)
        except Exception as e:  # never let a diagnostic kill training
            if getattr(trainer, "is_global_zero", True):
                print(
                    f"[PEREMB_PROBE] skipped step={pl_module.global_step}: {type(e).__name__}: {e}",
                    flush=True,
                )
        finally:
            pl_module.zero_grad(set_to_none=True)

    def _per_emb_probe(self, pl_module, batch):
        model = pl_module.model
        # Re-run forward once (matches training bf16 autocast); process_batch is non-mutating.
        b = model.process_batch_for_training(batch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            preds = model.forward_training(b)
        emb_losses = {
            k[: -len("_action_loss")]: v
            for k, v in preds.items()
            if k.endswith("_action_loss") and torch.is_tensor(v) and v.requires_grad
        }
        keys = list(emb_losses)
        for i, emb in enumerate(keys):
            pl_module.zero_grad(set_to_none=True)
            emb_losses[emb].backward(retain_graph=(i < len(keys) - 1))
            for key, v in self._bucket_grads(pl_module).items():
                pl_module.log(
                    f"Train/gE_emb{emb}/{key}",
                    v,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
        pl_module.zero_grad(set_to_none=True)
