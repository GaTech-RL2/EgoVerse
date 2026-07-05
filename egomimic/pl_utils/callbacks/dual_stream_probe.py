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
        """Return 'A' (agnostic), 'S' (specific), or None (skip), handling every
        dual-stream naming convention:
          * old two-trunk: ``_A``/``_S`` suffix on a segment (qkv_A, mlp_S, …)
          * new partitioned head: ``A_head`` (agnostic decode) / ``S_head``
            (specific) / ``gate`` (mixture selector -> skip)
          * multi_stream_trunk: per-stream INDEX after intra/inter/norm_f
            (0 = agnostic, >=1 = specific); ``inter`` only targets specific (asym)
          * per-embodiment modules (``PerEmb.table.<emb>``) are specific by build
        """
        segs = name.split(".")
        # 0) apex / any single-stream agnostic ComputeStage uses ``main_network`` and
        #    has NO A/S naming -> it IS the agnostic trunk.
        if "main_network" in segs:
            return "A"
        # 1) explicit _A/_S suffix (old two-trunk + chunker A/S path attrs)
        for seg in segs:
            if seg.endswith("_A"):
                return "A"
            if seg.endswith("_S"):
                return "S"
        # router projections (shared agnostic proj_a / per-emb proj_s)
        if "proj_a" in segs:
            return "A"
        if "proj_s" in segs:
            return "S"
        # 2) new partitioned head (check before .table since S_head/gate are per-emb)
        if "A_head" in segs:
            return "A"
        if "S_head" in segs:
            return "S"
        if "gate" in segs:
            return None  # mixture gate reads both streams -> not an A/S bucket
        # 3) multi_stream_trunk per-stream index
        for key in ("intra", "inter", "norm_f"):
            if key in segs:
                j = segs.index(key)
                if j + 1 < len(segs) and segs[j + 1].isdigit():
                    return "A" if int(segs[j + 1]) == 0 else "S"
                if key == "inter":
                    return "S"  # inter only writes specific streams (asym)
        # 4) any remaining per-embodiment module is specific
        if "table" in segs:
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
        # head BEFORE trunk: the new DualPartitionedGMMHead's _ModeHead has its own
        # internal ``.trunk.`` MLP, so it would otherwise be mis-bucketed as trunk.
        if "action_out" in name:
            return "head", cls._stream(
                name
            )  # A_head/S_head (proj_A/proj_S); gate -> None
        if "inner_stage" in name or ".trunk." in name:
            return "trunk", cls._stream(name)  # trunk_A vs trunk_S (separate weights)
        return None, None

    @staticmethod
    def _emb_of(segs):
        """The embodiment/domain name for a per-emb (specific) param, or None.
        Per-emb modules live under ``...table.<emb>...`` (PerEmb) or
        ``...encoders.<emb>...`` (MultiEmbodimentCondEncoder)."""
        for marker in ("table", "encoders"):
            if marker in segs:
                i = segs.index(marker)
                if i + 1 < len(segs):
                    return segs[i + 1]
        return None

    @staticmethod
    def _layer(segs):
        """Compact transformer-block tag ``s<stage>b<block>`` (e.g. 's0b3'), or
        None for non-block params (encoders, chunker combine/proj/router, norms).
        Catches both dual compute (``...stages.<j>.trunk.blocks.<i>...``) and the
        apex (``...stages.<j>.main_network.blocks.<i>...``)."""
        if "blocks" not in segs:
            return None
        bi = segs.index("blocks")
        if bi + 1 >= len(segs) or not segs[bi + 1].isdigit():
            return None
        stage = None
        if "stages" in segs:
            si = segs.index("stages")
            if si + 1 < len(segs) and segs[si + 1].isdigit():
                stage = segs[si + 1]
        return f"s{stage}b{segs[bi + 1]}" if stage is not None else f"b{segs[bi + 1]}"

    def on_after_backward(self, trainer, pl_module):
        # Per-bucket raw grad norm = sqrt(sum of per-leaf grad-norm^2).
        # A (agnostic) is weight-shared -> grad is COMBINED across embodiments
        # (cotrain sums both losses before one backward). S (specific) params are
        # per-emb -> each emb's grad is clean, so we ALSO emit per-emb S buckets.
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
            segs = name.split(".")
            emb = self._emb_of(segs)
            if emb is not None:  # per-emb (specific) -> per-embodiment buckets
                sq[f"{comp}_{st}_{emb}"] = sq.get(f"{comp}_{st}_{emb}", 0.0) + g2
                sq[f"{st}_{emb}"] = sq.get(f"{st}_{emb}", 0.0) + g2
            lyr = self._layer(segs)  # PER-LAYER (transformer block) A/S grad norm
            if lyr is not None:
                sq[f"L_{lyr}_{st}"] = sq.get(f"L_{lyr}_{st}", 0.0) + g2
                if emb is not None:  # per-layer, per-emb (specific stream)
                    sq[f"L_{lyr}_{st}_{emb}"] = sq.get(f"L_{lyr}_{st}_{emb}", 0.0) + g2
        for key, v in sq.items():
            pl_module.log(
                f"Train/grad_norm_{key}",
                v**0.5,
                on_step=False,
                on_epoch=True,
                sync_dist=False,  # per-rank diagnostic; no DDP allreduce
            )
        # Mode sharing (agnostic vs specific mixture mass), PER embodiment.
        for m in pl_module.modules():
            wbe = getattr(m, "_w_by_emb", None)
            if wbe:  # single shared head dispatched over all embodiments
                for emb, (wa, ws) in wbe.items():
                    pl_module.log(
                        f"Train/wA_{emb}",
                        float(wa),
                        on_step=False,
                        on_epoch=True,
                        sync_dist=False,
                    )
                    pl_module.log(
                        f"Train/wS_{emb}",
                        float(ws),
                        on_step=False,
                        on_epoch=True,
                        sync_dist=False,
                    )
                m._w_by_emb = {}
                m._fired = False
            elif getattr(m, "_fired", False):  # BC: per-emb ModuleDict heads
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
