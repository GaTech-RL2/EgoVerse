"""PipelineAlgo — the BATCHFLOW runner.

Satisfies the exact three-method contract pl_model.ModelWrapper expects
(process_batch_for_training / forward_training / compute_losses) so the
lightning glue is untouched. The model is ONE Pipeline of stages; losses are
stages; the runner:

  * seeds a FLAT batch dict per embodiment (obs/* keys, actions, packing
    meta, "embodiment", the aux/* accumulators),
  * runs the pipeline,
  * sums loss/* into the training loss,
  * prefixes log/* into per-emb metric keys (one naming scheme for every
    module: Train/{emb}_{name}).

Rollout (inference_step) runs the SAME pipeline on an obs-only dict resolved
by Pipeline.plan() — posterior/loss stages are provably excluded.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from typing import List

import torch
import torch.nn as nn

from egomimic.algo.algo import Algo
from egomimic.eval.inference_graph import (
    ActionCacheState,
    InferenceGraph,
    KeyedNode,
)
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.pipeline.core import Pipeline, Stage, sum_losses
from egomimic.pipeline.stages_hnet import ApexLevel

_PACKED_META_KEYS = (
    "cu_seqlens", "max_seq_len", "seq_lens", "batch_size",
    "embodiment", "episode_idx", "chunk_offset",
)

# Explicitly retained after the norm-stats key mapping so evaluation can use
# per-episode camera calibration and invert wrist-frame targets for overlays.
# Do not replace this allowlist with keep_unmapped=True: raw dataset scratch
# keys must not leak into the policy batch.
_VIZ_PASSTHROUGH_KEYS = (
    "front_intrinsics",
    "left_camera_extrinsics",
    "right_camera_extrinsics",
    "viz_current_wrist_poses",
)


class PipelineAlgo(Algo):
    def __init__(
        self,
        stages: List[Stage],
        norm_stats,
        domains: list = None,
        ac_keys: dict = None,
        auxiliary_ac_keys: dict = None,
        device=None,
        action_horizon: int = 2560,
        train_obs_transforms: list | None = None,
        episode_level_transforms: list | None = None,
        init_ckpt: str | None = None,
        rollout_apex_mode: str = "configured",
        rollout_apex_window: int | None = None,
        inference_stages: dict | None = None,
        # Default OFF: the residual-aware init pass was dead code until
        # 2026-08-18, so every existing config trained without it. Opting
        # in per-config keeps other experiments' fresh launches unchanged.
        init_range: float | None = None,
        lr_multipliers: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            # no dead knobs: unknown config keys fail loudly.
            raise TypeError(f"PipelineAlgo got unknown config keys: {sorted(kwargs)}")
        self.norm_stats = norm_stats
        self.domains = list(domains or [])
        self.domain_by_id = {get_embodiment_id(e): e for e in self.domains}
        self.ac_keys = dict(ac_keys or {})
        # Evaluators written against the older Algo base read this (5 sites in
        # eval/hpt/eval_hpt.py alone, which is the DEFAULT evaluator). Without it
        # every PipelineAlgo run raised AttributeError at its first validation.
        # Optional ctor arg rather than a hardcoded {} so a config can populate it.
        self.auxiliary_ac_keys = dict(auxiliary_ac_keys or {})
        self.action_horizon = int(action_horizon)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.train_obs_transforms = list(train_obs_transforms or [])
        self.episode_level_transforms = list(episode_level_transforms or [])
        self.rollout_apex_mode = str(rollout_apex_mode)
        self.inference_stages = inference_stages
        self.rollout_apex_window = (
            int(rollout_apex_window)
            if rollout_apex_window is not None
            else None
        )
        self._resolve_embodiment_keys(norm_stats)
        self.nets = nn.ModuleDict({"policy": Pipeline(list(stages))})
        # RESIDUAL-STREAM-AWARE INIT (2026-08-18). The flat pipeline defined
        # `_init_weights` on every stage but NOTHING ever called it, so the
        # whole seq lineage trained from PyTorch's default kaiming init with no
        # `1/sqrt(n_residuals)` damping -- unlike the nested lineage, whose
        # container applies it by default (stages_hnet.py:1211,1223, init_range
        # 0.02). This walks the flat stage list threading the cumulative
        # residual depth, exactly as the nested chain threads it through
        # `inner`. Set `init_range: null` to restore the un-initialized
        # behaviour of runs launched before this date.
        #
        # Ordering matters: this runs BEFORE `init_ckpt` / any checkpoint
        # restore, so a resume still ends up with the checkpoint's weights.
        self.init_range = None if init_range is None else float(init_range)
        self.lr_multipliers = dict(lr_multipliers) if lr_multipliers else None
        if self.init_range:
            self._apply_residual_aware_init(self.init_range)
        if init_ckpt:
            _ck = torch.load(init_ckpt, map_location="cpu", weights_only=False)
            _sd = _ck.get("state_dict", _ck)
            _new = {}
            for _k, _v in _sd.items():
                for _pfx in ("nets.", "model.nets."):
                    if _k.startswith(_pfx):
                        _new[_k[len(_pfx):]] = _v
                        break
                else:
                    _new[_k] = _v
            _miss, _unexp = self.nets.load_state_dict(_new, strict=False)
            print(f"[init_ckpt] {init_ckpt}: missing={len(_miss)} unexpected={len(_unexp)}")
            assert not _miss, f"init_ckpt missing keys: {_miss[:5]}"

        self.nets.to(self.device)
        # replan cadence for closed-loop AR = the TargetBuilder's stride
        # (introspected -> train/inference cadence can never desync).
        self.replan_stride = 1
        for s in self.nets["policy"].stages:
            if type(s).__name__ == "TargetBuilder":
                self.replan_stride = int(s.stride)
        # The evaluator-facing controller is the same keyed three-node graph
        # used by DF. Model/history layout remains H-Net-specific and lives in
        # the bound node methods below; weights and stages still come from the
        # loaded checkpoint pipeline.
        self._sim_action_cache = ActionCacheState()
        self._sim_action_queue = self._sim_action_cache.actions
        self._inference_graph = self._build_inference_graph()

    # ------------------------------------------------------------------ #
    def _apply_residual_aware_init(self, init_range: float) -> None:
        """Thread `_init_weights(init_range, parent_residuals)` down the stages.

        A stage returns the cumulative residual depth AFTER itself, so a trunk
        adds 2 per layer, a Dechunk adds 1, and a Chunk adds 0 -- the same
        accounting the nested chain does via `inner`. Stages without the method
        (obs encoders, heads, loss stages) self-init in their own ctors and are
        skipped, which is also what the nested container does.
        """
        import inspect

        n_residuals = 0
        touched, skipped = [], []
        for stage in self.nets["policy"].stages:
            fn = getattr(stage, "_init_weights", None)
            if not callable(fn):
                continue
            # SIGNATURE GUARD. Not every `_init_weights` in this codebase takes
            # (range, parent_residuals): `stages_dfot_v3.py:390` defines a
            # zero-arg `_init_weights(self)` that its own ctor already called.
            # Calling that here would TypeError and kill the job at
            # construction -- including a requeue of a live DFoT run. Only
            # stages with the 2-arg chain signature participate.
            try:
                params = [
                    p for p in inspect.signature(fn).parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                ]
            except (TypeError, ValueError):
                params = []
            if len(params) != 2:
                skipped.append(type(stage).__name__)
                continue
            out = fn(init_range, n_residuals)
            if not isinstance(out, int):
                raise TypeError(
                    f"{type(stage).__name__}._init_weights must return the "
                    f"cumulative residual count (int), got {out!r}. Returning "
                    "None here is what silently broke this chain before.")
            n_residuals = out
            touched.append(f"{type(stage).__name__}->{n_residuals}")
        print(f"[init] residual-aware init range={init_range} "
              f"final_depth={n_residuals} inited={len(touched)} "
              f"skipped={sorted(set(skipped)) or 'none'} :: "
              + ", ".join(touched))

    def parameter_groups(self, base_lr: float):
        """Per-stage LR multipliers -> AdamW param groups (opt-in).

        `lr_multipliers` is `{glob-over-parameter-name: multiplier}`, e.g.
        `{"policy.stages.10.*": 0.5}`. Matching uses fnmatch over
        `named_parameters()` -- the same convention as
        `egomimic.pl_utils.param_groups`. Returning None (the default, when no
        multipliers are configured) makes `pl_model.configure_optimizers` fall
        back to flat `parameters()`, so this is byte-identical for every
        existing run and does NOT change the optimizer state layout they resume
        into.
        """
        if not self.lr_multipliers:
            return None
        import fnmatch

        buckets: OrderedDict = OrderedDict()
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            mult = 1.0
            for pat, m in self.lr_multipliers.items():
                if fnmatch.fnmatch(name, str(pat)):
                    mult = float(m)
                    break
            buckets.setdefault(mult, []).append(p)
        groups = [{"params": ps, "lr": float(base_lr) * mult}
                  for mult, ps in buckets.items()]
        print("[lr] per-stage multipliers -> %d groups: %s"
              % (len(groups), {m: len(ps) for m, ps in buckets.items()}))
        return groups

    @property
    def policy(self) -> Pipeline:
        if hasattr(self, "nets") and "policy" in self.nets:
            return self.nets["policy"]
        if hasattr(self, "_rollout_policy"):
            return self._rollout_policy
        raise AttributeError("PipelineAlgo has no policy pipeline")

    @policy.setter
    def policy(self, value: Pipeline) -> None:
        """Install a pipeline without requiring the full training constructor.

        Small rollout adapters and contract tests build only the deployment
        surface.  Keeping this setter makes that surface use the same
        ``nets['policy']`` ownership as production checkpoints instead of a
        second shadow attribute.
        """
        if not isinstance(value, Pipeline):
            raise TypeError(
                f"policy must be a Pipeline, got {type(value).__name__}")
        if not hasattr(self, "nets"):
            self.nets = nn.ModuleDict()
        self.nets["policy"] = value
        # Retain the deployment pipeline if a lightweight adapter later
        # replaces ``nets`` with a model-only ModuleDict.
        self._rollout_policy = value

    def _normal_rollout_adapter(self):
        """Return the fixed-history adapter, if this policy owns one.

        Graph-only adapters may override ``step`` and intentionally omit a
        training pipeline.  In that case there is no normal-history adapter;
        cache/preprocess still remain valid and model-specific ``step`` owns
        its request layout.
        """
        try:
            policy = self.policy
        except AttributeError:
            return None
        return next(
            (stage for stage in policy.stages
             if getattr(stage, "rollout_obs_steps", None) is not None),
            None,
        )

    def apex_levels(self) -> list[ApexLevel]:
        return [m for m in self.policy.modules() if isinstance(m, ApexLevel)]

    def set_apex_attention_mode(
        self,
        mode: str,
        *,
        window: int | None = None,
    ) -> None:
        apexes = self.apex_levels()
        if not apexes:
            raise RuntimeError("PipelineAlgo has no ApexLevel")
        for apex in apexes:
            apex.set_attention_mode(mode, window=window)

    def activate_rollout_apex_attention(self) -> None:
        """Apply the CONFIGURED rollout apex regime at episode start.

        This runs from init_step_state on every episode and OVERRIDES any mode
        set earlier (e.g. by an eval CLI flag). It is logged once per process
        because a silent override here made rollouts look identical across
        regimes and cost hours of misdiagnosis (2026-07-31).
        """
        if self.rollout_apex_mode.lower() == "configured":
            if not getattr(self, "_apex_regime_logged", False):
                modes = {a.attention_mode for a in self.apex_levels()}
                print("[apex/rollout] regime=CONFIGURED (from the model config); "
                      "live apex mode(s)=%s" % sorted(modes))
                self._apex_regime_logged = True
            return
        self.set_apex_attention_mode(
            self.rollout_apex_mode,
            window=self.rollout_apex_window,
        )
        if not getattr(self, "_apex_regime_logged", False):
            print("[apex/rollout] regime=%s window=%s APPLIED at episode start "
                  "(overrides any earlier set_attention_mode)"
                  % (self.rollout_apex_mode, self.rollout_apex_window))
            self._apex_regime_logged = True

    def _seed(self, emb_id: int, _batch: dict) -> dict:
        """Flat batch dict for one embodiment (the ONE carrier)."""
        obs = self._build_obs(_batch, emb_id)
        actions = _batch[self.resolved_ac_keys[emb_id]]
        # The NORMAL (per-sample) reader carries no packing metadata: one sample
        # per frame, so seed a 1-token-per-sample grid. NormalObsExpand rewrites
        # it to the real obs-history grid as the first stage. The packed reader
        # supplies both keys and is untouched.
        cu = _batch.get("cu_seqlens")
        if cu is None:
            n = int(actions.shape[0])
            cu = torch.arange(0, n + 1, device=actions.device, dtype=torch.long)
            max_seq_len = 1
        else:
            max_seq_len = int(_batch["max_seq_len"])
        b = {
            "actions": actions,
            "cu_seqlens": cu,
            "max_seq_len": max_seq_len,
            "embodiment": self.domain_by_id.get(emb_id),
            "aux/chunker": [],
        }
        for k, v in obs.items():
            b[f"obs/{k}"] = v
        return b

    # ------------------------------------------------------------------ #
    # pl_model contract
    # ------------------------------------------------------------------ #
    def process_batch_for_training(self, batch):
        processed = {}
        for emb_name, _batch in batch.items():
            emb_id = get_embodiment_id(emb_name)
            if (self.episode_level_transforms and self.policy.training
                    and "cu_seqlens" in _batch):
                from egomimic.algo.hnet.episode_transforms import (
                    apply_episode_level_transforms,
                )
                _batch = apply_episode_level_transforms(
                    _batch, self.episode_level_transforms)
            out, _ = self._prepare_loader_batch(
                _batch, emb_id, packed_meta_keys=_PACKED_META_KEYS)
            for key in _VIZ_PASSTHROUGH_KEYS:
                if key in _batch:
                    out[key] = _batch[key]
            if self.train_obs_transforms and self.policy.training:
                for t in self.train_obs_transforms:
                    out = t(out)
            for key, value in out.items():
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    if value.is_floating_point():
                        value = value.float()
                    out[key] = value
            processed[emb_id] = out
        return processed

    def forward_training(self, batch):
        predictions = OrderedDict()
        for emb_id, _batch in batch.items():
            b = self.policy(self._seed(emb_id, _batch))
            total = sum_losses(b)
            dev = total.device
            predictions[f"{emb_id}_action_loss"] = total
            for k, v in b.items():
                if k.startswith("loss/"):
                    predictions[f"{emb_id}_{k[5:].replace('/', '_')}_loss"] = (
                        v.detach() if torch.is_tensor(v) else torch.tensor(float(v), device=dev))
                elif k.startswith("log/"):
                    predictions[f"{emb_id}_{k[4:].replace('/', '_')}"] = torch.tensor(
                        float(v), device=dev)
        return predictions

    def compute_losses(self, predictions, batch):
        losses = OrderedDict()
        embs = list(batch.keys())
        total = None
        for e in embs:
            t = predictions[f"{e}_action_loss"]
            total = t if total is None else total + t
        losses["action_loss"] = total / max(len(embs), 1)
        for k, v in predictions.items():
            if torch.is_tensor(v) and v.dim() == 0:
                losses[k] = v
        return losses

    def collect_chunkviz(self, batch):
        """Chunkviz interface parity with the old algo (user rule 2026-07-04).

        Returns {emb_id: {"levels": [(prob_packed, mask_packed), ...bottom->TOP],
                          "tokens": top-level tokens ndarray | None}}.
        Flattened chunk/L{i} records are DEEPEST-FIRST (inner recursion appends
        before the outer level), so reverse to put the OUTER level LAST as
        per_frame_chunk_ids expects.
        """
        out = {}
        for emb_id, _b in batch.items():
            if not _b.get("_packed", False):
                continue
            seed = self._seed(emb_id, _b)
            seed = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                    for k, v in seed.items()}
            # capture the RAW frame cu BEFORE the forward: TargetBuilder
            # rebinds seed["cu_seqlens"] to the decimated grid inside.
            raw_cu = seed["cu_seqlens"].cpu()
            gt_actions = seed["actions"].detach().float().cpu()
            seed["aux/trunk_enc"] = []          # opt in to the trunk-level probes
            seed["aux/trunk_dec"] = []
            with torch.no_grad():
                b = self.policy(seed)
            idxs = sorted({int(k.split("/")[1][1:]) for k in b
                           if k.startswith("chunk/L") and k.endswith("/cu_seqlens")})
            levels, best = [], None
            # compose contract: INNERMOST FIRST; flattened records are already
            # deepest-first (chunk/L0 = seam), so ascending order is correct.
            for i in idxs:
                prob = b[f"chunk/L{i}/boundary_prob"]
                mask = b[f"chunk/L{i}/boundary_mask"]
                levels.append((prob[..., 1].detach().float().cpu(),
                               mask.detach().cpu().to(torch.bool)))
            if not levels:
                # SEQ-PIPELINE FALLBACK (2026-08-13): stages_seq publishes
                # aux/chunker records but its graphs have no flattener stage,
                # so no chunk/L* flat keys exist and the scan above finds
                # nothing (export then strips the anchor and crashes on an
                # empty stack). Read the records directly. Compose contract
                # wants INNERMOST FIRST; a record's boundary_mask lives on
                # that level's INPUT grid and inner grids are strictly
                # shorter, so ascending mask length = innermost..outermost.
                for rec in sorted(b.get("aux/chunker") or [],
                                  key=lambda r: int(r["boundary_mask"].shape[0])):
                    prob = rec["boundary_prob"]
                    if prob.dim() > 1 and prob.shape[-1] == 2:
                        prob = prob[..., 1]
                    levels.append((prob.detach().float().cpu(),
                                   rec["boundary_mask"].detach().cpu().to(torch.bool)))
            # OUTERMOST pseudo-level: TargetBuilder's fixed stride decimation
            # exposed on the RAW frame grid so composition can anchor
            # (mask len == T_total). Kept frames: every `stride`-th per episode.
            stride = next((int(st.stride) for st in self.policy.stages
                           if hasattr(st, "stride")), 1)
            # DP CHUNK GRID (2026-08-14): chunkerless graphs (no learned levels)
            # get their FIXED ACTION CHUNK as the display grid -- stride=1 would
            # otherwise mean chunk-per-frame. chunk_len 16 -> 16-frame chunks.
            if not levels:
                stride = max(stride, next(
                    (int(st.chunk_len) for st in self.policy.stages
                     if hasattr(st, "chunk_len")), 1))
            T_raw = int(raw_cu[-1])
            keep = torch.zeros(T_raw, dtype=torch.bool)
            for _e in range(len(raw_cu) - 1):
                keep[int(raw_cu[_e]):int(raw_cu[_e + 1]):stride] = True
            levels.append((keep.float(), keep))
            toks = b.get("apex/tokens")
            if toks is None:
                # CHUNKERLESS FALLBACK (2026-08-13): DP-style graphs have no
                # apex; use the per-frame conditioning features so the PCA
                # panel still shows the model's latents. Decimated below to the
                # anchor chunk grid so PCA rows == chunks (2026-08-14).
                toks = b.get("obs_feat")
                if toks is not None and toks.shape[0] == keep.shape[0]:
                    toks = toks[keep]
            if toks is not None:
                best = toks.detach().float().cpu().numpy()
            entry = {"levels": levels, "tokens": best, "anchor": True,
                     "gt_frame": gt_actions.numpy()}
            # lowest trunk level (L0) A/S streams, on the decimated token grid
            # lowest trunk level = smallest level index. Encoders append
            # outermost-first but decoders append innermost-first, so pick by
            # the tagged index rather than by list position.
            for _key, _recs in (("", b.get("aux/trunk_enc") or []),
                                ("dec", b.get("aux/trunk_dec") or [])):
                if not _recs:
                    continue
                _lvl, _a, _s = min(_recs, key=lambda r: r[0])
                _suf = f"L0{_key}"
                # A single-stream level (DualTrunkLevel stream_keys=[A], as in
                # the StreamMLP arch) records S as None -- that level HAS no S
                # stream. Emit what exists instead of crashing on None.float().
                if _a is not None:
                    entry[f"{_suf}_A"] = _a.float().cpu().numpy()
                if _s is not None:
                    entry[f"{_suf}_S"] = _s.float().cpu().numpy()
            # tile pred chunks (kept grid, chunk C) back to the raw frame grid
            pred = b.get("pred_action")
            if pred is not None:
                import numpy as np
                pred = pred.detach().float().cpu().numpy()  # (T_kept, C, D)
                stride = next((int(st.stride) for st in self.policy.stages
                               if hasattr(st, "stride")), 1)
                kept_cu = b["cu_seqlens"].cpu().numpy()
                T_raw = int(raw_cu[-1])
                pf = np.zeros((T_raw, pred.shape[-1]), dtype=np.float32)
                C = pred.shape[1]
                for e in range(len(raw_cu) - 1):
                    a0, b0 = int(raw_cu[e]), int(raw_cu[e + 1])
                    k0 = int(kept_cu[e])
                    for f in range(a0, b0):
                        j = k0 + (f - a0) // stride
                        j = min(j, int(kept_cu[e + 1]) - 1)
                        off = min((f - a0) % stride, C - 1)
                        pf[f] = pred[j, off]
                entry["pred_frame"] = pf
            out[emb_id] = entry
        return out

    def forward_eval(self, batch):
        """Teacher-forced eval: same pipeline; pred_action is already decoded."""
        predictions = OrderedDict()
        for emb_id, _batch in batch.items():
            b = self.policy(self._seed(emb_id, _batch))
            ac = self.resolved_ac_keys[emb_id]
            pred = b["pred_action"]
            # Embodiment-SCOPED keys. A bare `ac` key is ambiguous in cotrain:
            # with a matched action space both embodiments use the same ac_key,
            # so the second overwrote the first. It also matched no evaluator --
            # HNetEvalVideo reads emb{id}_{ac} and silently skips when absent,
            # which produced empty videos/ and zero Valid/ metrics at exit 0.
            predictions[f"emb{emb_id}_{ac}"] = pred
            name = self.domain_by_id.get(emb_id)
            if name:
                predictions[f"{str(name).lower()}_{ac}"] = pred   # eval_hpt form
            predictions.setdefault(ac, pred)                      # legacy bare
        return predictions

    # ------------------------------------------------------------------ #
    # Rollout — the persistent dict IS the state. plan() resolves the
    # deployed subset once (obs-only seeds exclude posterior/loss stages).
    # ------------------------------------------------------------------ #
    def init_step_state(self, batch_size, T_max, device, dtype):
        assert batch_size == 1, "rollout is batch_size=1 (recompute-over-prefix)"
        self.activate_rollout_apex_attention()
        return {"obs_prefix": [], "device": device, "dtype": dtype, "plan": None}

    @torch.no_grad()
    def step(self, state: dict, obs_norm: dict, t: int, embodiment_id=None,
             obs_norm_history=None):
        state["obs_prefix"].append({k: v for k, v in obs_norm.items()})
        prefix = (list(obs_norm_history) if obs_norm_history is not None
                  else state["obs_prefix"])
        T = len(prefix)
        dev = state["device"]
        normal_adapter = next(
            (stage for stage in self.policy.stages
             if getattr(stage, "rollout_obs_steps", None) is not None),
            None,
        )
        b = {
            # NO "actions" key: plan() then provably excludes TargetBuilder,
            # posterior and every loss stage -> DENSE full-rate prefix, exactly
            # the old rollout semantics (stride cadence lives in the queue).
            "cu_seqlens": torch.tensor([0, T], device=dev, dtype=torch.long),
            "max_seq_len": T,
            "embodiment": embodiment_id,
            "aux/chunker": [],
            # Set before plan/forward: NormalObsExpand uses this explicit marker
            # to distinguish target-free deployment from malformed training.
            "rollout_t": t,
        }
        if normal_adapter is not None:
            # Standard DP was trained on a fixed n-frame sample axis. Recreate
            # that exact shape at rollout: keep the newest n frames and repeat
            # the episode's first frame at the left boundary (DP pad_before).
            n = int(normal_adapter.rollout_obs_steps)
            recent = prefix[-n:]
            frames = [prefix[0]] * (n - len(recent)) + recent
            for k in prefix[0].keys():
                vs = [f[k] for f in frames]
                b[f"obs/{k}"] = (
                    torch.cat(vs, 0).unsqueeze(0).to(dev)
                    if torch.is_tensor(vs[0]) else vs[-1]
                )
        else:
            for k in prefix[0].keys():
                vs = [f[k] for f in prefix]
                # env->zarr frames carry B=1 (e.g. (1,5), (1,3,H,W)) -> cat
                # along dim0 gives the packed (T, ...) layout encoders expect.
                b[f"obs/{k}"] = (torch.cat(vs, 0).to(dev)
                                 if torch.is_tensor(vs[0]) else vs[-1])
        if state["plan"] is None:
            runnable, excluded = self.policy.plan(list(b.keys()))
            state["plan"] = runnable
            if excluded:
                names = [(type(s).__name__, miss) for s, miss in excluded]
                print(f"[PipelineAlgo.step] plan excluded (train-only): {names}")
        for stage in state["plan"]:
            b = stage(b)
        # Packed-prefix models produce T rows; NormalObsCollapse produces one.
        # In both cases the deployable prediction is the final/current row.
        return b["pred_action"][-1]  # (C, D) decoded chunk at current token

    # ------------------------------------------------------------------ #
    # Keyed inference graph. The control topology is universal; these bound
    # nodes own H-Net's packed-prefix layout and checkpoint pipeline call.
    # ------------------------------------------------------------------ #
    def _build_inference_graph(self) -> InferenceGraph:
        defaults = {
            "check_cache": {"in": {"obs": "obs"},
                            "out": {"action": "policy.action"}},
            "inference_preprocess": {
                "in": {"obs": "obs"},
                "out": {"request": "model.request"}},
            "model": {"in": {"request": "model.request"},
                      "out": {"plan": "model.plan"}},
            "update_cache": {
                "in": {"plan": "model.plan", "obs": "obs"},
                "out": {"action": "policy.action"}},
        }
        cfg = self.inference_stages or {}
        nodes = cfg.get("nodes", cfg) if hasattr(cfg, "get") else {}

        def ports(name):
            node = (cfg.get("model", defaults[name]) if name == "model"
                    else nodes.get(name, defaults[name]))
            return {
                "in": dict(node.get("in", defaults[name]["in"])),
                "out": dict(node.get("out", defaults[name]["out"])),
            }

        return InferenceGraph(
            check_cache=KeyedNode(
                self._graph_check_cache, **ports("check_cache")),
            inference_preprocess=KeyedNode(
                self._graph_preprocess, **ports("inference_preprocess")),
            model=KeyedNode(self._graph_model, **ports("model")),
            update_cache=KeyedNode(
                self._graph_update_cache, **ports("update_cache")),
            terminal_key=str(cfg.get("terminal", "policy.action")),
        )

    def _reset_inference_graph(self, T_max=None) -> None:
        param = next(self.nets.parameters())
        self._sim_state = self.init_step_state(
            batch_size=1, T_max=int(T_max or self.action_horizon),
            device=param.device, dtype=param.dtype)
        self._sim_action_cache.reset()
        normal = self._normal_rollout_adapter()
        self._sim_raw_obs_history = deque(
            maxlen=int(normal.rollout_obs_steps) if normal is not None else 1)
        # Compatibility view for old diagnostics. Never replace this list:
        # the ActionCacheState owns it for the lifetime of the policy.
        self._sim_action_queue = self._sim_action_cache.actions
        self._sim_prev_chunk = None
        self._sim_prev_chunk_t = None
        self._sim_ema_a = None
        self._sim_blend_announced = False
        self._sim_ema_announced = False

    def _graph_check_cache(self, obs: dict):
        # Standard DP needs the immediately previous ENV frame at each replan,
        # not the previous model-query frame. Keep a tiny raw ring here; cache
        # hits still skip transforms, normalization, stacking and the model.
        normal = self._normal_rollout_adapter()
        if normal is not None:
            snap = {}
            for key, value in obs.items():
                if torch.is_tensor(value):
                    snap[key] = value.detach().clone()
                elif hasattr(value, "copy"):
                    snap[key] = value.copy()
                else:
                    snap[key] = value
            self._sim_raw_obs_history.append(snap)
        if not self._sim_action_cache:
            return None
        return self._graph_commit(self._sim_action_cache.pop())

    def _graph_preprocess(self, obs: dict) -> dict:
        adapter = getattr(self, "inference_obs_adapter", None)
        normal = self._normal_rollout_adapter()
        if normal is not None:
            raw = list(self._sim_raw_obs_history)
            raw = [raw[0]] * (int(normal.rollout_obs_steps) - len(raw)) + raw
            history = []
            for frame in raw:
                model_obs = adapter(frame) if adapter is not None else frame
                history.append(self.norm_stats.normalize(
                    model_obs, self._sim_emb_id))
            return {"obs_norm": history[-1],
                    "obs_norm_history": history}
        model_obs = adapter(obs) if adapter is not None else obs
        return {"obs_norm": self.norm_stats.normalize(model_obs, self._sim_emb_id)}

    def _inference_cache_value(self, key: str, default):
        overrides = getattr(self, "inference_cache_overrides", {}) or {}
        if key in overrides:
            return overrides[key]
        cfg = self.inference_stages or {}
        cache = cfg.get("cache", {}) if hasattr(cfg, "get") else {}
        value = cache.get(key, default) if hasattr(cache, "get") else default
        return default if value is None else value

    def _graph_model(self, request: dict) -> dict:
        step_kwargs = {}
        if request.get("obs_norm_history") is not None:
            step_kwargs["obs_norm_history"] = request["obs_norm_history"]
        chunk = self.step(
            self._sim_state, request["obs_norm"], self._sim_t,
            embodiment_id=self.domain_by_id.get(self._sim_emb_id),
            **step_kwargs)
        if chunk.dim() != 2:  # (D,)
            return {"actions": [chunk]}

        configured_keep = int(self._inference_cache_value(
            "n_keep", self.replan_stride))
        n_keep = max(1, min(configured_keep, chunk.shape[0]))
        # Existing eval-only interventions remain model-side and default OFF.
        import os as _os
        beta_default = float(self._inference_cache_value("blend", 0.0))
        beta = float(_os.environ.get(
            "PUSHSHAPES_PLAN_BLEND", str(beta_default)) or 0)
        if beta > 0 and chunk.shape[0] >= 2:
            n_keep = max(1, chunk.shape[0] // 2)
            prev = self._sim_prev_chunk
            prev_t = self._sim_prev_chunk_t
            if (prev is not None
                    and prev.shape[0] >= 2 * n_keep
                    and prev.shape[-1] == chunk.shape[-1]
                    and self._sim_t == prev_t + n_keep):
                blended = chunk.clone()
                blended[:n_keep] = (
                    beta * prev[n_keep:2 * n_keep]
                    + (1 - beta) * chunk[:n_keep])
                chunk = blended
                if not self._sim_blend_announced:
                    print(f"[stage4] PLAN_BLEND active beta={beta} "
                          f"t={self._sim_t} n_keep={n_keep} "
                          f"C={chunk.shape[0]}", flush=True)
                    self._sim_blend_announced = True
        self._sim_prev_chunk = chunk.detach()
        self._sim_prev_chunk_t = self._sim_t
        return {"actions": [chunk[j] for j in range(n_keep)]}

    def _graph_update_cache(self, plan: dict, obs: dict):
        self._sim_action_cache.replace(plan["actions"])
        return self._graph_commit(self._sim_action_cache.pop())

    def _graph_commit(self, a_norm):
        # PUSHSHAPES_ACTION_EMA=gamma: P6 output low-pass (normalized space).
        import os as _os
        gamma_default = float(self._inference_cache_value("action_ema", 0.0))
        gamma = float(_os.environ.get(
            "PUSHSHAPES_ACTION_EMA", str(gamma_default)) or 0)
        if gamma > 0:
            prev = self._sim_ema_a
            if prev is not None and self._sim_t > 0 and prev.shape == a_norm.shape:
                a_norm = gamma * prev + (1 - gamma) * a_norm
                if not self._sim_ema_announced:
                    print(f"[stage4] ACTION_EMA active gamma={gamma} "
                          f"t={self._sim_t}", flush=True)
                    self._sim_ema_announced = True
            self._sim_ema_a = a_norm.detach()
        out = self.norm_stats.unnormalize(
            {self._sim_ac_key: a_norm}, self._sim_emb_id)[self._sim_ac_key]
        return (out.detach().cpu().numpy().reshape(-1)
                .astype("float32"))

    # Sim-eval entry: the evaluator only drives this one public method.
    def inference_step(self, obs_zarr: dict, t: int, emb_id: int, T_max=None):
        from egomimic.rldb.embodiment.embodiment import get_embodiment

        if t == 0:
            self._reset_inference_graph(T_max)
        elif not hasattr(self, "_sim_state"):
            raise RuntimeError("inference_step must begin with t == 0")
        self._sim_t = int(t)
        self._sim_emb_id = int(emb_id)
        embodiment_name = get_embodiment(emb_id).lower()
        self._sim_ac_key = (
            self.ac_keys[embodiment_name]
            if embodiment_name in self.ac_keys else self.ac_keys[emb_id])
        return self._inference_graph(obs=obs_zarr)
