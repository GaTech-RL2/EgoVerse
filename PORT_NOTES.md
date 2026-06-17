# PORT_NOTES — BC-RNN / BC-RNN-Transformer family → EgoVerse-pact-2

Ported the complete current **BC-RNN** family (LSTM core + Transformer core +
chunk8 variant) from **EgoVerse2** (`/coc/flash7/paphiwetsa3/projects/EgoVerse2`,
branch `hpt-hnet-pusher-nc3`) into **EgoVerse-pact-2** (a copy of `EgoVerse-pact`,
branch `elmo/dfot-obsactimg-pact`). Date: 2026-06-05. Source files were read-only;
nothing in EgoVerse-pact or EgoVerse2 was modified.

The git delta of this port vs the pre-port baseline (`/tmp/pact2_baseline.txt`,
captured right after the copy) is EXACTLY the list below — 12 adds + 1 shared-file
modification. No scratch left behind (`__pycache__` is gitignored).

---

## ZOO DISSOLVE (2026-06-07) — act/hpt/pi promoted to first-class per-algo peers

Pre-edit tag: `pre-dissolve-zoo` (= clean HEAD `5b1dc20b`). One commit, local-only
(never pushed). The `zoo/` grouping was wrong — HPT and PI are *actively
developed* algorithms benchmarked against the H-Net line, not a pen of frozen
baselines. They are first-class peers of `bc`. So `algo/zoo/` and the mirrored
`eval/zoo/` were **dissolved into per-algo folders**; `algo/` and `eval/` mirror
each other. The principle is documented in **DESIGN.md §9.4**; old→new dotted
paths for old-ckpt `_target_` resolution are in
`scratch/hierarchy_path_map.txt` (ZOO DISSOLVE block).

**Moves (all `git mv` → R-status renames):**

| old | new |
|---|---|
| `egomimic/algo/zoo/hpt.py` | `egomimic/algo/hpt/hpt.py` |
| `egomimic/algo/zoo/pi.py`  | `egomimic/algo/pi/pi.py` |
| `egomimic/algo/zoo/act.py` | `egomimic/algo/act/act.py` |
| `egomimic/eval/zoo/eval_hpt.py` | `egomimic/eval/hpt/eval_hpt.py` |
| `egomimic/eval/zoo/eval_pi.py`  | `egomimic/eval/pi/eval_pi.py` |
| `egomimic/eval/zoo/eval_act.py` | `egomimic/eval/act/eval_act.py` |

Each new folder gets an `__init__.py` re-exporting its public class
(`egomimic.algo.hpt.HPT`/`HPTModel`, `egomimic.algo.pi.PI`,
`egomimic.algo.act.ACT`/`ACTModel`; `egomimic.eval.{hpt,pi,act}.<...>EvalVideo`).
The two `zoo/__init__.py` files were `git rm`'d. The old `algo/zoo/__init__`
PEP-562 lazy-PI shim is gone; **PI laziness is preserved** because the top-level
`egomimic.algo.__init__` never imports `egomimic.algo.pi` eagerly (importing
`egomimic.algo.pi` is what pulls optional `openpi`).

**Mirrored in the SAME commit:** 16 yaml `_target_`s (12 hpt model configs →
`egomimic.algo.hpt.hpt.HPT`; `act.yaml` → `egomimic.algo.act.act.ACT`;
`pi0.5_base.yaml` → `egomimic.algo.pi.pi.PI`; `eval_hpt.yaml`/`eval_pi.yaml`
evaluators → the per-algo eval dotted paths). Importers: `eval/__init__.py`
`_MODULE_HOMES` facade (eval_{hpt,pi,act} basenames repointed),
`algo/__init__.py` doc comments, `tests/test_pi.py` (`import egomimic.algo.pi.pi`).

**Untouched:** `bc` (still `algo/bc/algo.py` — outside this task; not wrapped or
unwrapped here) and the **shared spine** at `algo/` top
(`algo.py`, `packed_base.py`, `loss.py`, `outer_stage.py`, `obs_transforms.py`,
`packed_outer_stage.py`) — these are the contracts every algo composes against,
neither zoo nor bc-specific, so never moved under a per-algo folder. The shared
eval helper `eval/core/_viz_shared.py::cam_frame_mse_and_viz_batches` stays in
`eval/core/`; its import path is unchanged and still resolves from the moved
`eval_hpt.py`/`eval_pi.py`.

---

## 1. Files ADDED (new, no merge)

| Path | What |
|------|------|
| `egomimic/algo/bc_rnn.py` | BC-RNN algo+policy: `BCRNN` (HNet subclass), `BCRNNPolicy`, `_cut_windows` + `_cut_windows_strided` (the strided-obs / action-chunk window cut), `_pack_to_padded`, core dispatch (lstm/transformer), forward_training (windowed NLL), forward_eval (TF overlay), inference_step (closed-loop). Copied verbatim from EgoVerse2. |
| `egomimic/models/bc_rnn_nets/__init__.py` | Package exports. Updated vs EgoVerse2 to also export `VisualCore`/`SpatialSoftmax` from the new local `visual_core` module (see §2c). |
| `egomimic/models/bc_rnn_nets/obs_encoder.py` | `ObsEncoder` — per-frame low-dim+image fuse, `paper_exact` mode (raw low-dim, ReLU on image, no fusion MLP, concat 64+2=66). Verbatim. |
| `egomimic/models/bc_rnn_nets/lstm_core.py` | `LSTMCore` — LSTM over obs-embedding history (forward + init_hidden/step). Verbatim. |
| `egomimic/models/bc_rnn_nets/transformer_core.py` | `TransformerCore` — causal self-attention drop-in for LSTMCore (same interface; windowed train + long-seq overlay path + step rollout). Verbatim. |
| `egomimic/models/bc_rnn_nets/gmm_head.py` | `GMMActionHead` — diagonal-GMM head (NLL train, sample decode, `chunk_len` action-chunking). Verbatim. |
| `egomimic/models/bc_rnn_nets/visual_core.py` | **NEW FILE (not present in EgoVerse2 as a standalone module).** Holds `SpatialSoftmax` + `VisualCore`, copied verbatim from EgoVerse2's `hnet_nets/image_encoders.py`. See §2c for why it lives here instead of being merged into pact-2's `image_encoders.py`. |
| `egomimic/hydra_configs/model/bc_rnn_pushshapes_paperexact.yaml` | LSTM paper-exact config (RNN dim 1000, no actor MLP, repeat-pad unmasked, constant LR). VisualCore `_target_` rewritten (§2c). |
| `egomimic/hydra_configs/model/bc_rnn_pushshapes_paperexact_tx.yaml` | Transformer core (d_model=448, 5L/8H), constant LR. |
| `egomimic/hydra_configs/model/bc_rnn_pushshapes_paperexact_tx_cos.yaml` | TX + warmup→cosine LR (max_steps=90000). |
| `egomimic/hydra_configs/model/bc_rnn_pushshapes_paperexact_tx_cos_lowlr.yaml` | TX + warmup→cosine, peak LR 2.5e-5. |
| `egomimic/hydra_configs/model/bc_rnn_pushshapes_paperexact_tx_chunk8.yaml` | TX + obs_stride=8 + chunk_len=8 (action chunking), warmup→cosine. |
| `scripts/train_bc_rnn_paperexact.sh` | LSTM launcher. Paths rewritten for pact-2 (§3). |
| `scripts/train_bc_rnn_tx.sh` | TX launcher. |
| `scripts/train_bc_rnn_tx_cos.sh` | TX-cos launcher. |
| `scripts/train_bc_rnn_tx_low.sh` | TX-cos-lowlr launcher (uses model `..._tx_cos_lowlr`). |
| `scripts/train_bc_rnn_tx_chunk8.sh` | TX-chunk8 launcher. |
| `logs/sbatch/` | Created (empty) — the launchers' `#SBATCH --output/--error` target. |
| `PORT_NOTES.md` | This manifest. |

## 2. Shared files — what was MODIFIED / the dependency decisions

### (a) norm_stats minmax support — ALREADY PRESENT, no port needed
The BC-RNN launchers pass `norm_stats.norm_mode=minmax`. Checked pact-2:
- `egomimic/trainHydra.py:108` already reads `OmegaConf.select(cfg, "norm_stats.norm_mode", default="quantile")`.
- `egomimic/rldb/zarr/zarr_dataset_multi.py` already has the full minmax path:
  `_apply_norm_one` minmax branch (line 1213, `2*((x-min)/(max-min)) - 1`),
  `_apply_unnorm_one` minmax branch (line 1240), and the stats accumulator
  computes `"min"`/`"max"` (lines 1168-1169).
These predate the ~May-18 pact/EgoVerse2 split. **Nothing ported.**

### (b) `enable_grad_norm` gate — ALREADY PRESENT, no port needed
The paper-exact configs set `enable_grad_norm: false`. pact-2's
`egomimic/pl_utils/pl_model.py` already has `enable_grad_norm: bool = True` (line 38),
`self.enable_grad_norm = ...` (63), and `if not self.enable_grad_norm: ...` (148) —
byte-identical to EgoVerse2 (same line numbers). **Nothing ported.**

### (c) VisualCore — moved into `bc_rnn_nets/visual_core.py` (the CLEAN choice)
The BC-RNN model configs target an image backbone `VisualCore` with crop knobs
(`crop_aug/crop_height/crop_width/crop_eval_mode/crop_sample_mode` + `_crop`). In
EgoVerse2 this lives in `egomimic/models/hnet_nets/image_encoders.py` (alongside
`SimpleConv`, `ResNetEncoder`, `SpatialSoftmax`).

**pact-2's `hnet_nets/image_encoders.py` diverged significantly** — it contains
ONLY `SimpleConv`; it has NO `VisualCore`, `SpatialSoftmax`, or `ResNetEncoder`.
Per the port plan's stated preference ("if image_encoders.py diverged
significantly, move VisualCore into bc_rnn_nets/visual_core.py — CLEANER for a
foreign codebase"), I took that path:
- Created `egomimic/models/bc_rnn_nets/visual_core.py` with `SpatialSoftmax` +
  `VisualCore` copied verbatim from EgoVerse2 (dependency closure = torch / nn /
  numpy / torchvision only; fully self-contained).
- Rewrote the configs' `front_img_1._target_` from
  `egomimic.models.hnet_nets.image_encoders.VisualCore` →
  `egomimic.models.bc_rnn_nets.visual_core.VisualCore` (the ONLY config edit).
- **pact-2's existing `image_encoders.py` was NOT touched** — zero risk to its
  `SimpleConv` and the H-Net code that uses it.

### (d) `constant_scheduler` — PORTED (the one shared-file modification)
The LSTM paper-exact config and the `_tx` config use
`scheduler._target_: egomimic.utils.schedulers.constant_scheduler` (robomimic-exact
constant LR). pact-2's `schedulers.py` had `warmup_cosine_scheduler` (used by the
cos/lowlr/chunk8 configs) but was **missing `constant_scheduler`**. Minimal merge:
- Added `LambdaLR` to the `torch.optim.lr_scheduler` import list.
- Inserted `def constant_scheduler(optimizer) -> LambdaLR(..., lr_lambda=lambda _:1.0)`
  (copied from EgoVerse2) immediately above `warmup_cosine_scheduler`.
`warmup_cosine_scheduler` and `piecewise_linear` are untouched. This is the only
modified shared file (`M egomimic/utils/schedulers.py`).

### (e) Other import-pulled deps — ALL PRESENT, nothing else ported
`import egomimic.algo.bc_rnn` pulls `egomimic.algo.algo.Algo`,
`egomimic.algo.hnet.HNet` (base; provides `_build_obs`/forward_training/etc.), and
`egomimic.rldb.embodiment.embodiment.{get_embodiment,get_embodiment_id}`. norm_stats
exposes `keys_of_type / is_key_with_embodiment / normalize / unnormalize`. All exist
in pact-2 and the import succeeds (verified, §verification).

## 3. Launcher path rewrites (EgoVerse2 → EgoVerse-pact-2)
Each of the 5 launchers had exactly these path substitutions; everything else
(config-name, `data=tsimulation/tsimulation`, `model=...`, `evaluator=eval_hnet_sim`, the
override block, `norm_stats.norm_mode=minmax`, dataset `new_circle_3`, keymap) is
unchanged from EgoVerse2:
- `cd /…/EgoVerse2` → `cd /…/EgoVerse-pact-2`
- `source /…/EgoVerse7/.venv/bin/activate` → `source /…/EgoVerse-pact-2/.venv/bin/activate`
  (pact-2's `.venv` is a symlink → `../EgoVerse-pact/.venv`, the EgoVerse3 sharing
  pattern; read-only use, NO installs into it).
- `#SBATCH --output/--error` log dir → `/…/EgoVerse-pact-2/logs/sbatch/` (dir created).

## 4. Deliberately NOT ported
- The **obsolete pre-paperexact BC-RNN configs** (`bc_rnn_pushshapes.yaml`,
  `bc_rnn_pushshapes_cos.yaml`, `bc_rnn_pushshapes_minmax{,_crop,_crop_cos}.yaml`)
  and their launchers (`train_bc_rnn.sh`, `train_bc_rnn_cos.sh`,
  `train_bc_rnn_minmax*.sh`) — skipped to keep pact-2 clean (only the current
  paper-exact family was requested).
- The **eval / sim-rollout stack** was NOT deep-ported (per the plan: "the user
  asked for the bc-rnn integration, not the full eval migration — flag it"). See §6.
- pact-2's `image_encoders.py`, `pl_model.py`, `zarr_dataset_multi.py`,
  `trainHydra.py` — left as-is (they already had everything needed; see §2a/b/e).

## 5. Verification (on an a40 alloc, pact-2's symlinked .venv, PYTHONPATH=pact-2)
- **py_compile**: all 8 added/modified `.py` (bc_rnn.py, the 5 bc_rnn_nets modules,
  visual_core.py, schedulers.py) compile. PASS.
- **Import**: `from egomimic.algo.bc_rnn import BCRNN` + all bc_rnn_nets exports +
  `constant_scheduler`/`warmup_cosine_scheduler`. PASS.
- **Hydra composition** (`--cfg job` of `train_zarr_cartesian` + `data=tsimulation/tsimulation`
  + `evaluator=eval_hnet_sim` + `norm_stats.norm_mode=minmax`): ALL 5 model configs
  compose cleanly (rc=0) — each resolves `_target_: …bc_rnn.BCRNN`, the right
  `core:`, `chunk_len: 8` for chunk8, `enable_grad_norm: false`, `norm_mode: minmax`.
- **GPU construction + forward + rollout** (built LSTM, TX, TX-chunk8 policies
  directly from ported code on cuda):
  - LSTM:  window forward `raw (4,10,25)`, finite NLL, decode `(4,10,2)`, 12 step()
           calls — 23.50M params.
  - TX:    same shapes, 23.32M params (param-matched to LSTM, by design).
  - TX-chunk8: head `raw (4,10,200)=8*25` → decode **`(4,10,8,2)`** (the chunk8
           shape), 24 step() calls (obs_stride=8) — 23.39M params.
  All outputs finite, all shapes correct. PASS.

## 6. CURRENT LIMITATIONS — eval/data config compatibility (can a run launch as-is?)
**Training composes and the model runs; the launchers' sim-eval block does NOT work
as-is** because pact-2's eval stack diverged from EgoVerse2. Three concrete gaps,
empirically confirmed:

1. **`evaluator.rollout_mode=ar` override fails.** Every launcher passes it, but
   pact-2's `eval_hnet_sim.yaml` / `HNetSimEval.__init__` have **no `rollout_mode`
   key** (nor `delta_action`/`temporal_ensemble`/`chunk_k`/`goal_in_obs`, which
   EgoVerse2's eval config added). Hydra struct-mode error:
   `ConfigAttributeError: Key 'rollout_mode' is not in struct … Could not override
   'evaluator.rollout_mode'`. → A run launches only after removing the
   `evaluator.rollout_mode=ar` line (and any other eval-only overrides pact-2 lacks)
   from the launcher, OR adding those keys to pact-2's eval config/class.

2. **`inference_step` signature mismatch.** pact-2's evaluator calls
   `algo.inference_step(obs_zarr, t, emb_id, T_max=self.max_steps)` (eval_sim.py:251)
   — with a `T_max=` kwarg. The ported BC-RNN's `inference_step(self, obs_zarr, t,
   emb_id)` takes no `T_max` → `TypeError` at rollout. → Adapt either the BC-RNN
   `inference_step` to accept/ignore `T_max`, or pact-2's evaluator to drop it.

3. **`get_keymap_eval` is missing in pact-2.** The launchers set
   `KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval`, but pact-2's
   `pushshapes.py` has only `get_keymap` (+ `get_keymap_hpt`). EgoVerse2's
   `get_keymap_eval` = `get_keymap` (byte-identical in both repos) + two passthrough
   keys: `goal_pose` (read by pact-2's eval, eval_sim.py:333-385 — supported) and
   `init_action` (delta-rollout seed). → For a TRAIN-only smoke this can be swapped
   to `get_keymap`; for sim eval, add `get_keymap_eval` (a ~12-line addition, the
   base is already identical) and reconcile with the `rollout_mode`/`T_max` items.

**Bottom line:** the BC-RNN *model* is fully integrated and trainable in pact-2 —
all configs compose, the policies build and forward/rollout on GPU, minmax + grad
gate + schedulers are wired. To actually launch a BC-RNN **training with sim eval**
from a pact-2 launcher unchanged, the eval stack needs the three small adapts above
(`rollout_mode` key, `T_max` kwarg, `get_keymap_eval`). These were intentionally
flagged rather than deep-ported, per scope.

---
---

# DELTA SYNC — core=hnet (HNetCore) + chunk8-Q (action-query readout)

Date: 2026-06-06. Delta-synced the TWO BC-RNN builds that landed in EgoVerse2
(`hpt-hnet-pusher-nc3`) AFTER the original port (which left pact-2 at the
**chunk8** build: cores lstm|transformer, chunk_head=linear only):

1. **core=hnet** — `HNetCore`, the real dynamic-chunking H-Net as a drop-in core
   under the `lstm:` slot (`core: hnet`).
2. **chunk8-Q** — `chunk_head: queries`, an ACT/HPT-style `QueryActionDecoder`
   action-query chunk readout (transformer core only).

Source of truth = EgoVerse2 (READ-ONLY). Target = EgoVerse-pact-2 only.
EgoVerse-pact = STRICTLY READ-ONLY (untouched: 53 dirty entries, index mtime
2026-05-31, never written). pact-2 `.venv` symlink → EgoVerse-pact's, no installs.
Baseline for exact delta attribution: `/tmp/pact2_presync.txt` (68 entries,
captured at sync start).

## D1. Files REFRESHED (EV2 changed since the original port → re-copied)

| File | Verdict | Action |
|------|---------|--------|
| `egomimic/algo/bc_rnn.py` | DIFFERS | **Re-copied** from EV2. Gains the `core="hnet"` (HNetCore type-guard) branch, the `chunk_head` param + `query_decoder` slot + `_readout()` + the queries rollout branch in `step()`, and the construction guards (queries ⇒ transformer-only, `query_decoder.max_window ≥ rnn_horizon`, dim guards). Default `core="lstm"` / `chunk_head="linear"` path is byte-identical (state_dict construction proven equal, §D6). |
| `egomimic/models/bc_rnn_nets/__init__.py` | DIFFERS | **Merged** (NOT overwritten): EV2's version exports `HNetCore`/`QueryActionDecoder` but drops the local `visual_core` exports; pact-2's keeps `VisualCore`/`SpatialSoftmax` (the original port's local-divergence). Took the UNION — exports all of `ObsEncoder, LSTMCore, TransformerCore, HNetCore, GMMActionHead, QueryActionDecoder, VisualCore, SpatialSoftmax`. Docstring notes HNetCore's machinery is vendored locally (§D4). |
| `gmm_head.py`, `transformer_core.py`, `obs_encoder.py`, `lstm_core.py` | IDENTICAL | No refresh needed (byte-identical EV2 vs pact-2). |
| `visual_core.py` | (pact-2-specific) | Kept as-is; configs keep pointing at it (§D3). |

## D2. Files ADDED (new modules)

| File | Source | Notes |
|------|--------|-------|
| `egomimic/models/bc_rnn_nets/hnet_core.py` | EV2 verbatim **except** its 3 H-Net imports rewritten to the vendored package (§D4). | `HNetCore`: in_proj 66→d_model → EncoderDecoderStage(outer,causal) → ChunkerStage(dynamic chunk) → ComputeStage(inner,causal). obs-only, AdaLN-off, window-off, cross-attn-off, RoPE-off, causal. |
| `egomimic/models/bc_rnn_nets/query_decoder.py` | EV2 **BYTE-IDENTICAL** (no hnet_nets dep — torch only). | `QueryActionDecoder`: chunk_len learnable queries (self-attn + causal cross-attn over core features) → shared GMM proj. `forward` (windowed train) + `forward_step` (rollout). |
| `egomimic/models/bc_rnn_nets/_hnet_vendored/` (8 files) | EV2 `hnet_nets/{context,hnet,stages,blocks,routing,config,isotropic_builder}.py` + a new `__init__.py`. | The vendored H-Net import closure (§D4). |

## D3. Configs + launchers (pact-2 path/target rewrites)

| File → pact-2 location | Rewrites applied |
|------------------------|------------------|
| `bc_rnn_pushshapes_paperexact_hnet.yaml` → `hydra_configs/model/` | `front_img_1._target_`: `…hnet_nets.image_encoders.VisualCore` → `…bc_rnn_nets.visual_core.VisualCore` (matches the original port's image-encoder rewrite). Resolves `core: hnet`, `lstm._target_: …bc_rnn_nets.HNetCore`, d_model=256. |
| `bc_rnn_pushshapes_paperexact_tx_chunk8_q.yaml` → `hydra_configs/model/` | Same `front_img_1._target_` rewrite. Resolves `chunk_head: queries`, `query_decoder._target_: …bc_rnn_nets.QueryActionDecoder` (chunk_len=8, per_step=25, d_model=448). |
| `train_bc_rnn_hnet.sh` → `scripts/` | `cd …/EgoVerse2`→`…/EgoVerse-pact-2`; `source …/EgoVerse7/.venv`→`…/EgoVerse-pact-2/.venv`; `#SBATCH --output/--error` → pact-2 `logs/sbatch/`. Everything else (data=tsimulation/tsimulation, evaluator=eval_hnet_sim, rollout_mode=ar, get_keymap_eval, norm_mode=minmax) unchanged from EV2 — same as the original port's launchers (subject to the §6 eval limitations below). |
| `train_bc_rnn_tx_chunk8q.sh` → `scripts/` | Same 3 path rewrites. |

## D4. ⚠️ THE CAREFUL PART — hnet_nets compatibility verdict: **VENDORED**

`hnet_core.py` imports `HNetContext` (`context`), `HNet` (`hnet`), and
`ChunkerStage/ComputeStage/EncoderDecoderStage` (`stages`). Full transitive
import closure = **7 modules**: `context, hnet, stages, blocks, routing, config,
isotropic_builder` (self-contained — no `cond_encoders`, `image_encoders`, or
`_smoke_stages` reachable).

**Diff of the closure (EV2 vs pact-2's own `hnet_nets/`):**

| Module | EV2 vs pact-2 | Detail |
|--------|---------------|--------|
| `context.py`, `hnet.py`, `routing.py`, `config.py`, `isotropic_builder.py` | IDENTICAL | — |
| `stages.py` | **DIFFERS** | pact-2 ADDED a `residual_scale` scalar skip-gate (defaults `1.0` = upstream; driven by a `ChunkerResidualScheduler` Lightning callback). A pact-only superset; class signatures unchanged. |
| `blocks.py` | **DIFFERS (bidirectional)** | EV2 has sliding-window attn (`window`/`_W`) + per-frame cross-attn (`_forward_per_frame`/`step_per_frame`, "run E") that pact lacks; pact has `adaln_per_token` + a `causal_conv1d` channel-last-contiguous fix that EV2 lacks. **Neither is a strict superset.** |

**Verdict → VENDOR (the conservative, instruction-mandated choice).** `stages.py`
and `blocks.py` diverged, so per the plan ("if diverged → do NOT modify pact-2's
hnet_nets; VENDOR the needed modules") the entire EV2 import closure was copied
verbatim into `bc_rnn_nets/_hnet_vendored/` and `hnet_core.py`'s 3 imports +
every intra-closure import were rewritten
`egomimic.models.hnet_nets.X` → `egomimic.models.bc_rnn_nets._hnet_vendored.X`
(the ONLY edit vs the EV2 originals). This mirrors the `visual_core.py`
precedent and fully decouples HNetCore from pact's evolving `hnet_nets` (pact's
own code keeps using its `hnet_nets` untouched).

**Evidence the diverged code is inert for THIS config (so the verdict is safe,
not just defensive):** HNetCore's stage spec sets `cond: False` (no AdaLN →
`adaln_per_token` unreachable), no `window` key (sliding-window `_W` defaults
off), and `cond_key=None`/`d_cond=0` (no cross-attn → per-frame methods
unreachable). So EV2's and pact's `blocks.py`/`stages.py` are numerically
identical on the active path — vendoring guarantees HNetCore runs against the
exact code it was verified against regardless. Verified at runtime:
`hnet_core.HNet.__module__ == "…bc_rnn_nets._hnet_vendored.hnet"` (bound to the
vendored HNet, NOT pact's hnet_nets).

Vendored files + EV2 source (all from `EgoVerse2/egomimic/models/hnet_nets/`):
`_hnet_vendored/{context,config,routing,blocks,isotropic_builder,stages,hnet}.py`
+ a new `_hnet_vendored/__init__.py` (docstring-only package marker).

## D5. ⚠️ LIVE-TREE RACE — EV2 renamed the core slot `lstm:` → `core_net:` MID-SYNC

EV2 is a live tree (9+ active trainings). **During this sync** (≈14:33) the EV2
maintainer began a refactor renaming the core config slot `lstm:` → `core_net:`
(`bc_rnn.py` kwarg `core_net`, with `lstm` kept as a **deprecated alias**
explicitly "so EgoVerse-pact-2 ported yamls keep working"; the two new configs'
`lstm:` → `core_net:` too, +15 bytes each).

**Decision: pact-2 PINS to the pre-rename snapshot** (`lstm:` slot + `lstm`
kwarg) — what is deployed and FULLY VERIFIED here. Rationale: (a) the deployed
set is internally consistent and passed every check; (b) the EV2 author kept
`lstm:` working as an alias specifically for this port, so pact-2's configs are
forward-compatible with EV2 HEAD anyway; (c) re-syncing mid-refactor invites
another race. The `lstm`→`core_net` rename is a cosmetic in-flight EV2 refactor,
NOT one of the two builds requested. If pact-2 later re-syncs `bc_rnn.py` from
EV2 HEAD, the configs can stay on `lstm:` (alias) or be bumped to `core_net:`
in lockstep — a one-line-per-file change.

## D6. Verification (a40 alloc, pact-2 symlinked .venv, PYTHONPATH=pact-2)

- **py_compile**: all **17** touched `.py` (bc_rnn.py, hnet_core.py,
  query_decoder.py, the merged `__init__.py`, the 4 unchanged bc_rnn_nets
  modules + visual_core, and all 8 `_hnet_vendored/` files) compile. PASS.
- **Import**: `egomimic.algo.bc_rnn` (BCRNN, BCRNNPolicy) + all 8 bc_rnn_nets
  `__all__` exports + vendored `HNet/stages/context`. `HNetCore.HNet` confirmed
  bound to the **vendored** module (decoupled from pact hnet_nets). PASS.
- **Hydra compose** (`train_zarr_cartesian` + data=tsimulation/tsimulation +
  evaluator=eval_hnet_sim + norm_stats.norm_mode=minmax, WITHOUT the eval-only
  overrides pact-2 lacks — see §6): both new configs compose (rc=0). hnet →
  `core=hnet`/`HNetCore`; chunk8-Q → `chunk_head=queries`/`QueryActionDecoder`
  (chunk_len=8, per_step=25, d_model=448). Both: image `_target_` rewritten to
  `bc_rnn_nets.visual_core.VisualCore`, `enable_grad_norm=False`,
  `norm_mode=minmax`. PASS.
- **GPU construct + forward + rollout**:
  - **HNetCore** (23.367M): window forward `(4,10,25)` → decode `(4,10,2)` finite;
    core deterministic in eval (0.0); **prefix-causality** — perturb last frame:
    prefix Δ (pos 0..8) = **0.000**, last-pos Δ = 3.24 (causal ✓);
    **step()-vs-window prefix-consistency Δ = 4.77e-07** (the ~1e-6 spot-check).
  - **chunk8-Q** (26.634M): window forward `(4,10,200)` → decode `(4,10,8,2)`
    finite; **query distinctness** min pairwise Δ = 0.072 (8×448 `query_emb`);
    25-env-step queued rollout all finite/right shape; **window-vs-rollout
    raw-GMM-param parity Δ = 0.000** (the `forward` and `forward_step` readout
    paths are mathematically equivalent); RNG-pinned decoded-action parity
    Δ = 0.000. NOTE: `gmm_head.decode` SAMPLES the GMM, so two decode() calls
    diverge on the sampler RNG alone (~0.5 on the action) — NOT a model/sync
    issue; the meaningful invariant is the raw-param parity (0.0).
  - **lstm/tx defaults unchanged**: both pre-sync configs (paperexact LSTM,
    tx_chunk8) build with `chunk_head=linear`, `query_decoder=None`, and
    seeded state_dict construction equal across two builds (lstm 137 keys /
    23.502M; tx_chunk8 194 keys / 23.394M). The default path is byte-identical.
- **git delta** vs `/tmp/pact2_presync.txt` = EXACTLY the 4 new top-level
  untracked paths (2 configs + 2 launchers); the refreshed/added files inside
  the already-untracked `bc_rnn_nets/` + `bc_rnn.py` don't create new porcelain
  entries. No scratch left. EgoVerse-pact untouched (53 entries, index mtime
  2026-05-31).

## D7. Can pact-2 run all 7 BC-RNN variants train-only?  **YES (train-only).**

The 7 variants and their configs (all compose, all build/forward on GPU):
1. LSTM paper-exact — `bc_rnn_pushshapes_paperexact` (core=lstm)
2. TX — `…_tx`
3. TX-cos — `…_tx_cos`
4. TX-cos-lowlr — `…_tx_cos_lowlr`
5. TX-chunk8 — `…_tx_chunk8` (obs_stride=8, chunk_len=8, linear)
6. **TX-chunk8-Q — `…_tx_chunk8_q` (chunk_head=queries)  [NEW this sync]**
7. **HNet — `…_hnet` (core=hnet)  [NEW this sync]**

All 7 are **trainable** in pact-2 (model builds, forwards, rolls out; minmax +
grad gate + schedulers wired). The SAME eval-stack limitations from the original
port's §6 still apply to ALL 7 for **sim-eval** launches (the launchers pass
`evaluator.rollout_mode=ar` / `get_keymap_eval` / the evaluator calls
`inference_step(..., T_max=)` — three small adapts pact-2's diverged eval stack
needs). Train-only (e.g. drop `evaluator.rollout_mode=ar`, swap KM to
`get_keymap`, or run with `--skip-sim`-equivalent) works for all 7 today.

---
---

# DELTA SYNC — core slot rename `lstm:` → `core_net:` (un-pinning §D5)

Date: 2026-06-06. Brought pact-2 level with EgoVerse2's two config-facing
changes that §D5 had deliberately pinned away from (pact-2 was held at the
pre-rename `lstm:` snapshot). EV2 has since (a) finalized the core-slot rename
`lstm:` → `core_net:` in `bc_rnn.py` (with `lstm` kept as a **deprecated alias**)
and (b) switched the 7 paper-exact configs' literal `max_window: 10` to the
relative interpolation `max_window: ${..rnn_horizon}` (the core slot is a child
of `robomimic_model`, so `..` climbs to `robomimic_model` and reads the ONE
`rnn_horizon: 10`). Source of truth = EgoVerse2 (READ-ONLY,
`hpt-hnet-pusher-nc3`). Target = EgoVerse-pact-2 only. EgoVerse-pact = untouched.
Baseline for delta attribution + the state_dict-equality proof:
`/tmp/pact2_part2_baseline.txt` (porcelain) + `_part2_scratch/pact2_baseline.pkl`
(pre-change GPU-built state_dicts), captured before any edit.

## DS1. `bc_rnn.py` — REFRESHED from EV2 (plain copy, safe)
**Diffed FIRST** (the §D5-mandated check). pact-2's `bc_rnn.py` was EV2-verbatim
apart from being the *pre-alias* snapshot; the diff vs EV2 HEAD was **exactly and
only** the `core_net`/`lstm` alias rename (kwarg `core_net=None, …, lstm=None` +
the "pass core_net OR lstm, not both" guard + the slot-name strings in the
type-guard error messages + `core_net=lstm` in the policy hand-off). **No
`hnet_nets` / `_hnet_vendored` import appears in `bc_rnn.py`** (those live in
`hnet_core.py`, untouched) and pact-2 carried **no local edits** to `bc_rnn.py`
→ per the plan a **plain copy of EV2's version is safe**. Done:
`cp EgoVerse2/…/bc_rnn.py → EgoVerse-pact-2/…/bc_rnn.py`; the two are now
byte-identical (`diff` empty). Its imports still resolve from
`egomimic.models.bc_rnn_nets` (pact-2's merged `__init__.py` exports all of
`ObsEncoder/LSTMCore/TransformerCore/HNetCore/GMMActionHead/QueryActionDecoder`).

## DS2. Configs — flipped `lstm:` → `core_net:` + `max_window: ${..rnn_horizon}`
All **7** ported paper-exact configs (`bc_rnn_pushshapes_paperexact{,_tx,_tx_cos,
_tx_cos_lowlr,_tx_chunk8,_tx_chunk8_q,_hnet}.yaml`). Two surgical substitutions
each: (1) the top-level core slot key `^  lstm:` → `  core_net:`; (2) every
literal `max_window: 10` → `max_window: ${..rnn_horizon}` (with EV2's verbatim
comment). Count of `max_window` interpolation keys = **7** (paperexact base = 0,
it's LSTM with no `max_window`; `_hnet` = 1; `_tx`/`_tx_cos`/`_tx_cos_lowlr`/
`_tx_chunk8` = 1 each; `_tx_chunk8_q` = **2** — core_net AND query_decoder). The
`visual_core.VisualCore` image `_target_` rewrite from the original port is
**preserved** in all 7 (NOT clobbered — these were edited in place, not
re-copied from EV2). No older `bc_rnn_pushshapes*.yaml` (non-paperexact) exist in
pact-2 (they were never ported, §4) — nothing else to flip.

## DS3. Verification (a40 alloc, pact-2 symlinked .venv, PYTHONPATH=pact-2)
- **py_compile** `egomimic/algo/bc_rnn.py`: PASS.
- **Hydra compose** (`train_zarr_cartesian` + data=tsimulation/tsimulation +
  evaluator=eval_hnet_sim, the groups every launcher passes): `paperexact`,
  `_hnet`, `_tx_chunk8_q` all compose (rc=0). Every `max_window` under
  `robomimic_model` **resolves to 10** via `${..rnn_horizon}`; the core object
  now lives under `core_net:` (slot key confirmed flipped lstm→core_net).
- **GPU construct + state_dict equality vs PRE-CHANGE build** (same seed=0,
  built from the OLD `lstm:` configs + OLD `bc_rnn.py` before the edits):
  - **LSTM** `paperexact`  — 137/137 tensors `torch.equal`, 23.502M params.
  - **HNet** `_hnet`       — 352/352 tensors `torch.equal`, 23.367M; `HNetCore.max_window`=10.
  - **chunk8-Q** `_tx_chunk8_q` — 220/220 tensors `torch.equal`, 26.634M;
    `TransformerCore.pos_emb`=10 + `query_decoder.ctx_pos_emb`=10 (both pos
    tables) + both `max_window` attrs=10.
  Param counts + key counts identical pre/post ⇒ the rename + interpolation are
  a pure config-surface change; the built policy is byte-identical.
- **Both-keys guard** (refreshed `bc_rnn.py`): `BCRNN(core_net=…, lstm=…)` raises
  `ValueError("pass core_net OR lstm (deprecated alias), not both")`; neither
  raises `"pass core_net (or the deprecated lstm alias)"`. PASS.
- **git delta** vs the §DS pre-change porcelain = unchanged porcelain set (the 8
  touched paths — `bc_rnn.py` + the 7 configs — were already untracked `??` from
  the original port; editing in place adds no new entries). PORT_NOTES.md grows
  by this section. No scratch left (the `_part2_scratch/` build dir was removed
  after verification).

## DS4. §D5 status → RESOLVED
pact-2 is now **on** EV2's `core_net:` slot + `${..rnn_horizon}` interpolation
(no longer pinned to the pre-rename snapshot). The deprecated `lstm:` alias still
works (guard-tested), so any not-yet-flipped consumer keeps composing. The §6
eval-stack limitations (rollout_mode / T_max / get_keymap_eval) are **unchanged**
— this sync touched only the core-slot naming + the window interpolation, not the
eval path.


---

# STEP 13 — FINAL FLIP (DESIGN.md step 13 + amendments A & B)

**Date:** 2026-06-06 · **Branch:** `elmo/dfot-obsactimg-pact` · local commit only (no push).
**Goal:** `algo/` END-STATE clean — no shims, no dormant code, honest names everywhere.
**Safety tag:** `pre-step13-flip` (rollback point; parity baseline).

## What changed

### 1. Config + import mirror (every shim-routed `_target_` / import flipped to its real home)
Swept ALL hydra configs (model/, evaluator/, top-level) + every non-config importer
(`.py`, `scripts/*.sh`, `.ipynb`, `tests/`). 66 files mechanically rewritten + 9
manually (multi-name package imports / docstring prose). Canonical flip table:

| shim path (old) | real home (new) |
|---|---|
| `egomimic.algo.hnet.HNet` | `egomimic.algo.packed_base.HNet` |
| `egomimic.algo.hnet_outer_stage.*` | `egomimic.algo.packed_outer_stage.*` |
| `egomimic.algo.bc_rnn.*` | `egomimic.algo.bc.*` |
| `egomimic.algo.{act,hpt,pi}.*` (orig flat) and `egomimic.algo.zoo.{act,hpt,pi}.*` (interim) | `egomimic.algo.{hpt.hpt,pi.pi,act.act}.*` (first-class per-algo folders; `zoo/` dissolved — see DESIGN.md §9.4) |
| `egomimic.algo.input_modules.*` | `egomimic.models.stems.input_modules.*` |
| `egomimic.algo.dfot.DFoT` / `<OuterStage>` | `egomimic.algo.diffusion.*` |
| `egomimic.algo.dfot.outer_stage.*` | `egomimic.algo.diffusion.outer_stages.outer_stage.*` |
| `egomimic.algo.dfot.{DFoTBackbone,DFoTDiT3DBackbone}` | `egomimic.models.diffusion.*` |
| `egomimic.algo.dfot.{continuous,discrete}_diffusion.*` | `egomimic.models.diffusion.diffusion.*` |
| `egomimic.algo.vae.VAE` / `vae.algo` | `egomimic.algo.diffusion.VAE` / `diffusion.vae_algo` |
| `egomimic.models.hnet_nets.*` | `egomimic.models.hnet.*` |
| `egomimic.models.bc_rnn_nets.<sub>.*` | `egomimic.models.{stems,cores,heads}.*` (role-routed) |

### 2. Shims DELETED (grep-proven empty of live references first)
`algo/{act,hpt,pi,bc_rnn,hnet,input_modules}.py`, `algo/dfot/`, `algo/vae/`,
`models/hnet_nets/__init__.py`, `models/bc_rnn_nets/__init__.py`. Post-delete
`git grep` of every shim path comes back clean (modulo past-tense provenance
docstrings + DESIGN/PORT_NOTES historical mentions).

### 3. AMENDMENT A — dormant-class purge
`FlatFusedPolicy` (288 lines) + `HNetFused(HNet)` (15 lines) cut from
`packed_base.py` → `scratch/flat_fused_quarantine/flat_fused_classes_from_packed_base.py`
(manifest appended). Both were dead (no live `_target_` can construct them; their
`FlatFusedOuterStage` + 3 fused configs were quarantined in PHASE 1).
**`packed_base.py`: 1239 → 935 lines (−304).** Now carries only the live `HNet`
algo base + `HNetPolicy`.

### 4. AMENDMENT B — partner rename
`git mv algo/hnet_outer_stage.py → algo/packed_outer_stage.py` (R098 rename).
Importers (`packed_base.py`, scripts) + 7 hnet config `_target_`s flipped in the
SAME commit. No shim.

### 5. Hygiene
`__pycache__/` added to `.gitignore`. `tests/` import the role/real homes directly
(`algo.bc`, `algo.packed_base`, `algo.pi.pi`, `models.{hnet,cores,heads,stems}`).

## Verification (a40 alloc 3325557, pact-2 symlinked .venv, PYTHONPATH=pact-2)

| check | result |
|---|---|
| model config compose (`--cfg job`) | **37/37 PASS, 0 FAIL** (7 BC + 7 HNet incl obs_ar/_large + 13 DFoT/pixel + 5 VAE + 4 zoo act/hpt + pi0.5_base) |
| pytest tests/ — post-flip | **122 passed, 8 failed, 4 skipped** |
| pytest tests/ — pre-flip baseline (tag) | **122 passed, 8 failed, 4 skipped** — IDENTICAL 8 (pre-existing `TestAlgoWiring` old-signature fails), **ZERO new** |
| state_dict parity vs pre-flip worktree | LSTM (137 keys/23.5M), HNet (352/33.8M), chunk8-Q (220/26.6M) — **all `torch.equal`** |
| SMOKE=1 sbatch train_bc_rnn_hnet.sh e2e | **TRAIN_EXIT=0** (job 3325560) |
| `git grep _hnet_vendored` (code) | clean (docs only) |

## git delta
86 files changed, 263 insertions(+), 986 deletions(-) — 10 shims deleted, 1 rename
(`hnet_outer_stage.py` → `packed_outer_stage.py`), 304 dormant lines quarantined.


---

# DEDUP CAMPAIGN — GLOBAL ACCEPTANCE GATES (2026-06-06)

Post-step-13 dedup campaign: 3 behavior-preserving collapses landed on
`elmo/dfot-obsactimg-pact`, then verified against the pre-collapse baseline
(`scratch/dedup_baseline/manifest.json`, captured at step-13 / f70565d7).

## Collapses (all `torch.equal` behavioral-equality proven)

| # | commit | what | line delta | proof |
|---|---|---|---|---|
| c1 | f06330c1 | unify 3 pixel-policy DFoT outer stages into one `pixel_mode`-parameterized `PixelObsActionDFoTOuterStage` | 9 files, +363 / −404 | state_dict torch.equal (policy 82 keys/7.33M, regress 90/7.40M, decoupled 87/7.32M, keys_identical) + forward parity all tensors torch.equal + 3 mirrored configs compose |
| c2 | 32eb1fce | move `SimpleConv`+`CondEncoderModule` out of `models/hnet` → `models/stems/` (hnet now pure chunking machinery) | 35 files, +68 / −56 | cond_encoders.py byte-identical to pre-move source; 33 refs (13 py imports + 20 yaml `_target_`) updated; construction state_dict torch.equal + forward torch.equal |
| c3 | c289657e | factor shared zarr read/decode into `rldb/zarr/_common.py` | 4 files, +382 / −33 | old-vs-new bit-identity (tag dedup-c3-pre via git archive): 33 comparisons x 3 episodes x 2 paths all torch.equal; +6 new permanent tests (`test_loader_equality.py`) |

## Global gates (alloc 3325596, a40, pact-2 symlinked .venv)

| gate | result |
|---|---|
| **1. FULL compose sweep** (model+evaluator+eval/viz+data) | **107/109 PASS (2 fails = PI viz evaluator/viz/pi_cartesian_lang{,_wrist}: pre-existing MissingConfigException on parent default evaluator/pi_cartesian; configs UNTOUCHED by all 3 collapses, NOT a regression)** |
| **2. pytest tests/** | **128 passed / 8 failed / 4 skipped** — 8 = same pre-existing fails (7 TestAlgoWiring old-HNet-signature + 1 TestInferNormFromPacked missing-zarr); 128 = baseline 122 + 6 new c3 loader-equality tests. ZERO new failures. |
| **3a. BC smoke** `SMOKE=1 sbatch train_bc_rnn_hnet.sh` (job 3325599) | **TRAIN_EXIT=0**, max_epochs=2 reached |
| **3b. DFoT 1-ep pixel smoke** (srun on 3325596, exercises c1 unified stage) | **DFOT_TRAIN_EXIT=0**, max_epochs=1 reached |
| **4. NLL vs baseline (step-aligned)** | DFoT: 0.28878551721572876 (post) vs 0.28878551721572876 (base) — **bit-identical, Δ=0.0**. BC: [1.3453296422958374, 0.17481404542922974] vs base [1.3453673124313354, 0.1749003529548645] — Δ ≈ 3.8e-5 / 9.0e-5, **well within 1e-3**. PASS. |

`egomimic/models/hnet/` after c2 (pure chunking machinery — no encoder stems):
`blocks.py config.py context.py hnet.py __init__.py install_kernels.sh
isotropic_builder.py routing.py _smoke_stages.py stages.py`.

Determinism: trainHydra calls `L.seed_everything(cfg.seed, workers=True)` +
`set_global_seed`; A40 training is fully deterministic (DFoT loss bit-identical
across re-runs). BC smoke shows a tiny non-zero drift (~1e-5 scale) within the
1e-3 gate tolerance.


---

# DEEP-CLEAN DEAD-CODE PURGE — RECORD (collapses c4–c7, 2026-06-06→07)

The dedup gate record above (e4e55789) documents collapses **c1–c3** only. For
completeness, the four follow-on collapses are recorded here. All
behavior-preserving; each shipped its own equality/regression test.

| # | commit | what | delta | retained-despite-map landmines |
|---|---|---|---|---|
| c4 | `2d249751` | dead-code purge: 5 of 6 zero-ref symbols deleted (`models/diffusion_policy.py`, `models/ddim_scheduler.py`, `models/hnet/_smoke_stages.py`, `CompositeLoss`/`MSELoss`, `_ar_rollout_packed`, `HNetOuterStage.generate`, `pl_data_utils` dead wrappers) | 9 files, +6 / −1116 | **`HNetPolicy` (packed_base.py) PROVEN ALIVE and RETAINED** — map claimed dead, implementer disproved. **`algo/obs_transforms.py` KEPT** — dead-in-practice but the designed config-reachable obs-noise extension point. |
| c5 | `e131bdc4` | hoist shared embodiment-key resolution + `_build_obs` + `log_info` onto base `Algo` | 5 files, +221 / −116 | +`test_embodiment_key_resolution_shared.py` |
| c6 | `d7e02a4b` | hoist 6 identical eval uint8 helpers → `eval/core/img_utils.img_chw_to_uint8` | 8 files, +103 / −46 | +`test_eval_img_utils.py` |
| c7 | `37352ecd` | delegate image-only frame sampler to action-aware superset + hoist loss-reducer skeleton onto `Algo` | 5 files, +255 / −81 | +`test_c7_sampler_reducer_equality.py` |

`37352ecd` (c7) is the **hierarchy-pass baseline** — `pre-models-hier` tags it.


---

# HIERARCHY PASS — RECORD (2026-06-07)

Goal: every misplaced file under `egomimic/` (plus root launchers) relocated to
its semantic home so `egomimic/models/` holds **zero loose `.py`** (role dirs +
subsystem dirs only). See `DESIGN.md §9` for the placement contract.

Each folder-group is **one independently-revertible commit**, tagged `pre-<group>-hier`
beforehand. Every whole-file move is a `git mv` (R-status rename in `git show
--stat`); file *splits* (one source → multiple role homes) necessarily appear as
1 R + N adds and are itemized class-by-class in the path map below. Moved code
is byte-identical modulo import lines; every importer + config `_target_` was
updated in the SAME commit.

## Group commits (on `elmo/dfot-obsactimg-pact`, NOT pushed)

| group | commit | tag | whole-file R-renames | also |
|---|---|---|---|---|
| models | `44e837da` | `pre-models-hier` | 6 | act_nets/hpt_nets SPLIT → cores+heads+stems; denoising_nets/image_vae → diffusion/; 6 hpt + 3 denoising dead classes pruned |
| algo | `474950d2` | `pre-algo-hier` | 0 (in-place class rename) | `packed_base.HNet → PackedAlgoBase` (+ `HNet=` compat alias kept); importers + 7 hnet config `_target_`s flipped |
| utils | `9d124bb5` | `pre-utils-hier` | 6 | lightning-hydra tail → `pl_utils/`; `tensor_utils` → `vendored/`; `egomimicUtils` SPLIT (model helpers → `cores/model_utils`, draw fns → `viz_utils`); 3 dead modules deleted |
| rldb | `19e097d1` | `pre-rldb-hier` | 1 | `zarr_write_test` → `scripts/eva_process/`; 6 dead modules/subpackages deleted |
| eval+pl_utils | `10b2398c` | `pre-eval-plutils-hier` | 1 | `test_model_wrapper` → `tests/`; pushshapes sim-glue extracted → `rldb/embodiment/pushshapes_sim` (re-exported for back-compat) |
| scripts | `342065fd` | `pre-scripts-tests-hier` | 17 | 13 regression smokes → `tests/regression/` (module-level skip guards); `smoke_sim_eval` lib → `eval/core/ckpt_loading` (5 importers updated); 3 ops launchers → `scripts/ops/`; 2 dead fused-config launchers deleted |

**Total whole-file R-status renames: 31** (+ the algo class-rename group with 0 file moves; + 3 file splits itemized in the path map).

## Gate-fix commit (this pass)

Moving `test_model_wrapper.py` into `tests/` newly subjected it to `pytest tests/`
collection (it lived under `egomimic/pl_utils/` at baseline, outside the
collected tree, so its assertions were never exercised). One assertion was
stale: it asserted `optimizers["lr_scheduler"]` *is* a `StepLR`, but
`ModelWrapper.configure_optimizers()` returns the **Lightning scheduler-config
dict** `{"scheduler": <StepLR>, "interval", "frequency"}` — the production
contract is correct, the test was stale. Fixed the assertion to target the
nested `["scheduler"]` key. Debug-the-assertion only; no production behavior
changed.

## Final-gate results (alloc 3325806, a40 brainiac, pact-2 symlinked .venv)

| gate | result |
|---|---|
| `python -c "import egomimic"` | clean |
| **pytest tests/** | **139 passed / 8 failed / 10 skipped** — 8 = same pre-existing fails (7 `TestAlgoWiring` old-HNet-signature + 1 `TestInferNormFromPacked` missing-zarr). **ZERO new failures.** Skips 4→10 from regression-test `pytest.skip` guards. |
| **`ls egomimic/models/`** | `cores heads stems hnet diffusion __init__.py` — **zero loose `.py`** ✓ |

(Compose sweep + both deterministic smokes recorded in the pass commit message /
task report; baselines: compose 107/109, BC `TRAIN_EXIT=0`, DFoT NLL
`0.28878551721572876` bit-identical.)

## OLD-ckpt `_target_` path map (old dotted path → new dotted path)

OLD runs' resolved configs name pre-pass dotted paths; at eval those must still
resolve. The full map (every moved class) is below — also kept verbatim at
`scratch/hierarchy_path_map.txt`.

```
# --- models/ group ---
# act_nets.py SPLIT:
egomimic.models.act_nets.{ResNet18Conv,CoordConv2d,ConvBase,Module}     -> egomimic.models.stems.resnet_conv.*
egomimic.models.act_nets.{PositionalEncoding,Transformer,StyleEncoder}  -> egomimic.models.cores.act_transformer.*
# hpt_nets.py SPLIT (6 dead classes pruned: STPolicyStem, AttentivePooling, vit_base_patch16, T5TokenizerWrapper, T5Encoder, L2Norm):
egomimic.models.hpt_nets.{CrossAttention,Attention,MLP,BlockWithMasking,MultiheadAttention,SimpleTransformer} -> egomimic.models.cores.hpt_transformer.*
egomimic.models.hpt_nets.{PolicyStem,MLPPolicyStem,ResNet}              -> egomimic.models.stems.hpt_stems.*
egomimic.models.hpt_nets.{PolicyHead,MLPPolicyHead,TransformerDecoderBlock,MultiBlockTransformerDecoder} -> egomimic.models.heads.hpt_heads.*
# whole-file moves:
egomimic.models.fm_policy.FMPolicy            -> egomimic.models.heads.fm_policy.FMPolicy
egomimic.models.denoising_policy.DenoisingPolicy -> egomimic.models.heads.denoising_policy.DenoisingPolicy
egomimic.models.denoising_nets.*              -> egomimic.models.diffusion.denoising_nets.*  (PRUNED: ConditionalClassifier1D, CrossTransformerCfg2, CrossTransformerProj)
egomimic.models.image_vae.ImageVAE           -> egomimic.models.diffusion.image_vae.ImageVAE
egomimic.models.preprocess_pi_obs.*          -> egomimic.utils.preprocess_pi_obs.*

# --- algo/ group (OLD path STILL RESOLVES via HNet=PackedAlgoBase alias) ---
egomimic.algo.packed_base.HNet  -> egomimic.algo.packed_base.PackedAlgoBase
# (HNetPolicy in same module UNCHANGED — landmine)

# --- utils/ group ---
egomimic.utils.timing_callback.WandbProfilerLogger -> egomimic.pl_utils.callbacks.timing_callback.WandbProfilerLogger  (only config _target_ that moved)
egomimic.utils.instantiators.*  -> egomimic.pl_utils.instantiators.*
egomimic.utils.logging_utils.*  -> egomimic.pl_utils.logging_utils.*
egomimic.utils.rich_utils.*     -> egomimic.pl_utils.rich_utils.*
egomimic.utils.utils.{extras,task_wrapper,get_metric_value} -> egomimic.pl_utils.utils.*
egomimic.utils.tensor_utils.*   -> egomimic.vendored.robomimic_tensor_utils.*
egomimic.utils.egomimicUtils.{get_sinusoid_encoding_table,reverse_kl_from_samples,frechet_gaussian_over_time,EinOpsRearrange,AlohaFK} -> egomimic.models.cores.model_utils.*
egomimic.utils.egomimicUtils.{draw_actions,draw_dot_on_frame,draw_rotation_text,draw_annotation_text,miniviewer,fmt} -> egomimic.utils.viz_utils.*
# egomimicUtils REMAINDER (ARIA/EXTRINSICS/INTRINSICS, geometry, str2bool, interpolate_*, CameraTransforms, download_from_huggingface, STD_SCALE) STAYS at egomimic.utils.egomimicUtils
# DELETED dead (0 importers; scratch copies in scratch/utils_hier_deleted/): memory_utils, real_utils, obs_utils

# --- rldb/ group ---
egomimic.rldb.zarr.zarr_write_test -> egomimic.scripts.eva_process.zarr_write_test
egomimic.rldb.{compression_utils,data_utils} -> DELETED (dead; superseded by pose_utils)
egomimic.rldb.zarr.{benchmark_forward_pass,test_zarr} -> DELETED (dead/broken)
egomimic.rldb.scripts[.utils] -> DELETED (dead subpackage; nds_pq/str2bool live in egomimicUtils)

# --- eval + pl_utils group ---
egomimic.pl_utils.test_model_wrapper -> test_model_wrapper  (file: egomimic/pl_utils/ -> tests/; top-level import, no __init__)
egomimic.pl_utils.test_model_wrapper.DummyAlgo -> test_model_wrapper.DummyAlgo  (in-test _target_)
egomimic.eval.core.eval_sim.{_env_to_zarr_pushshapes,_state_to_init,_ENV_TO_ZARR} -> egomimic.rldb.embodiment.pushshapes_sim.*  (legacy eval_sim facade still re-exports)

# --- scripts/ group ---
scripts.smoke_sim_eval{,.load_algo_from_ckpt,._MockTrainer} -> egomimic.eval.core.ckpt_loading.*  (5 importers updated)
scripts/{test_dfot_inference,test_dfot_refactor_e2e,test_hnet_outer_stage,test_hnet_refactor_e2e,test_hnet_yamls_load,test_mamba_regression,smoke_packed_dataset,smoke_packed_norm_stats,smoke_packed_training,smoke_packed_training_e2e,smoke_packed_validation,smoke_composite_eval,smoke_teacher_eval}.py -> tests/regression/*  (file moves; pytest.skip guards added)
{setup_nvm,run_eva_docker,pull_models}.sh -> scripts/ops/*
scripts.sbatch_train_hnet_fused_{50,80}ep_cosine -> DELETED (ref removed model=hnet_pushshapes_fused; config quarantined in scratch/flat_fused_quarantine/)
```

## Hierarchy pass — old→new path map (2026-06-07)

Six folder-group commits (`44e837da` models, `474950d2` algo class-rename, `9d124bb5` utils, `19e097d1` rldb, `10b2398c` eval/pl_utils, `342065fd` scripts/tests) moved every misplaced file to its semantic role home. **The complete 91-line dotted-path map (old → new, every moved class) is git-tracked at `scratch/hierarchy_path_map.txt`** — consult it when evaluating OLD runs whose resolved configs name pre-move `_target_` paths (only `egomimic.algo.packed_base.HNet` has a runtime compat alias; all other old paths must be remapped via the table).

Headline moves: models/ loose files → cores/heads/stems/diffusion (act_nets + hpt_nets split by role; 9 dead classes pruned); `packed_base.HNet` → `PackedAlgoBase` (alias kept); utils junk drawer → pl_utils + vendored/robomimic_tensor_utils; egomimicUtils split (model math → cores/model_utils, drawing → viz_utils); regression scripts → tests/regression/ (collection-safe, skip-guarded); `smoke_sim_eval` → `eval/core/ckpt_loading`. Gates: tests 139/8/10 (zero new fails), compose 107/109 (2 pre-broken), BC smoke Δ≤9.3e-5, DFoT smoke bit-identical (0.28878551721572876).


# DFoT EVALUATOR COMBINE — RECORD (combines A + B, 2026-06-07)

Dedup-campaign DFoT-evaluator combine: collapse the near-duplicate
`egomimic/eval/dfot/` evaluators into a family-agnostic skeleton + shared
helpers, with the family-VARIANT sampler/decode pushed onto each outer stage's
`rollout_video_episode` hook (decode-on-outer-stage). All on
`elmo/dfot-obsactimg-pact`, NOT pushed. Tag before each combine
(`pre-combineA-dfoteval`, `pre-combineB-dfoteval`).

## Combine commits

| combine | commit | tag | what |
|---|---|---|---|
| A | `c73a5543` | `pre-combineA-dfoteval` | video-rollout TRIO → 1 family-agnostic `DFoTVideoRolloutEval` (decode-on-outer-stage). `eval_dfot_pixel_video_rollout.py` (360) + `eval_dfot_spatial_video_rollout.py` (308) DELETED; their behaviour now lives on `PixelSpatialDFoTOuterStage.rollout_video_episode` / `ImageSpatialDFoTOuterStage.rollout_video_episode`. Old class names kept as compat aliases (`DFoTPixelVideoRolloutEval = DFoTSpatialVideoRolloutEval = DFoTVideoRolloutEval`) — pure `_target_` redirects. |
| B | `7fde8626` | `pre-combineB-dfoteval` | policy PAIR merge (`eval_dfot_policy_action.py` 188 + `eval_dfot_policy_receding_horizon.py` 147 → one `eval_dfot_policy.py`, RH subclasses Action, shared `_rollout`/`_ddim_from_v`) + shared anchored-DDIM helper `_sampling.anchored_ddim_rollout` (adopted by `bundle_anchored` + `image_spatial` outer-stage anchored branch) + knob/path mixin `_base.DFoTVideoEvalMixin` (`store_dfot_knobs` + `video_dir`, adopted by every DFoT evaluator). |

`eval_dfot_self_rollout.py` (361) was **NOT touched** (byte-identical
`pre-combineA-dfoteval`..HEAD): its uint8 variant is genuinely different (checked
in deep-clean collapse c6), so it stays a standalone evaluator.

## eval/dfot/ before → after

PRE-A: 7 evaluator files = 1784 lines (+ `__init__.py` 36).
HEAD: 5 evaluator modules + 2 shared helpers + `__init__.py` = 1278 lines.
Net **−506 lines (−28%)** across the 7-file set; 4 per-family modules deleted,
2 reusable helpers (`_base.py` 53, `_sampling.py` 86) introduced.

| file (eval/dfot/) | pre-A | HEAD | Δ |
|---|---|---|---|
| eval_dfot_video_rollout.py | 272 | 265 | −7 (now family-agnostic, drives all 3 video families) |
| eval_dfot_spatial_video_rollout.py | 308 | — | −308 (DELETED → alias of video_rollout) |
| eval_dfot_pixel_video_rollout.py | 360 | — | −360 (DELETED → alias of video_rollout) |
| eval_dfot_policy_action.py | 188 | — | −188 (DELETED → merged into eval_dfot_policy) |
| eval_dfot_policy_receding_horizon.py | 147 | — | −147 (DELETED → merged into eval_dfot_policy) |
| eval_dfot_policy.py | — | 334 | +334 (NEW: Action + RH-subclass merged) |
| eval_dfot_bundle_anchored.py | 148 | 140 | −8 (adopts `anchored_ddim_rollout` + mixin) |
| eval_dfot_self_rollout.py | 361 | 361 | 0 (UNTOUCHED — genuinely different uint8 variant) |
| _base.py | — | 53 | +53 (NEW: `DFoTVideoEvalMixin`) |
| _sampling.py | — | 86 | +86 (NEW: `anchored_ddim_rollout`) |
| **7-eval-file total** | **1784** | **1278** | **−506 (−28%)** |

## `_target_` map (every config naming a moved/renamed evaluator class)

LIVE configs (compose-sweep-exercised, used by scripts):
```
eval_dfot_image_spatial.yaml  -> DFoTVideoRolloutEval        (was spatial-family class)
eval_dfot_pixel.yaml          -> DFoTVideoRolloutEval        (was pixel-family class)
eval_dfot_obs_action_image.yaml -> DFoTVideoRolloutEval x2   (composite, unchanged target)
eval_dfot_image_spatial_policy.yaml    -> DFoTPolicyActionEval            (now in eval_dfot_policy)
eval_dfot_image_spatial_policy_rh.yaml -> DFoTPolicyRecedingHorizonEval   (now in eval_dfot_policy)
eval_dfot_bundle_anchored.yaml -> DFoTBundleAnchoredEval     (module unchanged; adopts shared helpers)
eval_dfot_obs_action.yaml      -> DFoTSelfRolloutEval        (UNTOUCHED module)
```
DEAD-but-on-disk configs (NOT archived — must still compose): all resolve via
the compat aliases + the merged `eval_dfot_policy` module re-exports.
`egomimic/eval/dfot/__init__.py` exports all 7 class names (incl. the 2 aliases)
so every old `_target_` still imports.

## Final-gate results (alloc 3326107, a40 megabot, pact-2 symlinked .venv)

| gate | result |
|---|---|
| **pytest tests/** | **139 passed / 8 failed / 10 skipped** — same 8 pre-existing fails (7 `TestAlgoWiring` old-HNet-sig + 1 `TestInferNormFromPacked` missing-zarr). **ZERO new failures.** |
| **compose sweep** | **TOTAL_PASS=107 / TOTAL_FAIL=2** — all 11 DFoT evaluator yamls compose (incl. dead-on-disk `image_spatial_policy`, `_policy_rh`, `bundle_anchored`, `obs_action`); the 2 fails are the pre-broken `viz/pi_cartesian_lang` + `viz/pi_cartesian_lang_wrist` (schema/structured provider), NOT DFoT. |
| **real eval forward** | `evaluator=eval_dfot_image_spatial` + `eval_dfot_pixel` each built a REAL DFoT algo (random weights, fixed seed) and ran ONE `compute_metrics_and_viz` end-to-end through the unified eval + outer-stage decode hook. image_spatial: 11 finite metrics, mp4 (128,384,768,3)=599954 B. pixel: 13 finite metrics (incl PSNR/SSIM/LPIPS), mp4 (18,384,768,3)=288794 B. Both written, non-empty. |

Harness: `scratch/gate3_real_eval_forward.py`; videos at `scratch/gate3_out/`.
