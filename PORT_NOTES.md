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
(config-name, `data=tsimulation`, `model=...`, `evaluator=eval_hnet_sim`, the
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
- **Hydra composition** (`--cfg job` of `train_zarr_cartesian` + `data=tsimulation`
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
| `train_bc_rnn_hnet.sh` → `scripts/` | `cd …/EgoVerse2`→`…/EgoVerse-pact-2`; `source …/EgoVerse7/.venv`→`…/EgoVerse-pact-2/.venv`; `#SBATCH --output/--error` → pact-2 `logs/sbatch/`. Everything else (data=tsimulation, evaluator=eval_hnet_sim, rollout_mode=ar, get_keymap_eval, norm_mode=minmax) unchanged from EV2 — same as the original port's launchers (subject to the §6 eval limitations below). |
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
- **Hydra compose** (`train_zarr_cartesian` + data=tsimulation +
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
- **Hydra compose** (`train_zarr_cartesian` + data=tsimulation +
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
