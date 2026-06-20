# HPT RICL — retrieval in-context learning on the Qwen HPT model

Branch: `ryanco/hpt-icl`. Last updated: 2026-06-20.

This document covers the architecture, the tests run so far, and the tests that
still need a cluster/data to run. It is the HPT analog of the pi0.5 RICL work
(`egomimic/ricl/CLAUDE.md`, `README.md`).

---

## 1. Context — why HPT differs from pi0.5

RICL retrieves, for each query frame, its **k≈4 nearest demonstrations** (DINOv2
kNN, precomputed offline) and conditions the policy on their
`(image, state, action-chunk)`.

- **pi0.5** is a unified VLM, so RICL there is "no model surgery": retrieved
  images are appended to the image dict and retrieved state/action are discretized
  into the text prompt — the same encoders handle everything.
- **HPT** is a `stems → trunk → heads` model: every modality is a *separate stem*
  that cross-attention-pools its input into a fixed set of latent tokens. Crucially
  **HPT never encodes an action chunk as input** (actions only appear as learnable
  query tokens + head outputs).

So HPT RICL turns each retrieved demo's `(image, state, action-chunk)` into tokens
via **dedicated retrieved stems** and **flat-concatenates** them into the trunk.

---

## 2. Architecture overview

### 2.1 Data (reused unchanged from `egomimic/ricl`)

The retrieval pipeline is model-agnostic and reused as-is: `RiclDataModuleWrapper`
(`pl_utils/pl_data_utils.py`), the collate + bank loader (`ricl/data.py`), and the
DINOv2 kNN cache (`ricl/retrieval.py`). The collate attaches 5 keys per query:

| batch key | shape | notes |
|---|---|---|
| `ricl_retrieved_images` | `(B, k, C, H, W)` | CHW float in **[0,1]** (matches the ImageNet-Normalize augs) |
| `ricl_retrieved_state`  | `(B, k, Ds=32)` | normalized in the bank convention |
| `ricl_retrieved_action` | `(B, k, Ha=15, Da=32)` | normalized, converted to the 32-D shared space |
| `ricl_retrieved_mask`   | `(B, k)` bool | valid neighbor (handles `< k`) |
| `ricl_retrieved_dist`   | `(B, k)` float | kNN distance (unused in v1) |

### 2.2 Model (`egomimic/algo/hpt_ricl.py`)

`HptRicl(HPT)` + `HptRiclModel(HPTModel)`. Three retrieved stems (`embed_dim=256`):

- **`ricl_image`** — runs the **shared** query `front_img_1` ResNet backbone, then a
  *separate* cross-attention pooling head → 16 tokens.
- **`ricl_state`** — a separate `MLPPolicyStem` (32-D input) → 16 tokens.
- **`ricl_action`** — a new `ActionChunkStem` (`egomimic/models/hpt_nets.py`):
  temporal **Conv1D** over the chunk (`temporal_encoder='conv'`, swappable to
  `'transformer'`) → cross-attn pool → 16 tokens. This is HPT's first
  action-chunk-as-input encoder.

**Fusion = flat concatenation (no trunk architecture surgery).** Retrieved token
blocks are appended after the query stem tokens, tagged with learned
demo-index + modality embeddings (zero-init), and invalid demos are (a) zeroed at
the stem output and (b) hidden from attention via a per-sample `key_padding_mask`.

Trunk input sequence (each `[..]` = a 16-token block):

```
[action queries (64)] [q-img] [q-state] [q-lang]        <- plain HPT (positions unchanged)
[d0-img][d0-state][d0-act] ... [d(k-1)-img][...][...]   <- retrieved, appended last
   + learned (demo-index, modality) embeddings on the retrieved spans
   + key_padding_mask hides invalid-demo token spans (per-sample, True = ignore)
```

Because retrieved modalities are simply **absent when k=0** and **appended after**
the query blocks, query token positions and the global sinusoidal position embedding
are byte-identical to plain HPT, and the head always reads `trunk_tokens[:, :action_horizon]`.
**=> k=0 reduces to plain HPT exactly** (verified bit-exact).

Token budget at k=4: `64 (action) + 48 (3 query stems × 16) + 192 (3 mod × 4 demos × 16) = 304`.

### 2.3 Trunk masking (`egomimic/models/hpt_nets.py`)

The trunk previously threaded only PyTorch's `attn_mask` (`L×L`, batch-shared).
Added an optional, `None`-defaulted **`key_padding_mask`** (`(B, L)`, per-sample)
through `MultiheadAttention.forward`, `BlockWithMasking.forward`, and
`SimpleTransformer.forward` (incl. the checkpoint path). `HPTModel.forward_features`
reads it via `getattr(self, "_ricl_key_padding_mask", None)` — absent/None for plain
HPT, so every existing call site is unchanged.

### 2.4 Evaluator (`egomimic/eval/hpt_ricl_eval.py`)

`HptRiclEval(HPTEvalVideo)` runs the model on the same frames twice — **retrieval**
(full batch) vs **floor** (`ricl_*` stripped from the *raw* batch → genuine k=0) —
and reports `RICL/retrieval_*`, `RICL/floor_*`, `RICL/delta_*`, `RICL/retrieval_helps`.
`wants_raw_batch = True` so the floor is built *before* `process_batch_for_training`.
Random-demo and paired-seed flow-loss controls are wired but **off by default** (v1).

### 2.5 Key files

| File | Role |
|---|---|
| `egomimic/algo/hpt_ricl.py` | `HptRicl` + `HptRiclModel` (3 overrides + stem encode/mask) |
| `egomimic/models/hpt_nets.py` | `ActionChunkStem`; `key_padding_mask` plumbing |
| `egomimic/algo/hpt.py` | `HPT.model_cls` hook; `forward_features`/`resume_from_depth` read the mask |
| `egomimic/eval/hpt_ricl_eval.py` | `HptRiclEval` (retrieval vs floor) |
| `egomimic/hydra_configs/model/hpt_ricl_pickplace_qwen.yaml` | model recipe |
| `egomimic/hydra_configs/data/cotrain_hpt_ricl_pickplace.yaml` | data (forks pi RICL pickplace) |
| `egomimic/hydra_configs/evaluator/eval_hpt_ricl.yaml` | evaluator recipe |
| `egomimic/hydra_configs/train_zarr_hpt_ricl.yaml` | top-level train config |

### 2.6 Locked design decisions

- Fusion = **flat concat** + learned demo/modality embeds.
- First run = **eva→eva oracle** (robot bank). aria→eva cross-embodiment is a later
  data-config swap, not a code change.
- Retrieved image encoder **shares the query ResNet** (separate pooling head);
  state/action encoders fully separate.
- Eval v1 = **retrieval vs floor** (random/flow-loss flags exist, off).
- Retrieved inputs in the **32-D shared space** (`data state_dim/action_dim=32`);
  query stays 14-D — the retrieved stems are separate/learned, so the model config is
  identical for a later aria→eva run.
- Query keymap is `cartesian` (→ `front_img_1`); the **bank** stays `cartesian_pi`
  (`base_0_rgb`) and is repackaged into the model-agnostic `ricl_*` keys.
- HPT trunk `action_horizon` (# cond tokens, 64) ≠ head `act_seq` (prediction length,
  15) **by design** — they cross-attend; do not force-align. Data `chunk_length`=15.

---

## 3. Tests run (local, CPU — all passing)

No GPU/S3/cache needed; a fake `norm_stats` exercises the real algo + config path.

### 3.1 Component / correctness

| Test | Result |
|---|---|
| **k=0 ≡ plain HPT** (mask all-false vs no-ricl) | **bit-exact**, max diff `0.0` |
| Per-sample variable-k masking (partial mask) | sample with all demos invalid == no-ricl; 1 valid demo differs |
| Demos change outputs (mask all-true) | output differs from no-ricl |
| `ActionChunkStem` shapes + grads (conv **and** transformer) | `(B*k,15,32) → (B*k,16,256)`, grads flow |
| 3 retrieved stems instantiate from the **committed config** | each → `(4,16,256)` |
| Plain `HPTModel` regression (Step-1 mask edits) | default path unchanged, finite |
| `ruff` on all changed Python files | clean |

### 3.2 Full real-pipeline path (config + `HptRicl` + fake `norm_stats`)

| Test | Result |
|---|---|
| Construct `HptRicl` via real config | builds `HptRiclModel`, 3 ricl stems registered |
| `process_batch_for_training` carries `ricl_*`, cleans base `None`-key | ✅ |
| `forward_training` (k>0) finite loss + `backward` | grads reach `ricl_action` stem **and** `demo_embed` |
| k=0 floor via `strip_ricl_keys` | finite loss |
| `forward_eval` (sampled actions) | `(2,15,14)` finite predictions |
| **kpm length == trunk sequence length** | `304 == 64+48+192` (no off-by-one) |
| Full `HptRiclEval.compute_metrics_and_viz` | emits `RICL/retrieval_*`, `floor_*`, `delta_*`, `retrieval_helps`, `Valid/action_loss` |
| **Overfit a fixed batch** (80 AdamW steps) | loss `67.6 → 6.0` (**91% drop**) → the model genuinely learns |

### 3.3 Config

- Full top-level `train_zarr_hpt_ricl` composes (Hydra `--cfg job`); RICL fields
  resolve correctly (`HptRicl` target, `num_retrieved_observations` at algo+trunk,
  3 stems, head horizon 15, query keymap `cartesian`, bank `cartesian_pi`, 32-D,
  `bank_norm_path` interpolation).
- Plain qwen HPT config (`train_zarr_cartesian model=hpt_bc_pickplace_qwen_pooled`)
  still composes — **no regression**.

> Caveat: §3.2 metric values use random weights + random data, so `retrieval_helps`
> etc. are **meaningless** there — they validate the *machinery*, not learning. The
> overfit drop validates *learnability of the architecture*, not task performance.

---

## 4. Tests that still need the cluster (GPU + S3 + a retrieval cache)

The fake-`norm_stats` harness skips only the real S3 data loading + retrieval cache.
Run on a node (per `CLAUDE.md`; eval/smoke fits 48 GB, export `TORCH_COMPILE_DISABLE=1`
for short runs):

```bash
salloc -A gts-dxu345-rl2 -N1 -q inferno -t 1:00:00 --mem=75G --gres=gpu:l40s:1
source emimic/bin/activate
```

1. **1-step real `trainHydra`** — exercises the data pipeline → collate → loss:
   ```bash
   python egomimic/trainHydra.py --config-name train_zarr_hpt_ricl \
       data.retrieval_cache_dir=<pickplace train cache> \
       norm_stats.precomputed_norm_path=<eva norm_stats dir> \
       trainer=debug logger=debug
   ```
   Pass: finite training loss; `front_img_1`/`state_ee_pose`/`actions_cartesian`
   query keys line up; `ricl_*` flow through.

2. **Eval retrieval vs floor on real frames** (the scientific check) — on the eva→eva
   oracle expect `RICL/delta_*_mse ≤ 0` / `RICL/retrieval_helps = True`.

3. **k-sweep** (`model.robomimic_model.num_retrieved_observations` ∈ {0,1,4}, and the
   matching `trunk.num_retrieved_observations`): k=0 must match plain HPT; larger k
   should change/improve predictions.

4. **Real overfit** — drive one real episode's loss to ~0 (learnability on real data).

5. **Full training run** + (optional) the eval ablations below.

### Optional deeper tests (also cluster)
- `evaluator.compute_random=true` (random-demo control) and
  `evaluator.compute_flow_loss=true` (paired-seed flow loss) — already wired.
- A/B the action encoder:
  `model.robomimic_model.stem_specs.eva_bimanual.ricl_action.temporal_encoder=transformer`.
- Build a tiny smoke cache via `egomimic/ricl/scripts/build_ricl_smoke_cache.py` so
  step 1 needs no large precomputed cache.
- aria→eva cross-embodiment: new data config (bank=aria, `HumanBimanualCartesianEuler`,
  `bank_norm_path` → aria stats); model config unchanged (32-D retrieved stems).

---

## 5. How to run / wire

- **Norm stats invariant:** `norm_stats.precomputed_norm_path` must be the eva (bank)
  stats; `data.bank_norm_path` interpolates from it (`${norm_stats.precomputed_norm_path}`)
  for the eva→eva oracle. For a cross-embodiment run, set `bank_norm_path` to the
  bank's own stats explicitly.
- **k is set in two places** (keep in sync): `robomimic_model.num_retrieved_observations`
  (algo) and `robomimic_model.trunk.num_retrieved_observations` (sizes the demo-index
  embedding table).
- The data config reuses the validated pi RICL pickplace splits + eva bank wiring
  (`data/cotrain_pi_ricl_pickplace.yaml`); only the query camera keymap (→ `cartesian`)
  and the retrieved widths (→ 32-D) change.

---

## 6. Status summary

- **Code + configs:** complete; `ruff` clean; plain HPT unregressed.
- **Local verification:** complete (correctness incl. bit-exact k=0 equivalence,
  full algo path, learnability). 
- **Remaining:** the cluster end-to-end (§4) — the real data pipeline, the
  retrieval-vs-floor science on real frames, and the k-sweep.
