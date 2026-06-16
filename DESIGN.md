# Hourglass Restructure & Build-Out — Recommended Design (synthesis of A + B)

**Repo:** `/coc/flash7/paphiwetsa3/projects/EgoVerse-pact-2` · **HEAD:** `557babca` (snapshot, 69 untracked + 3 modified)
**Status:** decision-ready, awaiting approval. Nothing changed — this workflow is read-only. All claims re-verified on disk 2026-06-06.

## 0. The north star
per-embodiment temporal **ENCODERS** (action tokens cross-attending obs) → temporally **COMPRESSED** embodiment-agnostic tokens (ChunkerStage) → shared embodiment-agnostic **AR TRUNK** (+ language slot later) → embodiment-agnostic temporal **DECODER** → per-embodiment **DECODERS/heads**. Proxies: circle (`new_circle_3`), small-circle (`new_circle_small__3`).

**Stance:** B's end-state clarity where moves are cheap+verifiable; A's safety (snapshot-gated, additive-first, compat shims) everywhere a move is risky.

## 1. Ground-truth corrections to BOTH proposals (changes a decision)

The single most consequential decision in A and B — *which H-Net tree is the encoder source of truth* — is **backwards**:

- **⚠️ Cross-attn is NOT absent from pact `hnet_nets/`.** Verified: `hnet_nets/blocks.py` (1005 lines) has `CrossMultiHeadAttention` **plus a richer `cond_mode` switch** (`adaln`/`cross_attn`/`none`), **AR-step cross-attn with KV cache**, `residual_scale`, `causal_conv1d`, `adaln_per_token` — **28** feature hits vs `_hnet_vendored`'s 20. `hnet_nets/stages.py` has `residual_scale` (4 hits); vendored's has **0**. There's also a `CondEncoderModule` in `hnet_nets/cond_encoders.py`. **Pact's tree is the SUPERSET.** So the canonical survivor is `hnet_nets/`; `_hnet_vendored/` is the one to delete — the reverse of what both proposals said.
- **Skeleton collapse is mechanically safe:** normalized diff (import paths neutralized) = **0** lines for context/config/routing, **6** for isotropic_builder, **10** for hnet (pure import lines). No `blocks.py` merge needed — keep pact's superset, delete vendored.
- **DFoT `inference_step` already has `T_max=None`** (verified). BC-RNN (`bc_rnn.py:775`) does not. The eval gap is BC-RNN-only.
- **Two BC-RNN algos confirmed:** `bc_rnn.py::BCRNN(HNet)` (7 configs) vs `algo/bcrnn/{algo,outer_stage}` (`bcrnn_pushshapes.yaml` → `egomimic.algo.bcrnn.BCRNN`, 1 config).
- **`get_keymap_eval` is genuinely missing** (only `get_keymap`/`get_keymap_hpt`/`viz_gt_preds` exist).
- **`HNetCore` imports `_hnet_vendored`** (lines 99-101); **DFoT/PACT import `hnet_nets`**. Collapse → flip HNetCore's import to `models.hnet`.

## 2. Target module tree (end state)
Role-named (B), reached via `git mv`+`__init__` facades that keep old import paths alive until a final flip:

```
egomimic/models/
  stems/   {obs_encoder, visual_core, cond_encoders, image_encoders, image_vae}
  hnet/    {context,config,routing,isotropic_builder,hnet, blocks(SUPERSET), stages(SUPERSET), _smoke_stages}
           # DELETE bc_rnn_nets/_hnet_vendored/
  cores/   {lstm_core, transformer_core, hnet_core(import flipped -> models.hnet)}
  heads/   {query_decoder, gmm_head, flow/...}
  diffusion/ {backbones/, diffusion/, embeddings, sampling}   # <- algo/dfot/* model pieces
egomimic/algo/
  algo.py outer_stage.py loss.py            # SHARED spine, unchanged
  hnet.py hnet_outer_stage.py ...           # H-Net family base (bc/, hourglass/ subclass it)
  bc/algo.py                                # kept bc_rnn.py; left as-is by the zoo dissolve
  diffusion/{algo, outer_stages/, vae_algo} # <- algo/dfot
  hourglass/{algo, context}                 # *** NEW ***
  hpt/hpt.py  pi/pi.py  act/act.py          # first-class per-algo peers (was zoo/; dissolved)
egomimic/eval/{core,tf,dfot,probes, hpt,pi,act}/  # curated; eval mirrors algo (per-algo peers, was zoo/)
egomimic/rldb/embodiment/{embodiment(+SMALL=16), pushshapes(+get_keymap_eval)}
hydra_configs/model/{bc,diffusion,hourglass,hnet,vae}/
tests/  (consolidated)
ROOT: scripts/{train,eval,install}/  scratch/(gitignored)  docs/  (HANDOFF.md deleted)
```

## 3. Hourglass interfaces → existing verified classes (composition, not new math)

| Stage | Class (existing) | End-state home |
|---|---|---|
| Stem obs fuse | `ObsEncoder` | `models/stems/obs_encoder.py` |
| Stem image codec | `VisualCore` | `models/stems/visual_core.py` |
| **Stem action×obs cross-attn** | `CrossMultiHeadAttention` + `cond_mode="cross_attn"`, `CondEncoderModule` | `models/hnet/blocks.py`, `models/stems/cond_encoders.py` (**pact superset**) |
| Compression (waist) | `ChunkerStage` + `RoutingModule`, wired as `HNetCore` does | `models/hnet/stages.py`, `models/cores/hnet_core.py` |
| Trunk (AR core) | `build_isotropic`/`Isotropic` **or** `DFoTBackbone` (swap via `_target_`) | `models/hnet/`, `models/diffusion/backbones/` |
| Temporal decoder | `QueryActionDecoder` | `models/heads/query_decoder.py` |
| Per-emb heads | `GMMActionHead`, `ImageVAE.decode` | `models/heads/`, `models/stems/image_vae.py` |

Trunk-core interchangeability reuses the existing LSTM/TX/HNet drop-in contract (shape-identical, `_target_`-swappable). A `language=None` slot is reserved in `HourglassContext` now.

## 4. Unification seams (shared by DFoT + BC + Hourglass)
1. **Algo spine** `Algo`/`OuterStage`/`{Loss,CompositeLoss,DFoTLoss,MSELoss}`; Hourglass subclasses `HNet` (as `BCRNN` does at `bc_rnn.py:483`) for per-emb norm + packed path. 2. **Eval** `PackedSimEval` already algo-agnostic; contract `inference_step(...,T_max)`+`forward_eval`; DFoT conforms, hourglass conforms day one, only BC-RNN gaps remain. 3. **Data** per-`dataset_name` blocks keyed on `EMBODIMENT` enum; 2nd embodiment = cloned block + one enum line. 4. **Encoder/compression/decoder** one `models/hnet/` + one `QueryActionDecoder`.

## 5. Ordered migration plan (each step independently verifiable; repo trainable throughout)
Invariant after every step: 7 BC-RNN configs compose, DFoT + the per-algo peers (hpt/pi/act) import, `import egomimic` clean. Unit of work = `git mv`+shim.

- **0 Snapshot** (local commit, no push) — rollback point. *Verify: `git status` clean; BC-RNN composes.*
- **1 Sweep root debris → `scratch/`** (40 `.sh`, 11 `patch_*.py`, 12 debug/test `.py`, 9 png/mp4; sibling-repo dead weight), gitignore after. *Verify: grep-clean, import clean.*
- **2 Land DFoT WIP** (3 M + 3 pixel-policy outer-stages) — KEEP. *Verify: 3 configs dry-run.*
- **3 Collapse H-Net:** `git mv hnet_nets→models/hnet`; **delete `_hnet_vendored/`**; leave `hnet_nets/__init__` shim. *Verify: test_hnet_nets green.*
- **4 Flip `hnet_core.py` import** to `models.hnet` (superset flags default-OFF). *Verify: fixed-seed forward bit-identical.*
- **5 Resolve BC-RNN algos:** keep `bc_rnn.py`→`algo/bc/algo.py`; quarantine `algo/bcrnn/`+yaml→scratch; shim. *Verify: 7 configs resolve.*
- **6 Relocate model modules** → `models/{stems,cores,heads}`; `bc_rnn_nets/__init__` facade. *Verify: facade import + smoke train.*
- **7 Relocate DFoT pieces** → `models/diffusion/`; `algo/dfot→algo/diffusion`; shim. *Verify: dfot/vae configs compose.*
- **8 Home the per-algo peers (hpt/pi/act) + tests + curate eval/** (edit ~20 evaluator-yaml `_target_` directly — they don't resolve through `__init__` shims). *Verify: each evaluator yaml composes.* *(Originally grouped under a `zoo/` bucket; later dissolved into first-class per-algo folders — see §9.4.)*
- **9 Add `get_keymap_eval`** (~17 lines; `goal_keys`∉NORMALIZE_KEY_TYPES, safe; serves both circles). *Verify: 6 launchers resolve, batch has `goal_pose`.*
- **10 Close BC-RNN sim-eval gap:** add `T_max=None` to BC-RNN `inference_step`; strip `rollout_mode=ar` from launchers. *Verify: 1-batch sim eval logs coverage.*
- **11 BUILD HOURGLASS:** `algo/hourglass/{algo,context}` = `Hourglass(HNet)` + `HourglassOuterStage`; `inference_step` with `T_max` as thin replay of outer-stage forward; `hourglass_pushshapes.yaml` (single-emb) + `train/hourglass.sh` (sibling-cd/venv header verbatim) + `eval_hourglass.yaml` (`EvalList[HNetEvalVideo + PackedSimEval]`). *Verify: CPU forward; **TF overlay renders BEFORE sim**.*
- **12 Second embodiment** (only registry edit): `PUSHSHAPES_SIM_SMALL=16`; clone `tsimulation_two_emb.yaml`; **reconcile label double-resolution** (confirm dict-key overrides leaf id=15 before launch); 2nd `env_kwargs`. *Verify: two distinct norm-stats keys; 2-block smoke routes both.*
- **13 Final flip + re-doc:** mirror configs, flip `_target_` off shims, delete shims; refresh PORT_NOTES/AGENTS, delete HANDOFF.md. *Verify: full `--cfg job` sweep with zero shims; `git grep _hnet_vendored` empty.*

Steps 0+1 are hard prerequisites; stoppable after any step with a trainable tree.

## 6. Risks
1. **Delta-sync agent still writing** (highest) → gate all moves on snapshot + quiescent writer. 2. **Embodiment label double-resolution** (`zarr.json`=15 vs dict-key=16; silent merge + norm-stats last-writer-wins) → verify dict-key override before launch; leaf rewrite is a data write, flag it. 3. **Small-circle un-wired in sim env** (`pusher_shape∈{circle,stick}`, `PUSHER_RADIUS=15.0` hardcoded) → debug against TF overlay first; env change is separate later task. 4. **`_target_` shims weaker than python shims** → edit evaluator-yaml paths directly. 5. **Inference on algo not outer_stage** → keep hourglass `inference_step` a thin replay. 6. **Launchers hardcode sibling cd/venv** (pact-2 `.venv` is a symlink) → clone header verbatim. 7. **Numeric drift on collapse** → fixed-seed bit-identical check, superset flags default-OFF.

## 7. 53-WIP disposition
- **KEEP first-class:** 3 DFoT pixel-policy outer-stages + 3 configs + inference handlers (trained pixelpol_A/B/DEC).
- **ARCHIVE→scratch (gitignored, history kept):** 40 root `.sh`, 11 `patch_*.py`, 12 debug/test `.py`, 9 png/mp4, helper scripts.
- **QUARANTINE→scratch:** `algo/bcrnn/` (collides with active `bc_rnn.py`).
- **DELETE:** `_hnet_vendored/` (inferior dup); `HANDOFF.md` (stale).
- **REQUEST-DELETE LIST:** anything in scratch confirmed dead — your call, no auto-delete.
- All scratch moves / deletes / `zarr.json` rewrites are **file/data writes — out of this read-only scope; recommended, to be executed in a write-enabled follow-up.**

## 8. Deliberately UNTOUCHED
`algo/algo.py`, `outer_stage.py`, `loss.py` (the shared spine); `algo/hnet.py` (per-emb-norm/packed base); DFoT/PACT *behavior* (pure relocations behind shims); `pl_utils/`, `utils/`, `trainHydra.py` (only norm-stats loop, Step 12); the 7 BC-RNN + all DFoT + hpt/pi/act config *behaviors*; the H-Net `step()==forward` train==rollout contract.

---

## 9. Hierarchy contract (post-pass, 2026-06-07)

The hierarchy pass relocated every misplaced file under `egomimic/` to its
semantic home via `git mv` (R-status renames only — no delete+recreate). The
contract that governs *where a file lives* is below; treat it as the rule for
placing any new file.

### 9.1 Role-dirs vs subsystem-dirs asymmetry (intentional, not a smell)

`egomimic/models/` contains **no loose `.py`** — only `__init__.py` and these
subdirectories. Two kinds of subdir coexist on purpose:

- **Role dirs** — `cores/`, `heads/`, `stems/` — slice models by their
  *function in the hourglass* (encoder stem → core/trunk → decoder head). A
  network class lands here based on the role it plays, regardless of which
  algorithm family ships it. (e.g. `stems/resnet_conv.py`, `stems/hpt_stems.py`,
  `cores/act_transformer.py`, `cores/hpt_transformer.py`, `heads/fm_policy.py`,
  `heads/hpt_heads.py`.)
- **Subsystem dirs** — `hnet/`, `diffusion/` — are self-contained *machinery
  packages* that don't decompose cleanly into stem/core/head. `hnet/` is the
  chunking/routing/isotropic compression machinery; `diffusion/` is the
  backbones + noise schedules + sampling + VAE. They sit **beside** the role
  dirs rather than being shredded across them, because their internal files
  reference each other and only make sense as a unit.

This asymmetry is deliberate: roles answer "what part of the pipeline is this?",
subsystems answer "what cohesive engine is this?". Do not try to force
`hnet/`/`diffusion/` internals into `cores|heads|stems` — that would scatter a
tightly-coupled package for no gain.

### 9.2 `scripts/` split — root launchers vs `egomimic/scripts/` data CLIs

Two `scripts/` trees exist and mean different things:

- **Root `scripts/`** — *launchers / ops* (`*.sh` sbatch wrappers, install
  scripts, eval drivers, `scripts/ops/` env+docker+pull). These are
  **run-as-script**, never imported. Regression smokes that are run-as-script
  (not importable, GPU/clone-3-path dependent) live under
  `scripts/`→ now `tests/regression/` with module-level `pytest.skip` guards.
- **`egomimic/scripts/`** — *importable data CLIs* inside the package
  (`run_conversion.py`, `data_visualization.py`, `aria_process/`,
  `mps_process/`, `eva_process/`, language/embedding/mecka processors, …). These
  are part of the installed package and may be imported by other package code.

Rule of thumb: if it's `python foo.sh`/`sbatch foo.sh` glue → root `scripts/`;
if it's `python -m egomimic.scripts.foo` and other modules import from it →
`egomimic/scripts/`. A reusable *library* extracted out of a launcher goes to
its semantic package home, not `scripts/` (e.g. the ckpt-loading lib went
`scripts/smoke_sim_eval.py` → `egomimic/eval/core/ckpt_loading.py`).

### 9.3 Final `egomimic/models/` tree (end state)

```
egomimic/models/
  __init__.py                    # ONLY loose file under models/
  cores/    act_transformer.py hnet_core.py hpt_transformer.py
            lstm_core.py model_utils.py transformer_core.py
  heads/    denoising_policy.py fm_policy.py gmm_head.py
            hpt_heads.py query_decoder.py
  stems/    cond_encoders.py hpt_stems.py image_encoders.py
            input_modules.py obs_encoder.py resnet_conv.py visual_core.py
  hnet/     blocks.py config.py context.py hnet.py isotropic_builder.py
            routing.py stages.py install_kernels.sh        # SUBSYSTEM
  diffusion/ denoising_nets.py embeddings.py image_vae.py sampling.py
             backbones/{backbone,dit3d_backbone,spatial_backbone}.py
             diffusion/{continuous,discrete}_diffusion.py noise_schedule.py   # SUBSYSTEM
```

Sibling support packages reached by the pass: `egomimic/pl_utils/`
(lightning-hydra glue + `callbacks/`), `egomimic/vendored/`
(`robomimic_tensor_utils.py`, verbatim third-party), `egomimic/utils/` (generic
helpers only), `egomimic/rldb/embodiment/` (sim-glue), `tests/` +
`tests/regression/`.

### 9.4 Algorithms are first-class peers under `algo/` (the dissolve of `zoo/`)

There is **no `zoo/` grouping**. An earlier pass parked the comparison policies
(ACT, HPT, PI) under `egomimic/algo/zoo/` as if they were a frozen pen of
third-party baselines. That framing is wrong: **HPT and PI are actively
developed in this repo** (the H-Net work is benchmarked directly against them),
so they are first-class *algorithms* — peers of `bc`, not exhibits in a zoo. The
`zoo/` bucket (both `algo/zoo/` and the mirrored `eval/zoo/`) was therefore
**dissolved into per-algo folders**:

- **Every algorithm is a first-class peer under `algo/`.** When an algorithm may
  grow (its own model/head/config surface), it gets **its own folder**:
  `algo/hpt/hpt.py`, `algo/pi/pi.py`, `algo/act/act.py` (each with an
  `__init__.py` re-exporting its public class so `egomimic.algo.hpt.HPT` etc.
  import cleanly). `bc` is the one exception that stays a single flat
  module-home (no per-variant model surface to grow) and is **not wrapped** by
  this pass.
- **`eval/` mirrors `algo/`.** Each per-algo eval folder pairs with its algo:
  `eval/hpt/eval_hpt.py`, `eval/pi/eval_pi.py`, `eval/act/eval_act.py`. Shared
  eval machinery that any algo can call (e.g. the cam-frame MSE + viz helper
  `eval/core/_viz_shared.py::cam_frame_mse_and_viz_batches`) stays flat in
  `eval/core/` — it is not algo-specific.
- **The shared spine stays flat at the `algo/` top.** `algo/algo.py`
  (`Algo`/`PackedAlgoBase`), `packed_base.py`, `loss.py`, `outer_stage.py`,
  `obs_transforms.py`, `packed_outer_stage.py` are the contracts every algo
  composes against — they are **not** a baseline pen and **not** bc-specific, so
  they are never moved under a per-algo folder.

Rule for placing a new algorithm: give it `algo/<name>/<name>.py` +
`__init__.py` re-export (and `eval/<name>/eval_<name>.py` if it ships an
evaluator), point its config `_target_` at the new dotted path, and leave the
shared spine alone. Old checkpoint `_target_`s that still name
`egomimic.algo.zoo.*` / `egomimic.eval.zoo.*` resolve through
`scratch/hierarchy_path_map.txt`.

---

**Full doc on disk:** `/tmp/hourglass_design.md`

**Most important thing to decide before approving:** the H-Net survivor direction. Both A and B proposed building on `_hnet_vendored` and deleting/relegating pact's `hnet_nets/`. On-disk verification shows the opposite is correct — pact's `hnet_nets/` is the cross-attn + residual_scale + causal_conv1d **superset**, so this design keeps it, deletes `_hnet_vendored`, and flips `hnet_core.py`'s import (Steps 3-4). If you have context that `_hnet_vendored` was intentionally the "blessed/verified" copy despite being the subset, that single decision flips and Steps 3-4 invert.