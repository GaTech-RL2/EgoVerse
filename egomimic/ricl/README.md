# RICL on EgoVerse pi0.5 — retrieval-based in-context learning

Adds **retrieval-based in-context learning (RICL)** to EgoVerse's own **pi0.5**,
reusing the existing zarr / Cartesian / `trainHydra` / `PI`-algo stack plus a
DINOv2 embedding pass.
For each query observation we retrieve the *k* nearest demonstrations and inject
their `(image, state, action)` into the policy's prefix so it can imitate a task
it was never trained on. **Cross-embodiment**: retrieval **bank = human (aria)**,
**query = robot (eva)**, scoped by `egomimic/scripts/human_robot_pairs.json`.

`ricl_openpi` (pi0-FAST) is an architecture **reference only** — this is a native
pi0.5 (flow) implementation.

## How it works (no model surgery)

The flow pi0.5 `PI0Pytorch.embed_prefix(images, img_masks, lang_tokens, lang_masks)`
already (a) iterates over **all** images in the observation and (b) embeds the full
`tokenized_prompt`, with the entire prefix attended **bidirectionally**. EgoVerse
already feeds a variable image set to that same model (eva = 3 real cams; aria = 1
real + 2 masked duplicates). So RICL is purely *extending the inputs*:

1. **Retrieved images** → appended as extra entries in the observation `images`
   dict (`retrieved_{i}_base_0_rgb`); the model embeds them as prefix vision tokens.
2. **Retrieved state + action** → discretized with the *same* binning as the
   pi0.5 State block (`PI._discretize_state_for_sample`) and spliced into the
   prompt text (before the `;\nAction:` anchor).

`PIRicl` (a thin `PI` subclass) does only this; `PI0Pytorch` is unchanged.

Locked design (see `/Users/ryan/.claude/plans/async-sleeping-marble.md`):
finetune from the **openpi pi0.5 base**; retrieval representation = **DINOv2**
(`torch.hub dinov2_vitb14`, replicating ricl_openpi's encoder exactly) over
`base_0_rgb`, pooled to the 64-super-patch descriptor (8×8 grid, 49152-d, raw
L2 — ricl_openpi's `EMBEDDING_TYPE='64PATCHES'`); **per-observation, k≈4**;
retrieved actions as **discrete tokens**; eval = **D0 eva→eva sanity → D1
aria→eva** (Tier-2 pairs).

## Files

| File | Role |
|------|------|
| `egomimic/ricl/retrieval.py` | DINOv2 pooling + cKDTree index + per-query top-k cache; pairs→task-groups; CLI + `--smoke`. |
| `egomimic/ricl/conditioning.py` | The prefix surgery: discretize retrieved state/action → prompt text; append retrieved images to the obs dict. (import-light) |
| `egomimic/ricl/data.py` | `RiclQueryDataset` (surfaces `frame_idx`), `ZarrBankFrameProvider`, `build_ricl_collate` (attaches `ricl_*` batch keys). |
| `egomimic/ricl/metrics.py` | Cartesian MSE/L1, gripper accuracy, retrieval-vs-floor comparison. |
| `egomimic/algo/pi_ricl.py` | `PIRicl(PI)` — three small overrides wire the conditioning in. |
| `egomimic/eval/pi_ricl_eval.py` | `PIRiclEval` — retrieval (k) vs zero-context floor (k=0) on held-out eva. |
| `egomimic/pl_utils/pl_data_utils.py` | `+ RiclDataModuleWrapper`. |
| `egomimic/hydra_configs/model/pi0.5_ricl.yaml` | `_target_: PIRicl`, k=4, `max_token_len: 1280`, openpi base. |
| `egomimic/hydra_configs/data/cotrain_pi_ricl.yaml` | `_target_: RiclDataModuleWrapper`, eva query + aria bank. |
| `egomimic/hydra_configs/evaluator/eval_pi_ricl.yaml` | `_target_: PIRiclEval`. |
| `egomimic/ricl/tests/{conditioning,data,metrics}_test.py` | CPU unit tests (torch+numpy, no openpi/data). |
| `egomimic/ricl/scripts/**` | Runnable utilities (build/extract/validate/viz, DROID trainer, sbatch). |

## Cluster runbook

**0. Prereqs.** The full `emimic` env (with `openpi`, the pi0.5 base checkpoint at
`pytorch_weight_path`, and R2 access). `human_robot_pairs.json` is in the repo.

**1. DINOv2-embed `base_0_rgb` for bank + query episodes.** Reuse the in-repo
pipeline to write `observations.embeddings.dinov2.front_1` (patch tokens,
`(T, 256, 768)`) into each episode's zarr:
```bash
python egomimic/scripts/embedding_process/zarr_embedding.py \
  --transform dinov2 \
  --dataset-config-path egomimic/hydra_configs/data/cotrain_pi_ricl.yaml \
  --input-keys observations.images.front_1 \
  --output-keys observations.embeddings.dinov2.front_1 \
  --batch-size 64 --device cuda
```

**2. Build the retrieval cache** (pools the patch tokens into the 64-super-patch
descriptor, builds a cKDTree per task group, caches per-query top-k).
`--zarr-root` is a `{hash}` template to each episode's zarr store:
```bash
# D1 cross-embodiment (bank = aria): one group per aria scene, queries = matched eva
python -m egomimic.ricl.retrieval --mode cross_similar -k 4 \
  --zarr-root '/path/to/zarr/{hash}.zarr' --out ricl_cache_cross_k4
# D0 within-embodiment sanity (bank = eva)
python -m egomimic.ricl.retrieval --mode within_alignment -k 4 \
  --zarr-root '/path/to/zarr/{hash}.zarr' --out ricl_cache_within_k4
```
Sanity-check neighbors against `shared_objects` in `human_robot_pairs.json`.

**3. Finetune pi0.5 with retrieval** (from the openpi base). Use
`trainer=ddp_pi_ricl callbacks=checkpoints_ricl` — step-based validation (every 250
steps) + best-by-`Valid/action_loss` + `last.ckpt`. Plain `ddp_pi` validates only
every 200 epochs and `callbacks=checkpoints` tracks no monitor, so neither the
val-loss curve nor the best operating point is captured (the DROID verification
found new-task val loss bottoms early, ~step 500, then overfits). `trainer.max_steps`
defaults to 6000 — override per run.
```bash
python -m egomimic.trainHydra \
  model=pi0.5_ricl data=cotrain_pi_ricl evaluator=eval_pi_ricl \
  trainer=ddp_pi_ricl callbacks=checkpoints_ricl \
  data.retrieval_cache_dir=ricl_cache_cross_k4 \
  data.bank_zarr_root='/path/to/zarr/{hash}.zarr'
# D0 sanity uses bank=eva: also override the bank converter:
#   data.retrieval_cache_dir=ricl_cache_within_k4 \
#   data.bank_converter._target_=egomimic.utils.action_utils.RobotBimanualCartesianEuler
```
Hold out the eval task from the eva training filter so the cross-embodiment claim
is meaningful (edit the eva filter in `data=cotrain_pi_ricl`, inherited from `eva_pi`).

**4. Evaluate (retrieval vs zero-context floor).** `eval_pi_ricl` reports
`RICL/retrieval_*`, `RICL/floor_*`, `RICL/delta_*`, `RICL/retrieval_helps`.
**D0 (eva→eva) must beat its floor before trusting D1 (aria→eva).**

## Design notes / knobs

- **k & token budget**: k=4 retrieved images add ~1024 prefix vision tokens;
  retrieved state+action add text. The text block is encoded at the **full 32-D**
  openpi layout (`state_dim 32`, `action_dim 32` in `cotrain_pi_ricl.yaml`) so the
  bins stay in pi0.5's pretrained distribution — not compacted. That makes the k=4
  worst case ~1150 tokens, so `max_token_len`/`tokenizer_max_length` is 1280.
  Truncation drops the trailing `;\nAction:` anchor, so the budget must always
  exceed the worst case: re-check with
  `python -m egomimic.ricl.scripts.check_prompt_budget` (tokenizes the real
  worst-case prompt; exits non-zero if it would overflow) whenever you change
  `num_retrieved_observations`, `retrieved_action_steps`, or `max_token_len`.
- **Shared 32-D action space**: eva (14-D) and aria (12-D, no gripper→slot 0) are
  both mapped to the same 32-D layout by the converters, so retrieved-aria and
  query-eva bins are directly comparable (no manual padding).
- **Retrieval view**: `base_0_rgb` only (the single view both embodiments share;
  eva exterior vs aria ego — a real domain gap). If DINOv2 distances saturate,
  add per-task distance normalization (cf. ricl_openpi `max_distance.json`).
- **`num_workers`**: the collate closure holds the cache + a lazily-opened
  per-worker zarr handle; fine under fork (Linux), but watch worker memory.
