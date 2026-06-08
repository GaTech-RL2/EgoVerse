# RICL on EgoVerse pi0.5 — retrieval-based in-context learning

Adds **retrieval-based in-context learning (RICL)** to EgoVerse's own **pi0.5**,
reusing the existing zarr / Cartesian / `trainHydra` / `PI`-algo / DINOv3 stack.
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
finetune from the **openpi pi0.5 base**; retrieval representation = **DINOv3**
(`facebook/dinov3-vitb16-pretrain-lvd1689m`, 768-d) over `base_0_rgb`, mean-pooled;
**per-observation, k≈4**; retrieved actions as **discrete tokens**; eval =
**D0 eva→eva sanity → D1 aria→eva** (Tier-2 pairs).

## Files

| File | Role |
|------|------|
| `egomimic/ricl/retrieval.py` | DINOv3 pooling + cKDTree index + per-query top-k cache; pairs→task-groups; CLI + `--smoke`. |
| `egomimic/ricl/conditioning.py` | The prefix surgery: discretize retrieved state/action → prompt text; append retrieved images to the obs dict. (import-light) |
| `egomimic/ricl/data.py` | `RiclQueryDataset` (surfaces `frame_idx`), `ZarrBankFrameProvider`, `build_ricl_collate` (attaches `ricl_*` batch keys). |
| `egomimic/ricl/metrics.py` | Cartesian MSE/L1, gripper accuracy, retrieval-vs-floor comparison. |
| `egomimic/algo/pi_ricl.py` | `PIRicl(PI)` — three small overrides wire the conditioning in. |
| `egomimic/eval/pi_ricl_eval.py` | `PIRiclEval` — retrieval (k) vs zero-context floor (k=0) on held-out eva. |
| `egomimic/pl_utils/pl_data_utils.py` | `+ RiclDataModuleWrapper`. |
| `egomimic/hydra_configs/model/pi0.5_ricl.yaml` | `_target_: PIRicl`, k=4, `max_token_len: 512`, openpi base. |
| `egomimic/hydra_configs/data/cotrain_pi_ricl.yaml` | `_target_: RiclDataModuleWrapper`, eva query + aria bank. |
| `egomimic/hydra_configs/evaluator/eval_pi_ricl.yaml` | `_target_: PIRiclEval`. |
| `egomimic/ricl/{conditioning,data,metrics}_test.py` | CPU unit tests (torch+numpy, no openpi/data). |

## Local (M4) vs cluster

This box (Apple M4, no CUDA, no `openpi`/`egomimic` env) can verify **correctness
of the logic**, not run the 3B model. What is verified locally (all green):

```bash
PY=/Users/ryan/.ricl-smoke-venv/bin/python   # a CPU venv with torch+numpy+scipy
$PY egomimic/ricl/retrieval.py --smoke        # pooling, pairs→groups, cKDTree, cache I/O
$PY egomimic/ricl/conditioning_test.py        # discretize/splice/image-augment + mock embed_prefix grad
$PY egomimic/ricl/data_test.py                # collate shapes/mask, frame_idx, image norm/resize
$PY egomimic/ricl/metrics_test.py             # MSE/L1, gripper acc, floor comparison
```
(or `pytest egomimic/ricl/*_test.py` inside the full env.)

Everything below needs the GPU cluster (real `openpi` + R2 data).

## Cluster runbook

**0. Prereqs.** The full `emimic` env (with `openpi`, the pi0.5 base checkpoint at
`pytorch_weight_path`, and R2 access). `human_robot_pairs.json` is in the repo.

**1. DINOv3-embed `base_0_rgb` for bank + query episodes.** Reuse the in-repo
pipeline to write `observations.embeddings.dinov3.front_1` (patch tokens) into each
episode's zarr:
```bash
python egomimic/scripts/embedding_process/zarr_embedding.py \
  --transform dinov3 \
  --dataset-config-path egomimic/hydra_configs/data/cotrain_pi_ricl.yaml \
  --input-keys observations.images.front_1 \
  --output-keys observations.embeddings.dinov3.front_1 \
  --batch-size 64 --device cuda
```

**2. Build the retrieval cache** (mean-pools the patch tokens, builds a cKDTree per
task group, caches per-query top-k). `--zarr-root` is a `{hash}` template to each
episode's zarr store:
```bash
# D1 cross-embodiment (bank = aria): one group per aria scene, queries = matched eva
python -m egomimic.ricl.retrieval --mode cross_similar -k 4 \
  --zarr-root '/path/to/zarr/{hash}.zarr' --out ricl_cache_cross_k4
# D0 within-embodiment sanity (bank = eva)
python -m egomimic.ricl.retrieval --mode within_alignment -k 4 \
  --zarr-root '/path/to/zarr/{hash}.zarr' --out ricl_cache_within_k4
```
Sanity-check neighbors against `shared_objects` in `human_robot_pairs.json`.

**3. Finetune pi0.5 with retrieval** (from the openpi base):
```bash
python -m egomimic.trainHydra \
  model=pi0.5_ricl data=cotrain_pi_ricl trainer=ddp_pi evaluator=eval_pi_ricl \
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
  retrieved state+action add text → `max_token_len`/`tokenizer_max_length` bumped
  to 512. Tune `num_retrieved_observations`, `retrieved_action_steps` (default 1),
  and `max_token_len` together.
- **Shared 32-D action space**: eva (14-D) and aria (12-D, no gripper→slot 0) are
  both mapped to the same 32-D layout by the converters, so retrieved-aria and
  query-eva bins are directly comparable (no manual padding).
- **Retrieval view**: `base_0_rgb` only (the single view both embodiments share;
  eva exterior vs aria ego — a real domain gap). If DINOv3 distances saturate,
  add per-task distance normalization (cf. ricl_openpi `max_distance.json`).
- **`num_workers`**: the collate closure holds the cache + a lazily-opened
  per-worker zarr handle; fine under fork (Linux), but watch worker memory.
- **Future fidelity**: this uses one bidirectional prefix (the natural pi0 layout).
  `pi0_fast_ricl`'s block-causal mask (retrieved blocks can't see the query) is a
  possible upgrade but needs custom attention in `PI0Pytorch`. Action interpolation
  (`exp(-λ·dist)`) was intentionally dropped for the flow model (plan axis C1).
