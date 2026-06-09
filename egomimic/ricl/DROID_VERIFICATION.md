# Verifying the RICL pipeline on original-RICL (DROID) data

**Goal.** Prove EgoVerse's RICL machinery doesn't just *run* but *works the way RICL
is supposed to*: train the real `PIRicl` on one set of tasks, then on a **separate set
of genuinely new tasks** show that conditioning on retrieved in-context demos beats
both a zero-context floor *and* a plain finetune — i.e. the model learns to *use*
retrieved demos to do a task its weights never trained on.

**Not a port.** The original RICL paper's DROID demos (a clean, known-good retrieval
dataset) are fed through a thin in-memory shim — no Zarr writing, no SQL, no new
embodiment, no `trainHydra`. The model, collate, prefix conditioning,
optimizer/grad-norm, Lightning `Trainer`, and eval are all **production code**; only
the Zarr data layer and the Hydra entrypoint are bypassed.

---

## Experiment structure

### 1 · Train / eval split (mirrors `ricl_openpi`'s `baseline` mode)

The eval tasks are **structurally different** from the train tasks and the weights
never see them — so retrieved demos are the *only* source of task information.

- **Train** — `collected_demos_training/`, 11 tasks, 218 demos, 15,415 frames. The
  model finetunes here (retrieve within-group → splice into prefix → predict),
  learning the *in-context mechanism*:
  `move_apple_to_the_right`, `move_box_to_the_left`, `move_cup_to_the_left`,
  `move_duck_to_the_left`, `move_orange_to_the_right`, `move_pan_to_the_left`,
  `move_strawberry_to_the_right`, `move_the_coffee_pod_to_the_right`,
  `move_the_cup_to_the_right`, `move_pot_to_the_left`,
  `pick_up_a_coffee_pod_from_the_container_and_put_it_in_the_bowl`.
- **Eval** — `collected_demos/`, **9 NEW tasks**, 182 demos, ~10.3k frames:
  `move_the_idli_plate_to_the_right`, `pick_up_the_poke_ball_and_put_it_in_the_tray`,
  `move_the_squeegee_to_the_right_and_try_to_drag_it`,
  `open_the_door_of_the_bottom_shelf`, `pick_up_the_bagel_and_put_it_in_the_toaster`,
  `pick_up_the_bagel_and_put_it_in_the_toaster_other_side`,
  `push_the_lever_on_the_toaster`, `move_the_idli_plate_to_the_sink`,
  `use_the_squeegee_to_clean_the_counter_and_push_everything_into_the_sink`.
  (The `-5demos` / `-10demos` idli subset dirs are dropped so each task counts once.)
  An eval query retrieves its k demos from **its own new task's** bank (within-group
  leave-one-out).

> Earlier within-task splits (holding out *episodes*, or a few *training* tasks) were
> discarded: the model had already seen the task, so the floor was strong and "does
> retrieval help on a **new** task" was never actually tested.

### 2 · Eval conditions + the plain-finetune baseline

Two kinds of comparison, both scored on the new tasks under one shared per-batch seed
(so deltas isolate the conditioning, not the sampling RNG):

**(a) The RICL checkpoint at four inference conditions** — same weights, different
demos in the prompt:

| condition | demos shown | isolates |
|---|---|---|
| **retrieval** | k **nearest** (kNN) from the query's own new task | — |
| **random-within** | k **random** from the query's **own** task | does kNN *ranking* matter, given the right task? |
| **random-bank** | k **random** from **all** new tasks pooled | does the *right task* matter at all? |
| **floor** | none (k=0, clean) | does any in-context demo help? |

**(b) A separately-trained plain finetune** (`--no-incontext`) — a vanilla pi0.5
finetune on the same 11 train tasks whose weights **never saw a retrieved demo**. This
is the honest "what does plain finetuning give you" number that condition (a)'s k=0
*floor cannot provide* — the floor reuses the RICL weights (trained *with* the
mechanism) denied demos at eval. Same val collate, so it's scored head-to-head.

### 3 · What makes retrieval actually work (three pieces from `ricl_openpi`)

A direct comparison against `external/ricl_openpi` (`pi0_fast_droid_ricl`:
`action_horizon=15, num_retrieved_observations=4, use_action_interpolation=True`)
surfaced three things the pipeline lacked. Without them retrieval merely ties random;
with them it beats random *and* floor.

- **#1 Full retrieved action chunk** — was 1 step (near-zero, indistinguishable across
  kNN vs random); now the bank provider + collate carry the full `action_horizon`
  chunk (`--retrieved-action-steps`, default = full). Worst-case k=4 prompt ≈ 2150
  tok → `--max-token-len 2304`.
- **#4 Per-demo exemplar prompt** — was one `Demos: …` blob; now each demo is a
  self-contained `Task: …, State: …;\nAction: …|` exemplar prepended before the query
  block (demos-then-query order), mirroring ricl_openpi's per-observation blocks.
  Prompt structure only, no attention-mask surgery; shared with the eva path
  (`conditioning.py` / `pi_ricl.py`).
- **#2 Distance-weighted action interpolation** — **inference-only**, not trained-in.
  At sampling time, blend the predicted action toward the nearest demo's chunk:
  `a ← w·a_nn + (1−w)·a_model`, `w = exp(−λ·dist/dist_max)` (the continuous-flow analog
  of ricl_openpi's logit interpolation). Distances are normalized by the **batch-max
  kNN distance**, so the original's λ=10 is *inactive* here (w≈0) — the operating point
  is **λ≈1**. (`build_random_neighbor_cache` also fixed to store the true embedding
  distance so random demos don't get a fake interpolation weight.)

### 4 · The metric: sampled-action MSE, **not** flow loss

The flow-matching (velocity-matching) loss is the wrong yardstick — it's insensitive
to the difference the *integrated* action reveals, showing retrieval ≈ random
(retrieval−random ≈ −0.001) even when the real action error shows a clear gap.
Routine validation logs flow loss (cheap); **judge RICL on the sampled-action MSE**
(`--compute-sampled`, the slow `sample_actions` path the original repo reports).
`--flow-samples N` averages N noise/time draws — a *single* draw has noise as large as
the retrieval−floor gap and produced two wrong early reads (a phantom "overfits at
step 500" and a phantom "similarity edge shrinks").

---

## Results

Run `fullchunk_v1` (train on the 11 tasks, eval on the 9 new tasks; 2500 steps, bs2,
k4, ah15, lr 3e-5) vs the `baseline_noicl` plain finetune (**identical config**,
`--no-incontext`, same 2500/bs2/seed42). **Both `last.ckpt`s were scored in one matched
validate pass — same 25 val batches, same frames, `--flow-samples 8` — so the
head-to-head is strictly apples-to-apples** (`ricl_match_validate` / `baseline_noicl_validate`).
Sampled-action MSE on the new tasks (lower = better):

| model / condition | no-interp | interp λ=1 |
|---|--:|--:|
| **RICL — retrieval (nearest)** | **0.171** | **0.126** ✅ best |
| RICL — random-within (same-task random) | 0.168 | 0.137 |
| RICL — random-bank (cross-task) | 0.165 | 0.160 |
| RICL — floor (k=0) | 0.353 | — |
| **plain finetune — floor (no demos, its natural mode)** | **0.224** | — |
| plain finetune — *with* retrieved demos | 0.308 | 0.170 |

**What it shows.**

1. **The in-context architecture beats a plain finetune — the headline.** Every RICL
   demo condition (0.165–0.171, or 0.126–0.160 with interpolation) beats the plain
   finetune's *best* new-task number — its no-demo floor, **0.224**. RICL retrieval is
   **−24%** (0.171 vs 0.224); with interpolation λ=1, **−44%** (0.126 vs 0.224). The
   weights never trained on these tasks, so the win comes from the in-context machinery.
2. **It's the architecture, not just inference-time demo access.** Feeding the *same*
   retrieved demos to the plain finetune makes it **worse** (0.308 vs its 0.224 floor):
   weights that never learned to read in-context demos just see distribution shift. The
   0.308 → 0.171 gap between the two models *given identical demos* is the value of
   training with the mechanism.
3. **Why the separate baseline mattered (the gap the k=0 floor hid).** The plain
   finetune is *better* at bare zero-context than the RICL model's own k=0 floor
   (**0.224 vs 0.353**) — it spent no capacity learning to lean on demos, so its
   no-demo prediction is stronger. So the RICL-floor was a *flattering*, not honest,
   baseline; against the true plain finetune RICL still wins, but the comparison is now
   fair. RICL uses retrieval to go below the plain finetune's 0.224 ceiling.
4. **Retrieval ≥ random, amplified by interpolation.** On this 25-batch subset the
   model's raw predictions rank retrieval ≈ random (0.171/0.168/0.165 — a tie within
   noise, as the doc warns for small deltas); interpolation (#2, λ≈1) separates them
   (retrieval 0.126 < random-within 0.137 < random-bank 0.160), because the nearest
   neighbor sits closer and earns a larger blend weight — distance-weighting working as
   designed.

> **Originally-published headline (15 val batches), for the record.** The first
> `fullchunk_v1` write-up scored RICL on 15 batches: retrieval **0.137** (interp λ=1
> **0.094**), random-within 0.144, random-bank 0.150, floor 0.246 — i.e. retrieval
> cleanly beat both random and floor there. The matched 25-batch pass above adds 10
> harder batches (raising every absolute level) but is the *only* like-for-like
> comparison to the baseline, so it's the authoritative head-to-head; the plain-finetune
> win holds under both.
>
> **Earlier (weaker) configuration.** Before #1/#4/#2, retrieval beat the floor by only
> ~13–16% sampled MSE and merely *tied* random. The full chunk + exemplar prompt +
> interpolation are what turned the similarity signal from negligible into usable.

Reproduce: `--validate-only --compute-sampled --interp-lamdas "0.5,1,2,5,10"` on each
`last.ckpt`; per-condition metrics in
`outputs/droid_train/{ricl_match_validate,baseline_noicl_validate}/version_0/metrics.csv`
(baseline training trajectory in `baseline_noicl/version_0/metrics.csv`).

---

## Files

- `droid_data.py` — `DroidCorpus` (lazy per-array npz loader), `DroidQueryDataset`,
  `make_droid_bank_provider`, `build_droid_retrieval_cache` (within-group LOO kNN over
  the pre-pooled `top_image_embeddings`), `build_random_neighbor_cache` (within/bank
  controls), `DroidNormStats`/`Passthrough32`. CLI: `--build-cache`.
- `droid_eval.py` — `DroidRiclEval` (the four conditions vs a *true* zero-context
  floor + interpolation) and `DroidRiclModelWrapper`.
- `scripts/train_droid_ricl.py` — `--stage cpu` (data/collate/prompt/token checks, no
  model) and `--stage full` (PIRicl + ModelWrapper + Trainer fit). Key flags:
  `--eval-root` (selects the new-task split), `--no-incontext` (plain-finetune
  baseline), `--retrieved-action-steps`, `--interp-lamdas`, `--max-token-len`.
- `scripts/*.sbatch` (`baseline_noicl.sbatch`, etc.); `tests/droid_data_test.py`.

## Key design decisions

- **8→32 mapping.** DROID is single-arm 8-D; the model works in pi0.5's shared 32-D
  space. The query's continuous state/action are quantile-normalized to [-1,1] and
  slot-filled into dims 0..7 (no-op `Passthrough32` keeps them 32-D). Retrieved demos'
  state/action are **text-only** (the real 8 dims) — encoding the 24 pad-dims ballooned
  the prompt to ~1200 tokens; 8-D keeps it ~334.
- **Clean floor (strip before tokenize).** `process_batch_for_training` splices + tokenizes
  the demos, so stripping `ricl_*` *after* leaves demo text baked into the prompt. The
  eval strips `ricl_*` from the **raw** batch and *then* processes each condition →
  genuine k=0 floor. Now also fixed in production `PIRiclEval` (`wants_raw_batch`).
- **Retrieval.** Within new-task LOO over `top_image_embeddings` (ricl_openpi retrieves
  on the top view), raw-L2, k=4.

## Run

EgoVerse2's own venv lacks openpi; use the sibling env + this repo on the path
(`source /storage/project/r-dxu345-0/rco3/EgoVerse/emimic/bin/activate;
export PYTHONPATH=/storage/project/r-dxu345-0/rco3/EgoVerse2`).

```bash
EVAL=/storage/project/r-dxu345-0/rco3/ricl_openpi/preprocessing/collected_demos

# 1) build the train-task retrieval cache once (GPU node; torch.cdist for distances)
python -u egomimic/ricl/droid_data.py --build-cache --k 4 \
  --out egomimic/ricl/outputs/droid_cache

# 2) train + new-task eval (compile ON for the real run). Drop --no-incontext for RICL.
python -u egomimic/ricl/scripts/train_droid_ricl.py --stage full [--no-incontext] \
  --cache-dir egomimic/ricl/outputs/droid_cache --eval-root "$EVAL" --groups 0 \
  --max-steps 2500 --warmup-steps 200 --lr 3e-5 --seed 42 \
  --action-horizon 15 --k 4 --max-token-len 2304 --batch-size 2 \
  --val-every 250 --limit-val-batches 25 --ckpt-every 1000 --flow-samples 8 --name <run>

# 3) refined eval of a finished checkpoint, no retraining (eager via TORCH_COMPILE_DISABLE):
TORCH_COMPILE_DISABLE=1 python -u egomimic/ricl/scripts/train_droid_ricl.py --stage full \
  --validate-only --resume-from egomimic/ricl/outputs/droid_train/<run>/version_0/checkpoints/last.ckpt \
  --eval-root "$EVAL" --groups 0 --action-horizon 15 --k 4 --max-token-len 2304 \
  --flow-samples 8 --compute-sampled --interp-lamdas "0.5,1,2,5,10" \
  --batch-size 8 --limit-val-batches 25 --name <run>_validate
```

> **Always keep checkpoints** — best (by `Valid/action_loss`) + `last.ckpt` every
> `--ckpt-every` steps, so new eval conditions are added by `--resume-from` not retrain.

## Gotchas

- **Short / eval-only runs: `TORCH_COMPILE_DISABLE=1`.** `--compute-sampled` triggers a
  `torch.compile` max-autotune warmup (minutes on the first call) — pure overhead for a
  one-shot eval. Eager skips it. Leave compile ON for real training.
- **Use `sbatch`, not interactive `salloc`, for unattended jobs.** On the shared
  `inferno` `gpu-h200` queue salloc allocations get preempted mid-run; a wedged `srun`
  hangs at *"Requested nodes are busy"* (clear with `scancel <jobid>.<step>`, relaunch
  `srun --overlap`).
- **Don't build the retrieval cache on the login node.** numpy 2.4.x BLAS here is ~0.4
  GFLOP/s (one 49152-D matmul = 13.7 s) and a cKDTree in 49152-D is pathological, so
  distances use `torch.cdist` (per-group brute force) on a GPU/many-core node.
- `processed_demo.npz` is uncompressed but `np.load(...)[key]` still materializes the
  whole array — load lazily per-key so the cache build never decompresses images. Skip
  dot-dirs (`.git/`) in group discovery.
- The `--stage cpu` path asserts `ricl_*` keys exist, so it's incompatible with
  `--no-incontext`; smoke-test that flag with a short `--stage full` run instead.
