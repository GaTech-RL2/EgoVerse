# Training-speed experiments (H100, `ryanco/torch-compile-train`)

Where the HPT and Pi train steps actually spend their time on the `loaner`
H100 nodes, what was changed, and what each change bought. Every number below
is one H100 (gpu11), bf16 autocast, measured with the harnesses in this folder.

## The harnesses

| file | what it does |
| --- | --- |
| `bench_hpt.py` | HPT train step (`forward_training` + `compute_losses` + backward + optimizer) on a synthetic batch, real model config. `--variants` A/Bs one change at a time; `--profile` dumps a torch-profiler table. |
| `bench_pi.py` | The same for Pi (openpi pi0.5), random init -- step time does not depend on the base weights. |
| `check_compile_parity.py` | Eager vs compiled, same weights, same batch: is `torch.compile` changing the numbers? |
| `check_attn_parity.py` | `need_weights=True` (the old path) vs `False`, same weights, same batch. |
| `check_aug_distribution.py` | Two-sample KS between the per-image and batched jitter, over per-sample image statistics. |
| `ab_augs.sh`, `ab_compare.py` | Accuracy A/B for the aug rewrite: both paths, two seeds, held-out loss on disjoint episodes. |
| `e2e_hpt.sh`, `e2e_sweep.sh` | The real `trainHydra` path on locally cached mecka episodes, so the dataloader is in the measurement. |

```bash
salloc -p loaner -A loaner --gres=gpu:h100:1 -c 28 --mem=200G
source ./wt-env.sh
python bench/bench_hpt.py --variants base,compile_hook --steps 25 --warmup 10

# accuracy A/B: ~12 min an arm on one H100
for s in 42 43; do for arm in loop batched; do
  AB_SEED=$s AB_EPOCHS=20 bench/ab_augs.sh $arm <results>/${arm}_s${s}.json
done; done
python bench/ab_compare.py <results>
```

`bench_pi.py` needs `.venv-pi` (see "The venv", below), not the shared `emimic/`.

## What the baseline was doing

The 300M HPT step at batch 64 was **not** limited by its own arithmetic. The
profiler put ~14,600 CUDA kernel launches in one step, 230 ms of CPU against
111 ms of GPU: the step was bound by the rate the host could issue work. The
model's matmuls accounted for ~26 ms of that.

Two things dominated the launch count:

1. **Per-sample image augmentation.** `PerSampleAugs` called torchvision's
   `ColorJitter` once per image so each sample drew its own factors. At batch
   64 that is 64 Python round trips, each launching a few dozen kernels and
   copying its factors from the host: ~470 `cudaMemcpyAsync` per step, and
   **89 ms of a 196 ms step**.
2. **Single-tensor AdamW.** Not a bug in the repo -- `torch.optim.AdamW` with
   neither `foreach` nor `fused` set picks `foreach` automatically, which the
   shipped configs get. It is called out because passing `fused=False`
   *explicitly* silently turns the foreach default off too, which is how the
   first version of this benchmark ended up 17 ms/step slower than the real
   training loop.

## Changes

| change | where |
| --- | --- |
| Vectorized per-sample `ColorJitter`: per-sample factors drawn on-device, each adjustment run once over the batch | `egomimic/models/image_augs.py` |
| `torch.compile` knob: `model.compile.enabled=true` -> `<algo>.compile_for_training()` | `pl_utils/pl_model.py`, `utils/compile_utils.py`, `algo/{algo,hpt,pi}.py` |
| `need_weights=False` on the flow head's attentions, so they dispatch to `scaled_dot_product_attention` instead of materialising per-head scores | `egomimic/models/denoising_nets.py` |
| Decode frames to float32 instead of numpy's float64 default, halving what the collate, the pin and the H2D copy move (338 -> 169 MB per camera per batch at 360x640, batch 64; bit-identical) | `egomimic/rldb/zarr/zarr_dataset_multi.py` |
| `non_blocking=True` on the batch's H2D copy (a no-op today, needed by anything that later pins) | `egomimic/algo/{hpt,pi}.py` |
| Drop a per-camera, per-step device->host sync (`if torch.all(img == 0)`) | `egomimic/algo/hpt.py` |
| Skip the per-step deep copy of the batch unless the OT loss is on | `egomimic/algo/hpt.py` |

`compile_for_training` uses `nn.Module.compile`, not `torch.compile(module)`:
the former patches the call in place, the latter returns an `OptimizedModule`
whose parameters are all renamed `_orig_mod.*`, which would break every
checkpoint written before the flag was turned on. The flip side is that
`Module.compile` only routes `__call__`, so a submodule the algo invokes
through a named method (`stem.compute_latent`, `head.compute_loss`) stays
uncompiled no matter what -- that is why HPT compiles the trunk, the flow
head's denoiser and the image encoders and nothing else, and why `PI`'s call
site moved from `.forward(...)` to `(...)`.

## Results

### HPT model step, isolated (`bench_hpt.py`)

One H100, `hpt_bc_mecka_6d_300M` (298 M trainable), batch 64, bf16 autocast,
synthetic batch, `AdamW` as the configs have it (no explicit `foreach`/`fused`),
including the two grad-norm passes `ModelWrapper` runs every step.

360x640 -- the resolution mecka frames actually decode to:

| | ms/step | vs before | peak GB |
| --- | ---: | ---: | ---: |
| before (per-image `ColorJitter`) | 245.2 | 1.00x | 16.8 |
| vectorized augs | 173.1 | 1.42x | 16.8 |
| + `model.compile.enabled=true` | 109.4 | **2.24x** | 16.3 |

224x224, for comparison with anything already measured at that size:

| | ms/step | vs before | peak GB |
| --- | ---: | ---: | ---: |
| before | 205.3 | 1.00x | 14.0 |
| vectorized augs | 120.2 | 1.71x | 13.9 |
| + compile | 82.1 | 2.50x | 13.8 |
| + compile, `mode=reduce-overhead` | 77.7 | 2.64x | 13.8 |

Compile costs ~15-40 s at the first batch (it grows with image size) and
`reduce-overhead` roughly doubles that for another 5%. Inductor's cache is
keyed per Slurm job (`utils/compile_cache.py`), so every job -- including every
requeue -- pays it once.

### Pi model step, isolated (`bench_pi.py`)

One H100, `pi0.5_bc_mecka_6d` (3.6 B), batch 16, random init:

| | ms/step | vs base |
| --- | ---: | ---: |
| base | 826.8 | 1.00x |
| `fused` AdamW | 794.8 | 1.04x |
| `model.compile.enabled=true` | 588.1 | 1.41x |
| compile + `fused` AdamW | 560.1 | **1.48x** |

Compile costs ~4 min at the first batch. Forcing the gemma tower back to `sdpa`
attention did nothing for training -- openpi only pins it to `eager` inside
`sample_actions`, which the training pass never enters.

### Is compile changing the numbers? (`check_compile_parity.py`)

With every source of randomness pinned (dropout, drop-path, colour jitter, the
flow-matching noise and time) and both copies starting from identical weights:

* **fp32: the loss is bit-identical.** The compiled graph is the same function.
* The remaining fp32 gradient differences are confined to the ResNet's conv
  weights (cosine >= 0.9989), which is cudnn's non-deterministic
  backward-filter, not compile.
* Under bf16 autocast the loss drifts ~0.3% on a random-init network, which is
  reduction-order noise amplified through 23 trunk blocks.

### Does the aug rewrite change what the model learns? (`ab_augs.sh`, `ab_compare.py`)

The vectorized `ColorJitter` is the only change here that is not exactly
equivalent to what it replaced. Everything else is either bit-identical
(float32 decode, the OT clone, the `torch.where` camera select, `non_blocking`)
or floating-point reassociation (compile, `need_weights`). The jitter is
different in two ways, both deliberate:

* the **op order** is drawn once per call, not once per image, so samples in a
  batch share a permutation. A single sample's marginal is unaffected -- the
  order is still uniform over the 4! permutations and the factors are still
  per-sample -- but within-batch independence is gone.
* the RNG stream differs, so no run reproduces an old seed.

`check_aug_distribution.py` covers the marginal: 6400 augmented samples through
each path, two-sample KS on eight per-sample statistics (per-channel mean and
std, min, max). Worst p = 0.61. The marginals are indistinguishable.

The within-batch correlation only a training run can see. `ab_augs.sh` runs both
arms at two seeds: same config, same 32 train episodes, compile off, 2000 steps,
held-out loss on 15 **disjoint** episodes every 100 steps. The held-out pass runs
in eval mode -- deterministic eval augs -- and reseeds, so it measures what the
model learned, not what it was shown. The `loop` arm stubs out `_vectorize` so
`PerSampleAugs` takes its own per-image fallback: the old code path, not a
reimplementation of it.

| arm | seed | train (last 100) | held-out (mean of last 5) | wall s |
| --- | ---: | ---: | ---: | ---: |
| loop | 42 | 0.617 | 0.752 | 740 |
| loop | 43 | 0.647 | 0.822 | 734 |
| batched | 42 | 0.585 | 0.685 | 589 |
| batched | 43 | 0.598 | 0.631 | 566 |

**No regression: the batched arm is better at both seeds** (-9.0%, -23.2%), and
the 1.26x wall-clock gap independently confirms the `loop` arm really ran the
per-image path.

Do NOT read that as the rewrite improving accuracy. Two seeds cannot support
that claim, the two arms consume different amounts of RNG so each arm/seed pair
is effectively its own seed, and the held-out loss still swings 0.6-1.2 between
adjacent evals at step 2000. `ab_compare.py` prints `ARM EFFECT EXCEEDS SEED
NOISE` because the mean gap (0.129) is larger than the within-arm seed spread
(0.070); with n=2 that is weak evidence of a real difference and no evidence at
all of its direction. What it does rule out is the thing worth ruling out --
sharing the op order across a batch is not costing accuracy. Confirming the
apparent gain would need ~6 seeds an arm.

### Does `need_weights=False` change the numbers? (`check_attn_parity.py`)

The flow head's attentions used to take `nn.MultiheadAttention`'s default
`need_weights=True`, which forces the materialized-softmax path; they now
dispatch to `scaled_dot_product_attention`. Forcing the old path back on and
comparing a full fp32 train step:

* loss 197.009293 vs 197.009033, a relative delta of 1.3e-6
* 581 gradient tensors, min cosine similarity 0.999999, worst relative max-abs
  delta 1.1e-3 (all on ResNet conv weights, i.e. the same cudnn
  backward-filter nondeterminism the compile parity check sees)

Same function, different reduction order.

### End to end on real data (`e2e_hpt.sh`)

The real `trainHydra` path, 20 locally cached mecka episodes, 5 x 40 batches,
median of the steps after the first epoch (`bench/step_timer.py` -- Lightning's
`simple` profiler averages the compile and the worker spin-up into every step,
which at 240 steps is a 70 ms/step artefact):

| | step ms (median) | step ms (p10) | batch wait ms |
| --- | ---: | ---: | ---: |
| before | 298.6 | 266.2 | 41.1 |
| vectorized augs + float32 decode | 274.4 | 196.7 | 41.6 |
| + compile | 245.8 | 157.9 | 40.0 |

Read the p10 column. The isolated model step drops 64 ms when compiled
(173 -> 109) and the real run's *fast* steps drop 59 ms (217 -> 158), which
matches; the median only drops 33 ms because the real step time is bimodal,
with roughly +/-70 ms of input-side jitter on top of every step. Compile is
doing what the microbenchmark says it does -- the run just spends a chunk of
each step somewhere else.

That "somewhere else" is not worker count. 8 / 24 / 48 workers give 270.9 /
279.4 / 279.2 ms with a 40 ms batch wait throughout, so the pool is not the
limit and raising `num_workers` off the configs' 8 buys nothing here. The
constant ~40 ms wait is the main process receiving a 169 MB batch out of the
worker queue, which more workers cannot parallelise.

## Things that were tried and did not pay

Worth recording so nobody re-runs them:

| | result |
| --- | --- |
| `persistent_workers=True` on the train loader | No effect. The epoch-start wait is ~1.6 s either way (measured per epoch), and the steady-state step is identical (272.9 vs 274.4 ms). |
| `pin_memory=True` (+ `non_blocking` H2D) | A wash. It moves the 41 ms batch wait into the step (wait 0.9 ms, step 351 vs 312) -- the pinning thread does the copy the main thread used to. It would only win with a stream prefetcher that issues the next batch's H2D during the current step. |
| `prefetch_factor=4` | No measurable effect. |
| `torch.set_float32_matmul_precision("high")`, `cudnn.benchmark` | Under bf16 autocast there is almost no fp32 matmul left to speed up: <1%. |
| `channels_last` for the ResNet | 2-3% in the isolated benchmark, inside the run-to-run noise end to end. |
| `mode="max-autotune-no-cudagraphs"` | 697 s of autotuning for 0.3%. Not worth it. |
| `mode="reduce-overhead"` (CUDA graphs) | A further 5% for roughly double the compile time. Left off by default. |
| Caching the sinusoid position tables | ~0.2%. The tables are rebuilt on the CPU and copied to the device every forward, which is ugly, but it is not where the time goes. |
| `fused=True` AdamW | 5% on HPT, 4% on Pi. Not wired into the configs: it is a numerics change to the optimizer and belongs in its own PR, not this one. Note that passing `fused=False` **explicitly** also disables the `foreach` default and costs 17 ms/step -- the configs correctly pass neither. |

## Loose ends

* **`ModelWrapper` never receives `enable_grad_norm`.** `trainHydra` constructs
  it with `config_tree`, `norm_stats_state` and `scheduler_interval` only, so
  the `enable_grad_norm: false` in `model/pi0.5_base.yaml` has no effect and
  every run pays both grad-norm passes. They are 6.5 ms/step on HPT (two full
  `clip_grad_norm_(inf)` sweeps over 298 M parameters plus two host syncs), and
  one of the two is documented in `pl_model.py` as returning the same pre-clip
  number as the other.
* **The shared `emimic/` venv is off-pin.** It has `transformers` 5.17.0 against
  `pyproject.toml`'s 4.53.2; openpi's `transformers_replace` patch is for 4.53
  and the combination breaks Pi (`PaliGemmaForConditionalGeneration` has no
  `language_model`) and `tests/fixtures/train_harness.py`
  (`transformers.utils.hub._is_offline_mode` was removed in 5.x). One unit test
  fails there for that reason alone. This worktree carries a pinned `.venv-pi`
  (git-ignored) that the Pi benchmarks use; the whole unit suite passes in it
  (1127 passed, 8 skipped).
* **A stream prefetcher** is the obvious next input-side move: issue the next
  batch's H2D on a side stream during the current step, which is what would
  make `pin_memory` pay.
* **uint8 images** would cut the batch another 4x (169 MB -> 42 MB) over the
  float32 change, moving the `/255` and the transpose onto the GPU. Bigger
  blast radius -- the transform list and norm stats would need checking.
