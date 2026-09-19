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
| `e2e_hpt.sh`, `e2e_sweep.sh` | The real `trainHydra` path on locally cached mecka episodes, so the dataloader is in the measurement. |

```bash
salloc -p loaner -A loaner --gres=gpu:h100:1 -c 28 --mem=200G
source ./wt-env.sh
python bench/bench_hpt.py --variants base,compile_hook --steps 25 --warmup 10
```

`bench_pi.py` needs a venv on the pinned transformers (see "Loose ends", below).

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
including two grad-norm passes (`--grad-norm current`). That models a wrapper
older than this branch: `ModelWrapper` has run one pass since the MAD detector
went log-only, so every absolute time below carries one extra ~3 ms sweep; the
ratios are unaffected.

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

* **`enable_grad_norm` was never passed to `ModelWrapper`**, so
  `model/pi0.5_base.yaml`'s `false` did nothing. It is wired now; Pi drops the
  `false` and keeps logging its grad norm, as openpi does, and clips at openpi's
  1.0 (`gradient_clip_val` in the model config). HPT pays one pass, a
  `clip_grad_norm_(inf)` sweep plus a host sync, about half the 6.5 ms/step the
  two-pass bench measured; Pi pays that plus Lightning's clipping sweep, as
  openpi does.
* **Which transformers you get depends on the venv.** `pyproject.toml` pins
  4.53.2 plus openpi's `transformers_replace` patch. The venv these numbers were
  taken in had 5.17.0, where that patch breaks Pi (`PaliGemmaForConditionalGeneration`
  has no `language_model`) and `tests/fixtures/train_harness.py`
  (`transformers.utils.hub._is_offline_mode` was removed in 5.x); the Pi
  benchmarks used a pinned, git-ignored `.venv-pi`, in which the whole unit suite
  passes (1127 passed, 8 skipped).
* **A stream prefetcher** is the obvious next input-side move: issue the next
  batch's H2D on a side stream during the current step, which is what would
  make `pin_memory` pay.
* **uint8 images** would cut the batch another 4x (169 MB -> 42 MB) over the
  float32 change, moving the `/255` and the transpose onto the GPU. Bigger
  blast radius -- the transform list and norm stats would need checking.
