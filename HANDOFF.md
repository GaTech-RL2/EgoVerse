# Handoff — H-Net CUDA kernel install + viz polish (2026-05-15 ~23:30 EDT)

Hand this to another Claude terminal. The user is Elmo (elmoworld2005@gmail.com).
Project: `/coc/flash7/paphiwetsa3/projects/EgoVerse`. Branch: `hnet-variants`.
PR: https://app.graphite.com/github/pr/GaTech-RL2/EgoVerse/452

## Right now (in-flight)

1. **5-epoch kernel smoke** (`bx11nx0w7`, spot 3170865) — **DONE**, exit 0.
   Epoch 4/4 at 2:05/epoch. Same wall-clock as pre-kernel runs because
   `boundary_rate` is ~0% at init, so EMA's M is tiny and the new
   `mamba_chunk_scan_combined` path doesn't help yet.

2. **100-epoch H-Net training with kernels enabled** — launched.
   - Bash task id: `b3ogdv4nj`
   - SLURM job: `3170866` (sonny)
   - Started: 23:37 EDT, max_epochs=100, limit_train_batches=8, val every 25.
   - Expected ETA: ~3.5h based on the 2:05/epoch smoke (= matches the
     prior `adaln_recipe_lowlr_100ep_v3` baseline at 156.25 paired_mse).
   - Output will be under `logs/hnet_variants/kernels_on_100ep_*`.
   - The interesting comparison vs baseline will be wall-clock in the
     last 25 epochs once `boundary_rate` has grown — that's where the
     mamba scan should pull ahead.

3. **HPT 100ep v3** still running on crushinator (jobid 3170832, task
   id was started before this session; check `ps -u $USER` for ETA).
   Don't touch.

## What got accomplished in this session

### A. CUDA kernels installed (flash-attn / mamba_ssm / causal_conv1d)

The user asked whether mamba and flash-attn could be built locally
since EMA loop is the documented perf cliff. They CAN, and now ARE.

**Verified working:**
```
torch:           2.6.0+cu124
has_flash_attn:  True   (flash-attn 2.7.4.post1)
has_mamba:       True   (mamba_ssm 2.2.4)
has_mamba_scan:  True   (causal_conv1d 1.5.0.post8)
```

**Setup paths now in repo:**
- `scripts/install_cuda_kernels.sh` — one-shot installer. Runs on a
  compute node, drops micromamba into `.micromamba/`, installs the
  cu12.4 toolkit into `./cuda-12.4/` (~1.6 GB), then builds the three
  exts with the pinned versions + flags.
- `scripts/build_cuda_exts.sh` — just the build step (assumes toolkit
  already exists).
- `scripts/recover_torch_cu124.sh` — recovery if torch ever gets bumped.
- `CLAUDE.md` § "Installing the CUDA kernels" — full docs.

**Both `.micromamba/` and `cuda-12.4/` are gitignored.**

### B. Hard-won gotchas (DO NOT step on these)

1. `pip install <ext>` will silently **upgrade torch** to whatever the
   latest is (currently 2.12+cu130). `--no-build-isolation` alone does
   not stop this — that flag only affects *build*-time deps.
   **Always pass both `--no-deps --no-build-isolation`.**

2. **flash-attn 2.8+ / mamba_ssm 2.3+** require torch 2.7's c10 ABI
   (different `c10::Error` and `c10::Warning` signatures). They will
   build cleanly but throw `undefined symbol: _ZN3c105ErrorC2...` at
   import against torch 2.6. **Pin to:**
   - `flash-attn==2.7.4.post1`
   - `mamba_ssm==2.2.4`
   - `causal_conv1d==1.5.0.post8` (1.4.0 has a different csrc layout
     that breaks the ninja build entirely)

3. **`pip cache` is keyed by version, not torch ABI.** If you ever
   rebuild after a torch swap, run `pip cache remove "flash_attn*"`
   etc. first. The installer does this automatically.

4. **PyPI `nvidia-cuda-nvcc-cu12` wheel ships ptxas + nvvm but NOT the
   nvcc driver itself.** Use the conda channel `nvidia/label/cuda-12.4.1`
   via micromamba — that DOES ship `bin/nvcc`. The installer handles this.

5. **Cluster has no CUDA 12 system toolkit** — only `/usr/local/cuda-13.2`.
   Project-local install is the only path.

6. **Code regression risk**: an `Edit` failed earlier with `ENOSPC` and
   truncated `egomimic/models/hnet_nets/blocks.py` to 0 bytes. Recovered
   with `git restore`. If you see weird import errors after touching
   blocks.py, check its size.

### C. fp32 fallback in MultiHeadAttention._forward_packed

`flash_attn_varlen_func` only accepts fp16 / bf16. Added a dtype guard
in `egomimic/models/hnet_nets/blocks.py` so fp32 forwards (smokes /
no-autocast inference) gracefully fall back to SDPA. The actual
training runs under bf16 autocast (`trainer.precision=bf16-mixed`)
so the fast path is exercised there.

### D. Viz polish (committed earlier in session)

- **`eval_standard.yaml`**: switched `pad_h: min` → `pad_h: max` so the
  composite no longer crops PCA / HNet video panels to the boundary
  strip's height. Composite is now (3269, 384, 818, 3) instead of (..,
  256, ..).
- **`BoundaryStripEval`**: split into TWO side-by-side strips per chunker:
  - Left = continuous greyscale `P(boundary)` (gray_r).
  - Right = discrete red/white `boundary_mask` (committed dividers).
  - 2-px black divider between (so total stays even for x264).
  - Defensive even-pad in `EvalListSideBySide` too.
- **`PCATokenEval`**: hooks innermost `ComputeStage.main_network` to
  capture **chunker-output** tokens (one per chunk, T_chunked × d_inner)
  rather than per-frame action_out tokens. Now PCA scatter only steps
  when a boundary fires — matches user's intuition. Falls back to
  action_out for FlatFusedPolicy.

### E. Variant comparison (pulled metrics ranked by train_loss)

| Variant | Train loss | Notes |
|---|---:|---|
| fused_100ep | **0.0020** | best — FlatFusedPolicy, no chunker |
| fused_lowlr_v3 | 0.0021 | matches variant_ablation memory |
| full_run_150ep (baseline AdaLN, 150ep) | 0.0300 | |
| big_100ep | 0.0318 | bigger AdaLN |
| crossattn_100ep | 0.0330 | cross-attn cond |
| **adaln_recipe_lowlr_v3 (the new run)** | 0.0342 | +recipe+lowLR didn't help |
| delta_100ep | 0.1850 | delta actions |

Conclusion conveyed to user: the recipe + lowLR tweaks did NOT beat
plain AdaLN, and AdaLN's ceiling looks like `train_loss ≈ 0.03`
regardless of tricks. The architectural win (fused-tokens) is ~14-16×
better than any AdaLN variant. Matches the `hnet_variant_ablation`
auto-memory.

Val MSE numbers across variants aren't apples-to-apples — older
variants logged AR-rollout MSE (exposure-bias inflated ~37×), newer
v3 variants logged teacher-forced MSE. The CSV `Valid/emb15_actions_paired_mse`
column reflects whatever regime was active for that run.

## SLURM allocations (as of right now)

| Job | Node | Use |
|---|---|---|
| 3170865 | spot | **5-epoch H-Net kernel smoke** (the in-flight thing) |
| 3170866 | sonny | idle — use for the next training launch |
| 3170832 | crushinator | HPT 100ep v3 (started before this session) |
| 3169117 | clippy | finished adaln_recipe_lowlr v3; idle |
| 3161530 | brainiac | idle reserve |

## Repo state

- Branch: `hnet-variants`
- All work in this session pushed via `gt submit --stack`.
- Latest commit: `24e6eed Add one-shot installer for the CUDA kernels`.
- PR: #452 on graphite.

## Disk

- `/coc/flash7/paphiwetsa3` is at 25 G free after the build (started
  at 84 G). The build chain pulled extra cu13 packages mid-process
  that were cleaned, plus pip wheel cache hit ~4 GB at peak (purged).
  Don't fill it again — keep an eye on `df -h /coc/flash7/paphiwetsa3`.

## Auto-memory entries to know about

- `slurm_workflow.md` — always srun --jobid=N to compute node; never run on sky1.
- `hnet_ar_rollout_drift.md` — AR rollout MSE is ~37× train MSE due to
  exposure bias. Diagnosed, not a bug.
- `hnet_variant_ablation.md` — fused-tokens beats AdaLN+chunker 15-39×.

## User behaviour notes (from feedback memories)

- Wants compute-node-only execution. Confirmed in CLAUDE.md too.
- Wants the standard eval = composite (HNetVideo + PCA + boundary-strip)
  + sim rollout, side-by-side. HPT skips composite.
- Wants `EvalList` (base class) to NOT concat — only `EvalListSideBySide`
  (subclass) does that.
- Wants venv-activate stripped of slow graphite auth checks. Done in
  `~/.bashrc` (interactive-only gating).
- Likes short answers / direct results, not running commentary.

## What's NOT done yet

- After the smoke + full-run finish: confirm the kernels measurably
  speed up training. Can compare wall-clock per epoch against the
  prior `adaln_recipe_lowlr_100ep_v3` run (which was 2:04 per epoch
  at 8 batches without kernels). The win scales with `boundary_rate`,
  which starts at ~0% and grows during training, so the speedup will
  only be visible in later epochs.
- Composite + sim eval haven't been run on the new variant checkpoints
  yet. Per user's "standard eval for all H-Net models" rule, this is
  TODO for each completed variant.
- Task #19 "Plot metrics for all variants" still pending.
