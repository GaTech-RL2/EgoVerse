# EgoVerse2 → EgoVerse-refactor integration

This repo is a copy of `EgoVerse-gmm` with EgoVerse2 features integrated into
gmm's package structure. See the approved plan for scope. OMITTED: JEPA, DAgger,
closed-loop H-Net training.

## Baseline test state (fresh copy of EgoVerse-gmm, before any integration)

`PYTHONPATH=. .venv/bin/python -m pytest tests/ -q`
→ **150 passed, 16 failed, 9 skipped** (~74s)

These 16 failures are **pre-existing in EgoVerse-gmm** (the copy is byte-identical),
NOT regressions. Several correspond to EV2 features whose *tests* already live in
gmm but whose *implementations* do not yet — these are EXPECTED to flip to passing
as integration proceeds:

- `tests/test_pi.py::test_visualize_preds_*` (4) — EV2 pi visualize_preds. Expect FIX after Stage 3/2.
- `tests/test_training_recipe.py::TestAlgoWiring::*` (7) — EV2 HNet algo wiring
  (`init_weights_range`, `lr_multipliers`, `use_parameter_groups`, `weight_decay`).
  Expect FIX after Stage 3 (HNet algo port).
- `tests/test_loader_equality.py::TestReferenceHashes::{test_padded,test_packed}_reference_hashes` (2) — data hash refs.
- `tests/test_packed_pipeline.py::TestInferNormFromPacked::test_full_pipeline_collects_per_feature_stats` (1).
- `tests/test_scan_interface.py::TestScanRouterMambaPath::{test_contract_and_causality_cuda,test_grad_flows_cuda}` (2) — require mamba CUDA kernels (fallback env).

**Regression rule:** any test passing in this baseline that later fails is a
regression I introduced and must fix. The 11 non-CUDA, non-data-hash failures
above should ideally become passes by end of integration.

### Known-flaky (pre-existing, NOT a regression)
- `tests/test_core_defaults_byte_identical.py::test_tx_matches_reference_fingerprint`
  — fails on the **pristine untouched EgoVerse-gmm** too (verified in isolation).
  Its `tx` forward checksum is ~1e-6 with a near-zero-tight tolerance, so float/
  threading nondeterminism flips it pass/fail run-to-run. Ignore for regression
  accounting (params ARE byte-identical; only the tiny forward sum drifts).

## Progress log
- Stage 0 ✅ copy + git baseline + sanity import OK.
- Stage 1 ✅ conflict edits (_cut_windows window_anchor, SimpleConv spatial/tokens,
  ResNetEncoder); compat conflicts verified no-op. Smoke OK.
- Stage 2 ✅ Qwen/T5 conditioning: text_encoders.py, HPT annotation params +
  _build_prompts + stem_process list branch + _robomimic injection;
  scheduler_utils.warmup_then_cosine; model+data Qwen configs (gmm paths).
  Smoke: QwenPooled/PerToken compute_latent -> (2,16,256), _build_prompts logic OK.
  Full pytest: 149 passed / 16 pre-existing fails (+ the flaky tx fingerprint) — no new regressions.
