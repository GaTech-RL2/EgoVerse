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

## Progress log
- Stage 0 ✅ copy + git baseline + sanity import OK.
