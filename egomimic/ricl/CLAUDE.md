# RICL (retrieval-based in-context learning) — navigation

This directory is the entire RICL experiment on EgoVerse pi0.5: for each query
observation, retrieve k≈4 nearest human (aria) demos and splice their
`(image, state, action)` into the pi0.5 prefix. `PI0Pytorch` is unchanged; RICL is a
thin `PI` subclass + a data wrapper + DINOv2 retrieval.

## Read these first (don't re-derive — they're current)
- `README.md`              — architecture + cluster runbook (canonical).

## Files in this dir, by role
Layout: importable **core** modules live at the top level; runnable utilities are
under `scripts/` (package `egomimic.ricl.scripts`) and CPU unit tests under
`tests/`. Inter-script imports use the `egomimic.ricl.scripts.<name>` path; scripts
resolve `episode_lists/`, `pg_tokenizer/`, `outputs/` relative to the parent dir.

- Core logic (top level): `retrieval.py` (DINOv2 pool + cKDTree + per-query top-k
  cache), `conditioning.py` (prefix surgery), `data.py` (`RiclQueryDataset`,
  `ZarrBankFrameProvider`, `build_ricl_collate`), `metrics.py`.
- DROID verification core (top level; the original-RICL data as a known-good
  testbed — "thin shim, not a port"; see `DROID_VERIFICATION.md`):
  `droid_data.py` (corpus + query dataset + bank provider + leave-one-out cache
  builder over the pre-pooled `top_image_embeddings`; runnable `--build-cache`),
  `droid_eval.py` (`DroidRiclEval` retrieval vs a *true* zero-context floor,
  paired flow-loss; `DroidRiclModelWrapper`).
- Embedding -> index: embed a corpus with
  `egomimic/scripts/embedding_process/zarr_embedding.py` — supports SQL-registry
  filters (`--filter-lambda` + `--sync-root`) and writes to a writable mirror
  folder (`--output-root <out>` -> `<out>/<hash>/observations.embeddings.dinov2.front_1`)
  when the source stores are read-only. `scripts/build_embedding_index.py` then
  consolidates `<out>/<hash>` stores into one `<out>/_index/` (vectors.npy +
  refs.npz + manifest.json) for efficient kNN; `build_retrieval_index()` loads
  it back into a `retrieval.RetrievalIndex`. Embeddings are an OFFLINE-only
  artifact (used to build the cache; not read at train time). Note: embedding
  arrays are zero-padded to a chunk multiple — readers slice to `total_frames`.
- Segment-scoped retrieval: `scripts/segment_retrieval.py` builds a `RetrievalCache`
  whose pool is the frames *inside selected action segments* (vs `retrieval.py`'s
  episode-level pairs). Segments come from `episode_lists/action_segments.csv`
  (one row per `[start,end)` unit) selected by a `--filter` lambda over the row
  dict; bank == query == the in-segment pool with **leave-one-segment-out**
  (`--loso-scope segment|episode`, self-retrieval excluded for free). Reads vectors
  from a consolidated `--index` (recommended) or per-episode `--zarr-root`; output
  is the standard cache format (absolute frame index; out-of-segment frames are
  padding). Run repeatedly with different `--filter`/`--out` for multiple caches.
- `scripts/` — annotation pipeline (`build_action_table.py`,
  `build_episode_objects.py`, `extract_action_groups.py`,
  `extract_object_segments.py`, `validate_object_labels.py`,
  `compute_norm_stats.py`, `check_bank_norm.py`, `viz_retrieval.py`), smoke-test
  helpers (`build_ricl_smoke_cache.py`, `preview_aligned_pairs.py`,
  `ricl_smoke.sbatch`), and the DROID trainer (`train_droid_ricl.py`
  `--stage cpu|full`, reuses the real `ModelWrapper`+`Trainer`;
  `train_droid_ricl.sbatch`).
- `tests/` — CPU unit tests (run `pytest egomimic/ricl/tests/`):
  `conditioning_test.py`, `data_test.py`, `metrics_test.py`, `droid_data_test.py`.
- Data (LARGE — grep/inspect targeted, never Read whole): `episode_annotations.json`
  (~15 MB), `episode_lists/*.{csv,json,txt}` (e.g. `action_segments.csv` ~1.1 MB).
- Gitignored (on disk, never committed/Read): `outputs/` (regenerable viz, smoke cache,
  aligned-pair previews), `pg_tokenizer/` (vendored PaliGemma tokenizer for offline runs).

### Data folders live INSIDE the repo, under `egomimic/ricl/outputs/` (gitignored)
All RICL on-disk data artifacts go under `egomimic/ricl/outputs/` — **never** the home
dir or the project root (`/storage/project/r-dxu345-0/rco3/…`). This keeps a run's inputs
and outputs co-located with the code and uncommittable in one shot (the whole `outputs/`
tree is gitignored at `.gitignore:53`). Canonical layout (`<dataset>` e.g. `pickplace_eva`):
- `outputs/ricl_embeddings/<dataset>/<hash>/` — per-episode DINOv2 mirror
  (`observations.embeddings.dinov2.front_1`), written by `zarr_embedding.py --output-root`.
- `outputs/ricl_embeddings/<dataset>/_index/` — consolidated kNN index
  (`vectors.npy` + `refs.npz` + `manifest.json`) from `build_embedding_index.py`.
- `outputs/ricl_sync_cache/<dataset>/` — S3 sync scratch for `zarr_embedding.py --sync-root`.
- `outputs/ricl_caches/<name>/` — `RetrievalCache`s (episode-level or segment-LOSO).
- `outputs/<run>/…` — viz montages and other regenerable artifacts.

These are LARGE (raw patch tokens ≈ 0.77 MB/frame; the smoke set of 5 eva episodes ×
3000 frames = 13 GB). The per-episode mirror is offline-only (consumed solely by the
index build, never at train time), so it can be deleted once `_index/` exists if disk is
tight. Pass these paths explicitly on the CLI (the scripts default to no output path).

## RICL's integration points OUTSIDE this dir (the rest of its working set)
- `egomimic/algo/pi_ricl.py` — `PIRicl(PI)`, 3 overrides; parent `egomimic/algo/pi.py`.
- `egomimic/eval/pi_ricl_eval.py` — `PIRiclEval` (retrieval-vs-floor).
- `egomimic/pl_utils/pl_data_utils.py` — `RiclDataModuleWrapper`.
- Configs: `egomimic/hydra_configs/{data/cotrain_pi_ricl*.yaml, data/ricl_stats_*.yaml,
  data/eva_pi.yaml, model/pi0.5_ricl.yaml, model/pi0.5_base.yaml, evaluator/eval_pi_ricl.yaml}`.
- Pre-embedding: `egomimic/scripts/embedding_process/{zarr_embedding,dinov2_embedding}.py`.
- Pairing: `egomimic/scripts/{human_robot_pairs.json, human_robot_pairing.md,
  pair_episodes_by_language.py, inspect_episode_metadata.py}`.
- Builds on (don't modify casually): `egomimic/trainHydra.py`, `egomimic/rldb/**`
  (esp. `rldb/embodiment/{eva,human}.py`, `rldb/zarr/`), `egomimic/utils/action_utils.py`.
- Reference only (read when needed, never edit): `external/openpi/` (pi0.5 base),
  `external/ricl_openpi/` (pi0-FAST architecture reference).

Keep this file updated when the RICL working set changes.
