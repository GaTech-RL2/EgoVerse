# Local EgoVerse recovery

- User constraint: work without SSH access to Skynet. Do not contact Skynet,
  submit cluster jobs, or run historical remote orchestration scripts.
- Read `recovery/README.md` and `recovery/adaptations.json` before edits.
- This repository is based on `GaTech-RL2/EgoVerse`, branch
  `rby1_aria_shared`, commit recorded in `recovery/import-manifest.json`.
- `campaign/` is the recovered training/model code. `recovery/snapshots/` is
  historical source/evidence, often containing absolute remote paths. Preserve it
  except for the user-requested English translations recorded in adaptations.
  Do not mistake an archived READY or RUNNING report for current readiness.
- Keep all documentation, comments and human-readable metadata in English.
- Preserve source provenance. Record further adaptations separately from exact
  imported bytes; do not infer remote parity from a local test.
- Keep raw data, weights, runtime environments, and machine-local path mappings
  outside Git. `.gitignore` excludes these from this recovery branch.
- CPU checks: `.venv/bin/python -m recovery.verify`. Optional old deployed
  checkpoint check: add `--checkpoint-package /absolute/path/to/package`.
- Training runners retain CUDA/BF16 and release/smoke gates. A Mac synthetic
  smoke is not full CUDA training, real-data QC, or hardware acceptance.
- Primary hardware remains RBY1 Model-M + XHand, torso4:10/right10:17/left17:24/head24:26.
