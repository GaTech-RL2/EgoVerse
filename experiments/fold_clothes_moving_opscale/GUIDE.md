# Folding clothes moving-budget operator scaling

The 36 configurations cross 24/32/48/64/72/96 hours with 4/8/16/32/64/128 operators.
Read the generated local README.md for the exact private operator order, all
validation episode IDs, actual hours, and launch instructions. It is gitignored
because these identifiers must stay private; a copy is also written to Downloads.

`reference_recipe.json` records the verified W&B recipe and renderer provenance.
`build_splits.py` merges original/replacement metadata, checks the supplied
blacklists and authoritative operator CSV, chooses one representative per known
location group, reserves all validation episodes, and builds nested budgets.
Private data is written only outside the repository, beneath
`EGOVERSE_MECKA_PRIVATE_DIR`. Unknown locations are not treated as duplicates.

`generate_configs.py` generates the 36 YAMLs and shared data/model configuration.
`validate_setup.py` validates every split and real Hydra composition against the
reference. It keeps the production gate blocked until actual GPU evaluation,
W&B video logging and full-state checkpoint/resume checks pass.
`check_readiness.py` checks the selected converted episodes independently.
`launch_pace.py` submits one GPU per run with global batch 64 through the existing
training entrypoint. `monitor.py` maintains status and durable lifecycle evidence.

Validation uses every episode in three disjoint cohorts: 40 fixed matched,
3 × N perlevel, and 400 fixed unseen. Validation is capped at 800 batches per cohort, batch size 32, as explicitly
requested by the user (the source reference used 500). Metrics use the existing even-stride frame
selection; the callback explicitly verifies full episode coverage. Videos use
the recovered reference open-loop renderer (13 or14 deterministic episode videos
per visualization event). Required reference extensions are isolated under
`egomimic/eval/opscale_reference_*`; archived originals and hashes are in
`reference_source/`. No run is declared complete from scheduler exit alone.

The source contains both legacy and corrected left-wrist conventions. Data
readiness builds a canonical local view without changing R2 or the raw download.
The existing 180-degree local-Z correction is applied exactly once to legacy
episodes; all model inputs then use the reference transform configuration.
Converted frame counts are checked against the conversion index, since video
and tracking streams can have different lengths. Source durations remain the
budget basis, and converted frame-clock hours are reported separately.
