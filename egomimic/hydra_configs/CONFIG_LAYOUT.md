# Hydra configuration layout

Configs are organized by purpose, task, and model family:

- `model/<family>/`: model architecture recipes (`bf`, `hnet`, `hpt`, `dfot`, `bc`, `pi05`, and others).
- `data/<task>/`: dataset/data-module configs (`pusht`, `human`, `robot`, `cotrain`, or `misc`).
- `evaluator/<family>/`: evaluator configs; visualization configs remain under `evaluator/viz/`.
- `experiment/<task>/`: run-level overrides that compose model, data, evaluator, logger, and trainer settings.
- `callbacks/<purpose>/`: checkpoint, chunker, dropout, scheduler, and base callback configs.
- `logger/<purpose>/`: CSV, W&B, composite, and debug logger configs.
- `trainer/<purpose>/`: base, distributed, and debug trainer configs.
- `paths/runtime/`: runtime path configs.
- `data_schematic/<family>/`: data schematic definitions.
- `hydra/launcher/`: cluster launcher configs.

Existing active and preserved YAMLs were both imported. If an active and preserved file had the same name but different contents, the active version keeps the original name and the preserved version is named with a `_legacy` suffix so neither is overwritten. These are collision-safe copies, not an archive directory.

Hydra references in repository configs and launchers were updated to the new group paths.
