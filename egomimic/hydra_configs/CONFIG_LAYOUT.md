# Hydra configuration layout

Configs are organized by purpose and family:

- `model/<family>/`: model architecture/config recipes (`bf`, `hnet`, `hpt`, `dfot`, `bc`, `pi05`, and others).
- `data/<task>/`: dataset/data-module configs (`pusht`, `human`, `robot`, `cotrain`, or `misc`).
- `evaluator/<family>/`: evaluator configs; visualization configs remain under `evaluator/viz/`.
- `experiment/<task>/`: run-level overrides that compose model, data, evaluator, and trainer settings.

Existing active and preserved legacy YAMLs were both imported. If an active and preserved file had the same name but different contents, the active version keeps the original name and the preserved version is named with a `_legacy` suffix so neither is overwritten.

Hydra references in repository configs and launchers were updated to the new group paths.
