# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

EgoVerse is the data-processing, training, and evaluation codebase for learning robot
policies from large-scale egocentric (Aria glasses) and robot (EVA) datasets.

## Environment & Commands

Always activate the venv before running any project commands (`python`, `pytest`, `pip`):

```bash
source /storage/project/r-dxu345-0/rco3/EgoVerse/emimic/bin/activate
```

### SLURM
Training and eval must run on a GPU. Request one before running/testing:
- sky1/sky2: `salloc -p rl2-lab -A rl2-lab --gres=gpu:a40:1 -c 12 --mem=30G`

For the Hydra submitit launcher (`-m` multirun) to work, `submitit_launcher.py` in the
installed `hydra_submitit_launcher` plugin must be patched — see README "Submitit modification".

## Running Training & Eval

All training goes through `egomimic/trainHydra.py` (PyTorch Lightning + Hydra, DDP-capable).

```bash
# Interactive run with default configs
python egomimic/trainHydra.py --config-name=train_zarr_cartesian

# Pick a dataset + model, multirun to launch on SLURM via submitit
python egomimic/trainHydra.py -m --config-name=train_zarr_cartesian_pi \
    data=mecka_pi model=pi0.5_bc_mecka launch_params.nodes=1 launch_params.gpus_per_node=4

# Debug (tiny run): trainer=debug logger=debug norm_stats.sample_frac=0.001
```

Eval reuses a finished run's saved `config` and checkpoint by setting `++mode=eval` and
`+evaluator=...`; see `training.md` for the exact invocation. `training.md` is the
authoritative guide for training/data-config usage — read it before changing configs.

## Architecture

### Hydra config composition (`egomimic/hydra_configs/`)
Top-level configs (`train_zarr_cartesian*.yaml`) compose groups: `data/`, `model/`,
`trainer/`, `logger/`, `callbacks/`, `evaluator/`, `data_schematic/`, and
`hydra/launcher/` (SLURM submitit). Override groups inline (`data=aria model=...`) or
individual values (`train.batch_size=64`). For most work only the `data/` YAML changes.

### Data pipeline — `rldb` (`egomimic/rldb/`)
`rldb` is the data-loading layer. The pipeline, end to end:
1. **SQL filters** select episodes from a Postgres episode table (all dataset metadata).
2. `S3EpisodeResolver` (`zarr/zarr_dataset_multi.py`) finds matching dataset units and
   auto-downloads their Zarr from Cloudflare R2 if absent locally.
3. Units are instantiated as Zarr datasets and virtually merged by `MultiDataset`.
4. **Embodiment transforms** (`rldb/embodiment/{eva,human}.py`) apply frame transforms,
   key renaming, concatenation, and action chunking. Each embodiment exposes
   `get_keymap` (raw keys → pre-transform names) and `get_transform_list`.
5. `DataSchematic` (`zarr/utils.py`) maps post-transform keys → the batch keys a model
   expects, and holds shapes + normalization stats.
6. Normalization stats are computed on the fly (controlled by `norm_stats` in the top
   config; lower `sample_frac` for large datasets, or supply `precomputed_norm_path`).

### Training / eval glue
- `pl_utils/pl_model.py` — `ModelWrapper`, the LightningModule wrapping an `Algo`.
- `pl_utils/pl_data_utils.py` — LightningDataModule.
- `eval/eval.py` + `eval/eval_{act,hpt,pi,video}.py` — `Eval` classes own all metric
  computation and visualization; models only return predictions from `forward_eval`.
