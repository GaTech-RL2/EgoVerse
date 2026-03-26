# Quick Start
Behavior cloning and flow-matching based policy training for EgoVerse
datasets.

All commands below assume you are inside the `egomimic/` subfolder.

``` bash
cd egomimic
```

Launch interactive training with default configs on an example episode of RL2 lab robot data.

``` bash
python trainHydra.py --config-name=train_zarr
```

This will:

1.  Filter episodes from SQL
2.  Download those episodes locally from cloudflare R2 (if not already present)
3.  Build the Zarr dataset
4.  Compute normalization statistics on the fly
5.  Start training using default configs

------------------------------------------------------------------------

# Configuration Overview

All Hydra configs live in:

    hydra_configs/

Main subfolders:

    hydra_configs/
    ├── train_zarr.yaml        # top-level training config
    ├── data/                  # dataset configs
    ├── model/                 # model instantiations
    ├── hydra/launcher/        # SLURM submitit configs

For most use cases, you only need to modify the data YAML.

------------------------------------------------------------------------

# Training Aria Behavior Cloning (Interactive)

## 1. Modify data config

Open:

    hydra_configs/data/aria.yaml

Modify:

-   `folder_path` → local dataset directory\
-   `batch_size`\
-   `num_workers`

Reference run: - `batch_size = 32`\
- `num_workers = 10`\
- Single L40S GPU\
- 150k steps

## 2. Launch training

``` bash
/python trainHydra.py   --config-name=train_zarr   data=aria model=hpt_bc_flow_aria
```

------------------------------------------------------------------------

# Training on SLURM (Cluster)

## 1. Configure SLURM launcher

Edit:

    hydra_configs/hydra/launcher/submitit.yaml

Match this to your `sbatch` configuration: - partition\
- GPUs\
- memory\
- timeout\
- nodes

## 2. Submit job

``` bash
python trainHydra.py   --config-name=train_zarr   data=aria model=hpt_bc_flow_aria   -m
```

The `-m` flag enables Hydra multirun mode and triggers the Submitit
SLURM launcher.

------------------------------------------------------------------------

# Aria + EVA Co-Training

## 1. Modify dataset config

Edit:

    hydra_configs/data/eva_human_cotrain.yaml

Modify: - `folder_path`\
- `batch_size`\
- `num_workers`

## 2. Launch

Interactive:

``` bash
python trainHydra.py   --config-name=train_zarr   data=eva_human_cotrain model=hpt_cotrain_flow_shared_head
```

SLURM:

``` bash
python trainHydra.py   --config-name=train_zarr   data=eva_human_cotrain model=hpt_cotrain_flow_shared_head   -m
```

------------------------------------------------------------------------

# Data YAML Walkthrough

Main training config:

    hydra_configs/train_zarr.yaml

## DataSchematic

`DataSchematic`:

-   Maps post-transform dataset keys → batch keys expected by the model\
-   Stores key shapes and normalization stats\
-   Computes normalization dynamically during training

------------------------------------------------------------------------

## Dataset Structure

Each file in:

    hydra_configs/data/

Contains:

``` yaml
train_datasets:
  eva_bimanual:
    _target_: egomimic.rldb.zarr.zarr_dataset_multi.MultiDataset._from_resolver
```

### MultiDataset

Virtually merges multiple dataset units into a single dataset.

### Resolver

``` yaml
resolver:
  _target_: egomimic.rldb.zarr.zarr_dataset_multi.S3EpisodeResolver
```

Responsible for:

-   Applying SQL filters\
-   Finding matching dataset units\
-   Instantiating Zarr datasets

### Key Map

``` yaml
key_map:
  _target_: egomimic.rldb.embodiment.eva.Eva.get_keymap
```

Maps:

    raw dataset keys → pre-transform key names

### Transform List

``` yaml
transform_list:
  _target_: egomimic.rldb.embodiment.eva.Eva.get_transform_list
```

Defines:

-   Frame transforms (e.g., base → camera frame)\
-   Key renaming\
-   Concatenation\
-   Action chunking\
-   Post-processing

### Filters

``` yaml
filters:
  episode_hash: "2025-12-26-18-07-46-296000"
```

Filters are applied against the S3 SQL table to construct the dataset.\
The list of available filters is visible directly in the SQL table.

### Mode

``` yaml
mode: total
```

Options: - `train`\
- `valid`\
- `percent`\
- `total`

Controls how dataset units are sampled.

------------------------------------------------------------------------

# Mental Model of the Data Pipeline

1.  SQL Filters\
2.  `S3EpisodeResolver` finds matching units\
3.  Units instantiated as Zarr datasets\
4.  `MultiDataset` merges them\
5.  Embodiment transforms applied\
6.  `DataSchematic` maps to model batch keys\
7.  Normalization computed\
8.  Training begins

------------------------------------------------------------------------

# Overriding Configs Inline

Example:

``` bash
python trainHydra.py   --config-name=train_zarr   data=aria   train.batch_size=64   train.num_workers=8
```


# Norm Stats
By default in `train_zarr.yaml` norm stats are computed over the whole dataset.  Decrease `norm_stat_fraction` in `train_zarr.yaml` when training on large datasets.

------------------------------------------------------------------------

# Tips

-   Always verify `folder_path` exists and has enough disk space.\
-   Large datasets will auto-download from S3 if not present locally.\
-   For debugging small runs, filter by a single `episode_hash`.  You can also set logger=debug trainer=debug to run fewer epochs.