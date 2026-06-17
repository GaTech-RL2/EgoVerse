# Norm Stats

During a normal training run, norm stats are computed on the fly over
`norm_stats.sample_frac` of the data and cached to the run's `norm_stats/`
dir. Lower `norm_stats.sample_frac` (default `0.2`, set in
`train_zarr_cartesian.yaml`) when training on large datasets.

## Computing norm stats standalone (no training)

`egomimic/scripts/compute_norm_stats.py` computes and caches norm stats
**without** instantiating the model or trainer. It accepts the same Hydra
composition as `trainHydra.py` (`--config-name`, `data=`, `model=`, …), so
existing run recipes can be reused. It writes
`<save_cache_dir>/norm_stats/norm_stats.json`; point a follow-up training run
at it via `norm_stats.precomputed_norm_path=<that dir>` to skip recompute.

Key options:

-   `norm_stats.sample_frac` — fraction of data to sample (default `0.2`).
-   `norm_stats.num_workers` — dataloader workers; set to roughly match the
    allocated CPUs.
-   `norm_stats.save_cache_dir` — output dir (defaults to the Hydra run dir).
-   The script **always recomputes** — it ignores any
    `norm_stats.precomputed_norm_path` from the config.

### Interactive

``` bash
python egomimic/scripts/compute_norm_stats.py \
    --config-name=train_zarr_cartesian_pi \
    data=<your_data_config> \
    norm_stats.sample_frac=0.4
```

### SLURM (CPU-only, via submitit)

Use the CPU launcher `hydra/launcher/submitit_cpu_pace.yaml` and add `-m` to
trigger submitit. It defaults to `cpus_per_task: 12`; set
`norm_stats.num_workers` to match. Override `hydra.launcher.cpus_per_task` (and
`norm_stats.num_workers`) for a different CPU count. Example — 0.4 sample_frac
for `mecka_pi_fold_clothes_freeform` on the default 12 CPUs:

``` bash
python egomimic/scripts/compute_norm_stats.py -m \
    --config-name=train_zarr_cartesian_pi \
    data=mecka_pi_fold_clothes_freeform \
    name=mecka_fold_clothes_freeform \
    description=norm_stats_frac0.4 \
    norm_stats.sample_frac=0.4 \
    norm_stats.num_workers=12 \
    hydra/launcher=submitit_cpu_pace
```

`name`/`description` set the output dir
(`logs/<name>/<description>_<timestamp>/`); the stats land under
`<that dir>/0/norm_stats/norm_stats.json` (the `0` is the submitit multirun
subdir), with submitit logs in `<that dir>/.submitit/<jobid>/`.

Tip: validate the recipe cheaply on the login node before queuing with a
config-only dry run (no dataset load, no submission):

``` bash
python egomimic/scripts/compute_norm_stats.py \
    --config-name=train_zarr_cartesian_pi data=<your_data_config> \
    norm_stats.sample_frac=0.4 hydra/launcher=submitit_cpu_pace --cfg job
```
