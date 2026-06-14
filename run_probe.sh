#!/bin/bash
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
DS=/coc/flash7/paphiwetsa3/datasets
python /coc/flash7/paphiwetsa3/projects/EgoVerse2/probe_ranges.py \
  $DS/new_circle_3_normalized/episode_T_circle_obs0_000000.zarr \
  $DS/new_circle_3/episode_T_circle_obs5_000000.zarr \
  $DS/new_circle_3/episode_T_circle_obs0_000000.zarr
