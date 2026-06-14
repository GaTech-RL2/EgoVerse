#!/bin/bash
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
srun --jobid=3332267 --export=ALL bash scripts/smoke_one_step.sh \
  bc_rnn_pushshapes_paperexact_hnet_chunk8 \
  /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnHnetC8FH_nc3/bc_rnn_hnet_chunk8_fullhist_2026-06-10_00-57-36/checkpoints/epoch_epoch=399.ckpt \
  /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnHnetC8FH_nc3/bc_rnn_hnet_chunk8_fullhist_2026-06-10_00-57-36/norm_stats/norm_stats.json \
  bcrnnHnetC8FH_nc3_cont2xSMOKE 8 \
  model.robomimic_model.rnn_horizon=80 model.robomimic_model.core_net.max_window=80
