#!/bin/bash
# Launch the 40-seed / 1800-step generalization-level sweep for the 3 table families.
# Usage: launch_genlvl2_40.sh [ARRAY]   (ARRAY defaults to 0-29; pass e.g. 0 for a smoke)
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
mkdir -p logs/genlvl40 logs/genlvl40_vids
ARRAY=${1:-0-29}
SB=${SB:-scripts/genlvl2_40seed.sh}
TX=bc_rnn_pushshapes_paperexact_tx_chunk8_fullhist
HN=bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist
C3000=/coc/flash7/paphiwetsa3/datasets/circle_3000
CGEN=/coc/flash7/paphiwetsa3/datasets/circle_3000_plus_cotrain240
L=logs   # pinned to the EXACT run dirs the original genlvl table used (not ls -dt|head -1)
submit() { # cell family rdir dataset model
  jid=$(sbatch --parsable ${SBFLAGS:-} --job-name=genlvl2_$1 --array=$ARRAY \
    --export=ALL,CELL=$1,FAMILY=$2,RDIR=$3,DATASET=$4,MODEL=$5 $SB)
  echo "$1  array=$ARRAY  rdir=$3  job=$jid"
}
CELLS=${CELLS:-all}
do_cell() { case " $CELLS " in *" all "*) return 0;; *" $1 "*) return 0;; *) return 1;; esac; }
do_cell hptC8_nogen  && submit hptC8_nogen  hpt    $L/hptFlowC3000CausalAct8/hpt_flow_circle_3000_causal_obs1_act8_2026-06-15_00-02-10/      $C3000 none
do_cell hptC8_gen    && submit hptC8_gen    hpt    $L/hptFlowC3000Act8PlusGen/hpt_flow_circle_3000_PLUSGEN_causal_obs1_act8_2026-06-17_01-58-10/ $CGEN  none
do_cell txC8_nogen   && submit txC8_nogen   packed $L/bcrnnTxC8FH_c3000v2/bc_rnn_tx_chunk8_fullhist_c3000_2026-06-15_01-57-13/                 $C3000 $TX
do_cell txC8_gen     && submit txC8_gen     packed $L/bcrnnTxC8FHPlusGen/bc_rnn_tx_chunk8_fullhist_PLUSGEN_2026-06-17_01-48-18/                $CGEN  $TX
do_cell hnetC8_nogen && submit hnetC8_nogen packed $L/bcrnnHnetC8FH_c3000v2/bc_rnn_hnet_chunk8_fullhist_c3000_2026-06-15_01-40-00/             $C3000 $HN
do_cell hnetC8_gen   && submit hnetC8_gen   packed $L/bcrnnHnetC8FH_cotrain/bc_rnn_hnet_chunk8_fullhist_cotrain_2026-06-16_00-15-22/           $CGEN  $HN
