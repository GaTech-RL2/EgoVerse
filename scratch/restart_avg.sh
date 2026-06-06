#!/bin/bash
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
pkill -f eval_DEC_avg.sh 2>/dev/null
/opt/slurm/Ubuntu-20.04/current/bin/scancel 3283665.7 2>/dev/null
sleep 3
nohup bash eval_DEC_avg.sh 3283665 > /tmp/o_DECavg_wrap.txt 2>&1 &
echo RESTART_AVG_DONE_PID=$!
