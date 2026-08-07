#!/bin/bash
# Kill the slow unbounded cold-start eval, launch the bounded fast one.
pkill -f eval_A_coldstart.sh 2>/dev/null
/opt/slurm/Ubuntu-20.04/current/bin/scancel 3283426.5 2>/dev/null
pkill -f "description=cs_chunk1" 2>/dev/null
sleep 3
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
if pgrep -f eval_A_cs_fast.sh >/dev/null 2>&1; then
  echo FAST_ALREADY_RUNNING
else
  nohup bash eval_A_cs_fast.sh 3283426 > /tmp/o_Acsf_wrap.txt 2>&1 &
  echo FAST_LAUNCHED_PID=$!
fi
