#!/bin/bash
set -u
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
git checkout 58e598d4 -- Tsimulation/pushshapes/ && echo ENV_CHECKED_OUT
JID=$(salloc -A rl2-lab -p rl2-lab --time=1:00:00 --gres=gpu:a40:1 --cpus-per-task=4 --mem=24G --no-shell 2>&1 | grep -oP 'allocation \K[0-9]+')
echo alloc=$JID
srun --jobid=$JID bash -lc "export PYTHONPATH=.; export SDL_VIDEODRIVER=dummy; .venv/bin/python /tmp/verify_env.py" > /tmp/o_verifyenv.txt 2>&1
echo "rc=$?"; tail -15 /tmp/o_verifyenv.txt
scancel $JID 2>/dev/null
