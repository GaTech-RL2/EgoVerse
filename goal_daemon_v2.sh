#!/bin/bash
# v2: keep-alive for 3 tx (baseline7) + 4 hnet_ed (baseline8 encdec arch) cells until ep2000.
# Detects a dead cell (metrics frozen > STALE AND no running/pending slurm job) and
# resubmits it (launcher resumes from last.ckpt). Safeguards: per-cell cooldown,
# resubmit cap, and controller-down skip (never blind-resubmit during a slurm outage).
# Logs a status line every cycle. Exits once ALL cells have reached ep TARGET.
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-gmm
CELLS="tx_bc_big tx_bc_small tx_cotrain hnet_ed2_cotrain hnet_b1_cotrain hnet_b3_cotrain hnet_b4_cotrain"
TARGET=2000
COOLDOWN=2400   # >=40 min between resubmits of the same cell
MAXSUB=8        # give up a cell after this many resubmits (needs a human)
STALE=1200      # metrics frozen >20 min => candidate dead
declare -A LAST_SUB COUNT_SUB

log(){ echo "$(date +%m-%d_%H:%M:%S) $*"; }

submit_cell(){
  # Recoveries go to overcap (broad scavenger capacity) + --requeue so a preemption
  # auto-resumes via the launcher's ckpt-resume hook. hoffman-lab is saturated by our
  # own 6 long-running cells, so recovery jobs pend there indefinitely.
  sbatch --parsable \
    --partition=overcap --account=overcap --qos=scavenger_qos --time=2-00:00:00 \
    --nodes=1 --ntasks-per-node=3 --cpus-per-task=12 --mem=378G --gres=gpu:a40:3 \
    --requeue --exclude=baymax,cyborg,ig-88,hk47 \
    --job-name=indomain_c4_200m_v5_$1 \
    --output=logs/slurm/v5_$1_%j.out \
    $( [ "${1#hnet_}" != "$1" ] && echo hnet_ed_launch.sh || echo indomain_c4_200m_v5_launch.sh ) "$1"
}

log "GOAL_DAEMON start target=ep$TARGET pid=$$"
for iter in $(seq 1 100000); do
  now=$(date +%s)
  free=$(df --output=avail -BG /coc/flash7 2>/dev/null | tail -1 | tr -dc '0-9')
  alldone=1; line=""
  for cell in $CELLS; do
    csv=$(find logs/indomain_c4_200m_v5/${cell}_2026* -name metrics.csv -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)
    if [ -n "$csv" ]; then
      mt=$(stat -c %Y "$csv" 2>/dev/null); age=$((now-mt))
      ep=$(awk -F, 'NR==1{for(i=1;i<=NF;i++)if($i=="epoch")c=i} c&&NR>1&&$c~/^[0-9]/{e=$c} END{print e+0}' "$csv")
    else ep=0; age=999999; fi
    st="ok"
    if [ "$ep" -ge "$TARGET" ]; then st="DONE"; else
      alldone=0
      if [ "$age" -gt "$STALE" ]; then
        sq=$(squeue -h -n indomain_c4_200m_v5_${cell} -o "%T" 2>&1)
        if echo "$sq" | grep -qiE "error|controller|invalid"; then st="sqdown_skip"
        elif echo "$sq" | grep -qE "RUNNING|PENDING|CONFIGURING|COMPLETING"; then st="reviving"
        else
          last=${LAST_SUB[$cell]:-0}; cnt=${COUNT_SUB[$cell]:-0}
          if [ "$cnt" -ge "$MAXSUB" ]; then st="GIVEUP@ep$ep"
          elif [ $((now-last)) -gt "$COOLDOWN" ]; then
            jid=$(submit_cell "$cell")
            LAST_SUB[$cell]=$now; COUNT_SUB[$cell]=$((cnt+1))
            st="RESUB#$((cnt+1))->$jid"
            log "RECOVER $cell dead ep$ep age${age}s -> $jid (attempt $((cnt+1))/$MAXSUB)"
          else st="dead_cooldown"; fi
        fi
      fi
    fi
    line="$line ${cell}:${ep}(${st})"
  done
  [ -n "$free" ] && [ "$free" -lt 60 ] && log "DISK_CRITICAL ${free}G — ENOSPC risk, consider scratch migration"
  log "STATUS free=${free}G |$line"
  [ "$alldone" -eq 1 ] && { log "ALL_REACHED_ep$TARGET — exiting"; break; }
  sleep 600
done
