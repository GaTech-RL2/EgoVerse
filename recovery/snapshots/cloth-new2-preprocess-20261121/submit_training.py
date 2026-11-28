import os,json,subprocess
from pathlib import Path
R=Path('/coc/flash7/cxin34/cloth-new2-scratch-20261121');S='/opt/slurm/Ubuntu-24.04/current/bin/sbatch'
ready=json.loads((R/'releases/cloth-new2-scratch/READY.json').read_text());assert ready['passed'] and ready['visual_review_passed']
cfg=json.loads((R/'configs/cloth-new2-B-msc-scratch-s42.json').read_text());assert cfg['stage']=='scratch' and cfg['steps']==180000 and 'parent' not in cfg
intent=R/'reports/train-submit.intent';fd=os.open(intent,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
jobs=json.loads((R/'reports/jobs.json').read_text())
def submit(args):return subprocess.check_output([S,'--parsable',*args],text=True).strip().split(';')[0]
def save(): (R/'reports/jobs.json').write_text(json.dumps(jobs,indent=2))
gpu=['--partition=overcap','--account=overcap','--gres=gpu:a40:1','--cpus-per-task=4','--mem=40G','--time=04:00:00']
smoke=submit([*gpu,'--job-name=new2-MSC-scratch-smoke','--output='+str(R/'logs/smoke_%j.log'),str(R/'scripts/smoke.sbatch')]);jobs['smoke']=smoke;save()
gate=submit(['--partition=interactive-cpu','--cpus-per-task=2','--mem=24G','--time=00:30:00','--job-name=new2-MSC-smoke-audit','--dependency=afterok:'+smoke,'--output='+str(R/'logs/smoke_gate_%j.log'),str(R/'scripts/gate.sbatch')]);jobs['smoke_gate']=gate;save()
formal=submit([*gpu,'--job-name=new2-MSC-scratch180k','--dependency=afterok:'+gate,'--signal=B:USR1@90','--export=ALL,CAMPAIGN_ROOT='+str(R)+',RUN_ID=cloth-new2-B-msc-scratch-s42,SMOKE=0','--output='+str(R/'logs/formal_%j.log'),str(R/'scripts/train.sbatch')]);jobs['formal']=formal;save();print(json.dumps(jobs,indent=2))
