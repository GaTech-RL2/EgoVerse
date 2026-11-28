import json,os,subprocess
from pathlib import Path
R=Path('/coc/flash7/cxin34/cloth-new2-scratch-20261121');S='/opt/slurm/Ubuntu-24.04/current/bin/sbatch'
intent=R/'reports/batch-submit.intent';fd=os.open(intent,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
def submit(args):return subprocess.check_output([S,'--parsable',*args],text=True).strip().split(';')[0]
audit=submit(['--job-name=cxin_new2_pilot_audit','--partition=interactive-cpu','--cpus-per-task=2','--mem=16G','--time=00:30:00','--dependency=afterok:4826296','--output='+str(R/'logs/pilot_audit_%j.log'),str(R/'scripts/pilot_audit.sbatch')]);(R/'reports/jobs.json').write_text(json.dumps({'pilot':'4826296','pilot_audit':audit},indent=2))
batch=submit(['--job-name=cxin_new2_features','--partition=overcap','--account=overcap','--gres=gpu:a40:1','--cpus-per-task=4','--mem=32G','--time=01:00:00','--array=1-59%4','--dependency=afterok:'+audit,'--output='+str(R/'logs/features_%A_%a.log'),str(R/'scripts/worker.sbatch')])
jobs={'pilot':'4826296','pilot_audit':audit,'batch':batch,'concurrency':4};(R/'reports/jobs.json').write_text(json.dumps(jobs,indent=2));print(json.dumps(jobs))
