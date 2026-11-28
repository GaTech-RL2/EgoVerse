import json,subprocess,time
from pathlib import Path
R=Path('/coc/flash7/cxin34/cloth-valid-recovery-fix1124-20261124');p=R/'reports/visual-submit.json';intent=R/'reports/visual-submit.intent';assert not p.exists()and not intent.exists()
cmd=['/opt/slurm/Ubuntu-24.04/current/bin/sbatch','--parsable','--partition=interactive-cpu','--account=rl2-lab','--qos=debug','--exclude=hal,bmo','--cpus-per-task=1','--mem=8G','--time=00:30:00','--dependency=afterany:4855089','--job-name=cloth-recovery-visual','--output='+str(R/'logs/visual-%j.log'),str(R/'scripts/visual.sbatch')]
intent.write_text(json.dumps(dict(command=cmd,time=time.time())));z=subprocess.run(cmd,text=True,capture_output=True);p.write_text(json.dumps(dict(command=cmd,returncode=z.returncode,stdout=z.stdout,stderr=z.stderr),indent=2));assert z.returncode==0,z.stderr;print(z.stdout)
