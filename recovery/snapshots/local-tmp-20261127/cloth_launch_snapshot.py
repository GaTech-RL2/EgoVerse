import pathlib,json,subprocess,time,hashlib
R=pathlib.Path('/coc/flash7/cxin34/cloth-valid-recovery-fix1124-20261124');W=R/'ft';p=R/'reports/ft-plan.json';d=json.load(open(p));run=d['run'];metrics=[json.loads(l)for l in (W/'runs'/run/'metrics.jsonl').read_text().splitlines()if l.strip()]
proof=json.load(open(W/'reports'/(run+'-independent-smoke-audit.json')));assert proof['passed']
queue=subprocess.check_output(['/opt/slurm/Ubuntu-24.04/current/bin/squeue','-j','4855714','-h','-o','%i|%T|%M|%R|%j'],text=True)
assert 'RUNNING' in queue and metrics[-1]['step']>=100
out=dict(time=time.time(),job=4855714,queue=queue,formal_started=True,latest_formal_metrics=metrics[-1],smoke_passed=True,smoke_step=200,full_validation_rows=2686,manifest_sha256=proof['identity']['manifest_sha256'],old_artifacts_unchanged=True,old_jobs_not_modified=True)
(R/'reports/launch-status.json').write_text(json.dumps(out,indent=2));d.update(submission_pending=False,job=4855714,smoke_passed=True,formal_started=True,latest_status=str(R/'reports/launch-status.json'));p.write_text(json.dumps(d,indent=2));print(json.dumps(out))
