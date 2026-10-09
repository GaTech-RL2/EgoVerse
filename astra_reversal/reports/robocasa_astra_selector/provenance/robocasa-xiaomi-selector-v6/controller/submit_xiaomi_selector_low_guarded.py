"""Submit a prepared owned GPU job once and reconcile its allocation ledger."""
import fcntl
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ops=Path(__file__).resolve().parent
directory=Path(sys.argv[1]).resolve()
if directory.parent != ops or not directory.name.startswith('launch-'):
    raise SystemExit('Launch directory must belong to this study')
plan=json.loads((directory/'launch_plan.json').read_text())
identity=plan['id']
assert plan['priority']=='LOW' and plan['automatic_rescheduling'] is False
if not identity.startswith(('robocasa-','bench-')) or plan['gpu_count'] != 1:
    raise SystemExit('This guard admits only the declared one-GPU study jobs')
if plan['pool']!='groot-l40s-01' or plan['startup_reserved_gpu_hours']!=1.51:
    raise SystemExit('Unexpected compute reservation')
for item in plan['files']:
    p=(directory/item['file']).resolve()
    if not p.is_relative_to(directory) or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
        raise SystemExit('Immutable launch file changed')
spec=importlib.util.spec_from_file_location('accounting',ops.parent/'reasoning-learning-20261003/budget_accounting.py')
accounting=importlib.util.module_from_spec(spec)
spec.loader.exec_module(accounting)

def save(ledger):
    p=ops/'budget.json.tmp'
    p.write_text(json.dumps(ledger,indent=2)+'\n')
    p.replace(ops/'budget.json')

with (ops/'budget.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    ledger=json.loads((ops/'budget.json').read_text())
    if any(r['id']==identity for r in ledger['entries']):
        raise SystemExit('Reservation already exists; inspect it before any retry')
    active=[r for r in ledger['entries'] if 'actual_gpu_hours' not in r]
    if sum(r['gpu_count'] for r in active)+1>ledger['maximum_concurrent_gpus']:
        raise SystemExit('Concurrency ceiling reached')
    charged=ledger['previous_study_gpu_hours']+sum(accounting.charge(r) for r in ledger['entries'])
    reservation=plan['execution_gpu_hours']+plan['startup_reserved_gpu_hours']
    if charged+reservation>ledger['authorized_total_gpu_hours']-0.1:
        raise SystemExit('Insufficient authorized budget')
    row={'id':identity,'gpu_count':1,'allocation_gpu_hours':plan['execution_gpu_hours'],'startup_allowance_gpu_hours':1.5,'startup_cancel_after_seconds':1200,'budget_reserved_gpu_hours':reservation,'status':'submission_unknown','source_revision':plan['source_revision'],'scope':plan['scope']}
    row.update(priority='LOW',automatic_rescheduling=False)
    ledger['entries'].append(row)
    save(ledger)
    p=subprocess.run(['/usr/local/bin/osmo','workflow','submit',str(directory/'workflow.yaml'),'--pool',plan['pool'],'--priority','LOW','--format-type','json'],capture_output=True,text=True,timeout=60)
    (directory/'submission.json').write_text(p.stdout)
    (directory/'submission.stderr.log').write_text(p.stderr)
    if p.returncode:
        raise SystemExit('Submission uncertain; reservation retained')
    workflow=json.loads(p.stdout)['name']
    if not workflow.startswith('astra-complex-20261006-'+identity+'-'):
        raise SystemExit('Unexpected workflow identity; reservation retained')
    row['workflow']=workflow
    save(ledger)
    print(json.dumps({'workflow':workflow,'charged_plus_reserved_gpu_hours':charged+reservation,'authorized_gpu_hours':ledger['authorized_total_gpu_hours']}),flush=True)

last_status=None
errors=0
unavailable_since=None
cancel_requested=False
monitor_deadline=time.monotonic()+plan['execution_gpu_hours']*3600+7200
while time.monotonic()<monitor_deadline:
    time.sleep(20)
    try:
        p=subprocess.run(['/usr/local/bin/osmo','workflow','query',workflow,'--format-type','json'],capture_output=True,text=True,timeout=35)
        if p.returncode:
            raise RuntimeError('query_failed')
        state=json.loads(p.stdout)
    except (subprocess.TimeoutExpired,RuntimeError,ValueError):
        errors+=1
        unavailable_since=unavailable_since or time.monotonic()
        print(json.dumps({'status_query_failures':errors,'reservation_retained':True}),flush=True)
        if time.monotonic()-unavailable_since>=1200 and not cancel_requested:
            # Bound unobserved allocation; only this exact owned workflow.
            p=subprocess.run(['/usr/local/bin/osmo','workflow','cancel',workflow],capture_output=True,text=True,timeout=45)
            (directory/'observability-cancel.log').write_text(p.stdout+p.stderr)
            cancel_requested=p.returncode==0
        if time.monotonic()-unavailable_since>=1380:
            raise SystemExit('Monitor transport unavailable; reservation retained and manual reconciliation required')
        continue
    errors=0
    unavailable_since=None
    (directory/'latest-status.json').write_text(json.dumps(state,indent=2)+'\n')
    with (ops/'budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=json.loads((ops/'budget.json').read_text())
        row=next(r for r in ledger['entries'] if r['id']==identity)
        accounting_state=state
        if state.get('end_time') and state.get('duration') is None and not state.get('start_time'):
            accounting_state={**state,'duration':0}
        accounting.update_entry(row,accounting_state)
        save(ledger)
    if state['status']!=last_status:
        print(json.dumps({'workflow':workflow,'status':state['status']}),flush=True)
        last_status=state['status']
    if state.get('end_time'):
        (directory/'final-status.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps({'workflow':workflow,'actual_gpu_hours':row['actual_gpu_hours'],'charged_or_reserved_total':ledger['previous_study_gpu_hours']+sum(accounting.charge(r) for r in ledger['entries'])}),flush=True)
        break
    start=row.get('allocation_start_time')
    if not state.get('start_time') and start and not cancel_requested:
        stamp=datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        if (datetime.now(timezone.utc)-stamp).total_seconds()>1200:
            p=subprocess.run(['/usr/local/bin/osmo','workflow','cancel',workflow],capture_output=True,text=True,timeout=45)
            (directory/'startup-cancel.log').write_text(p.stdout+p.stderr)
            cancel_requested=p.returncode==0
else:
    raise SystemExit('Monitor deadline reached; reservation retained for reconciliation')
