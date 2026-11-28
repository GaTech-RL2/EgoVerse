import pathlib,subprocess,json,time,fcntl,datetime
R=pathlib.Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125')
lock=(R/'reports/drawer-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
m=json.load(open(R/'drawer/segment-manifest.json'))
while True:
 p=subprocess.run(['python3',str(R/'scripts/drawer_dispatch_once.py')])
 if p.returncode:raise SystemExit(p.returncode)
 live=json.load(open(R/'reports/drawer-downstream-dispatch-live.json'))
 qc=[json.load(open(pathlib.Path(e['segment'])/'retarget-proof-v2.json'))for e in m if(pathlib.Path(e['segment'])/'retarget-proof-v2.json').exists()]
 status=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),retarget_terminal=len(qc),retarget_pass=sum(bool(x['passed'])for x in qc),downstream_pass=len(live['passed']),outstanding=live['outstanding'],status=live['status'])
 if live['status']!='ACTIVE':status['controller']='STOPPED_FOR_AUDIT'
 elif len(qc)==len(m) and all(not x['passed'] or x['index'] in live['passed'] or(pathlib.Path(m[x['index']]['segment'])/'downstream-geometry-quarantine.json').exists()for x in qc):status['controller']='READY_FOR_ASSEMBLY'
 else:status['controller']='RUNNING'
 (R/'reports/drawer-controller-live.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
 if status['controller']!='RUNNING':break
 time.sleep(45)
