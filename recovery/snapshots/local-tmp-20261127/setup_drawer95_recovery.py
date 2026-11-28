import pathlib,json,subprocess,hashlib,ast,datetime
R=pathlib.Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');S='/opt/slurm/Ubuntu-24.04/current/bin/'
e=json.load(open(R/'drawer/segment-manifest.json'))[95];p=pathlib.Path(e['segment']);depth=R/'drawer/depth'/e['recording_id'];new=p/'downstream-recovery-4872689'
assert not new.exists();assert not list(depth.glob('**/*.npy'));assert 'CUDA is required.'in(depth/'depth.log').read_text()
state=subprocess.check_output([S+'sacct','-j','4872689','-n','-X','-P','--format=JobID,State,ExitCode,NodeList'],text=True);assert '|FAILED|1:0|protocol'in state
assert not subprocess.check_output([S+'squeue','-h','-j','4872689','-o','%i'],text=True).strip()
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
audit=dict(index=95,uuid=e['uuid'],time=datetime.datetime.now(datetime.timezone.utc).isoformat(),slurm=state,original_depth_intent_sha256=sha(depth/'intent.json'),original_depth_log_sha256=sha(depth/'depth.log'),original_failure_sha256=sha(p/'downstream/failure.json'),original_downstream_intent_sha256=sha(p/'downstream/intent.json'),reason='protocol CUDA unavailable before any depth output. One bounded downstream-only recovery; retarget, original intents and logs retained.',excluded_node='protocol')
ap=R/'reports/drawer95-recovery-audit.json';assert not ap.exists();ap.write_text(json.dumps(audit,indent=2))
t=(R/'scripts/drawer_downstream.py').read_text().replace("out=p/'downstream';out.mkdir(exist_ok=True)","assert int(sys.argv[1])==95;out=p/'downstream-recovery-4872689';out.mkdir(exist_ok=True)")
t=t.replace("depthroot/'intent.json'","depthroot/'intent-recovery-4872689.json'").replace("depthroot/'depth.log'","depthroot/'depth-recovery-4872689.log'")
ast.parse(t);(R/'scripts/drawer_downstream95_recovery.py').write_text(t)
wrapper="""import pathlib,json,subprocess,hashlib
R=pathlib.Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125')
PY='/coc/flash7/cxin34/mobilemani-pretraining/env-train/bin/python'
subprocess.run([PY,str(R/'scripts/drawer_downstream95_recovery.py'),'95'],check=True)
e=json.load(open(R/'drawer/segment-manifest.json'))[95];p=pathlib.Path(e['segment']);new=p/'downstream-recovery-4872689';old=p/'downstream'
audit=json.load(open(R/'reports/drawer95-recovery-audit.json'));sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(old/'failure.json')==audit['original_failure_sha256'];proof=json.load(open(new/'proof.json'));assert proof['passed']and sha(new/'episode.json')==proof['episode_sha256']
proof['recovered_from_job']='4872689';proof['recovery_audit_sha256']=sha(R/'reports/drawer95-recovery-audit.json')
with(old/'episode.json').open('x')as f:f.write((new/'episode.json').read_text())
with(old/'proof.json').open('x')as f:json.dump(proof,f,indent=2)
print('DRAWER95_RECOVERY_PUBLISHED',flush=True)
"""
ast.parse(wrapper);(R/'scripts/drawer_publish95_recovery.py').write_text(wrapper)
b=(R/'scripts/drawer-downstream.sbatch').read_text().replace('drawer_downstream.py','drawer_publish95_recovery.py');(R/'scripts/drawer95-recovery.sbatch').write_text(b)
subprocess.run(['python3',str(R/'scripts/submit_stage.py'),'drawer-depth95-recovery',str(R/'scripts/drawer95-recovery.sbatch'),'--array=95','--gres=gpu:a40:1','--exclude=brainiac,spot,bishop,dynamics,protocol','--nice=0'],check=True)
