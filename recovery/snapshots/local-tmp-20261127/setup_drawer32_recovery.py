import pathlib,json,subprocess,hashlib,ast,datetime
import numpy as np
R=pathlib.Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');S='/opt/slurm/Ubuntu-24.04/current/bin/'
m=json.load(open(R/'drawer/segment-manifest.json'));e=m[32];p=pathlib.Path(e['segment']);attempt=p/'attempts/cpu-node-recovery-4872397';assert not attempt.exists()
state=subprocess.check_output([S+'sacct','-j','4872397','-n','-X','-P','--format=JobID,State,ExitCode,NodeList'],text=True);assert '|TIMEOUT|'in state
assert not subprocess.check_output([S+'squeue','-h','-j','4872397','-o','%i'],text=True).strip()
old=json.load(open(p/'retarget.intent.json'));assert old['job']=='4872397';assert not(p/'retarget/fix.hdf5').exists()
work=p/'retarget/work';tag=e['retarget_tag'];a=np.load(work/(tag+'_aria60.npz'));rgb=np.load(work/(tag+'_rgb224.npz'));smpl=np.load(work/(tag+'_smpl.npz'))
np.testing.assert_array_equal(rgb['grid_us'],a['ts'].astype(np.int64));assert len(rgb['sel'])==len(a['ts'])and rgb['frames'].shape[1:]==(224,224,3)and rgb['frames'].dtype==np.uint8 and rgb['sel'].max()<len(rgb['frames'])
np.testing.assert_array_equal(smpl['orig_frame'],np.arange(len(a['ts'])));assert abs(smpl['head_p_in'][0,2]-1.57)<1e-6 and np.isfinite(smpl['positions']).all()
reused={str(work/(tag+ext)):hashlib.sha256((work/(tag+ext)).read_bytes()).hexdigest()for ext in ['_rgb224.npz','_smpl.npz','_human.csv']}
attempt.mkdir(parents=True);(attempt/'logs').mkdir()
audit=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),index=32,uuid=e['uuid'],original_intent=old,original_intent_sha256=hashlib.sha256((p/'retarget.intent.json').read_bytes()).hexdigest(),slurm=state,original_logs_sha256={x.name:hashlib.sha256(x.read_bytes()).hexdigest()for x in(p/'logs').glob('*.log')},reused=reused,reason='Original30min CPU job timed out on hal during unfinished fixed retarget, after completing RGB/EgoPoser/bone. Reuse validated completed inputs; resume only fixed retarget and downstream BS/QC on another CPU node. Original intent/log untouched.',attempt=str(attempt))
(R/'reports/drawer32-recovery-audit.json').write_text(json.dumps(audit,indent=2))
t=(R/'scripts/drawer_retarget_cpu.py').read_text()
t=t.replace("assert not(p/'retarget.intent.json').exists(),'Unresolved attempt; audit required'", "assert int(sys.argv[1])==32\nassert json.load(open(p/'retarget.intent.json'))['job']=='4872397'\nattempt=p/'attempts/cpu-node-recovery-4872397'\nassert not(attempt/'retarget.intent.json').exists(),'Unresolved recovery'\nassert not(p/'retarget/fix.hdf5').exists()")
t=t.replace("(p/'retarget.intent.json').write_text", "(attempt/'retarget.intent.json').write_text").replace("with(p/'logs'/(name+'.log')).open('x')", "with(attempt/'logs'/(name+'.log')).open('x')").replace("(p/'processing-error.json').write_text", "(attempt/'processing-error.json').write_text")
t=t.replace("def run(name,args,cwd=None):", "def run(name,args,cwd=None):\n if name in ['rgb','egoposer','bone']:\n  audit=json.load(open(R/'reports/drawer32-recovery-audit.json'))\n  assert all(sha(f)==v for f,v in audit['reused'].items())\n  print('REUSE_COMPLETED_STAGE',name,flush=True);return")
ast.parse(t);(R/'scripts/drawer_retarget_cpu32_recovery.py').write_text(t)
w=(R/'scripts/drawer_retarget_cpu_then_qc.py').read_text().replace('drawer_retarget_cpu.py','drawer_retarget_cpu32_recovery.py');ast.parse(w);(R/'scripts/drawer_cpu32_recovery_then_qc.py').write_text(w)
b=(R/'scripts/drawer-retarget-cpu.sbatch').read_text().replace('drawer_retarget_cpu_then_qc.py','drawer_cpu32_recovery_then_qc.py');(R/'scripts/drawer-cpu32-recovery.sbatch').write_text(b)
subprocess.run(['python3',str(R/'scripts/submit_stage.py'),'drawer-cpu32-node-recovery',str(R/'scripts/drawer-cpu32-recovery.sbatch'),'--array=32','--mem=16G','--time=00:45:00','--exclude=brainiac,spot,hal'],check=True)
