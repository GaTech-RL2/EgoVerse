from pathlib import Path
import json,hashlib,shutil,subprocess,ast
B=Path('/coc/flash7/cxin34');W=B/'egoa-shared-fix1124-ft-20261124';C=W/'deploy-control'
C.mkdir(exist_ok=True);(C/'logs').mkdir(exist_ok=True)
with (C/'prepare.intent').open('x')as f:f.write('Shared fix1124 Cart look-at and Basket exact10k offline deployment\n')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for v in iter(lambda:f.read(8388608),b''):h.update(v)
 return h.hexdigest()
plans=[]
for task,label,run,port,ntrain,nvalid in [
 ('cart','cart-la','egoa180-cart-la-sharedfix1124-ft60k-MSCstrong-s42',8280,30,10),
 ('basket','basket55','egoa180-basket55-sharedfix1124-ft60k-MSCstrong-s42',8281,50,5)]:
 old=B/f'egoa180-{task}-fix-FT10k-robotpc-20261124';r=B/f'egoa180-{label}-sharedfix1124-FT10k-robotpc-20261124';w=W/task
 ck=w/'runs'/run/'step_010000.pt';expected=sha(ck);oldplan=json.load(open(old/'reports/plan.json'))
 r.mkdir(exist_ok=False)
 for d in ['scripts','client-support','gaze']:shutil.copytree(old/d,r/d,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for d in ['campaign','source']:shutil.copytree(w/d,r/d,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
 for d in ['reports','testdata','configs','assets']:(r/d).mkdir()
 for n in ['serve_robotpc.py','check_server.py']:shutil.copy2(old/n,r/n)
 for n in ['robot.json','robot.urdf']:shutil.copy2(w/'assets'/n,r/'assets'/n)
 shutil.copy2(w/'configs'/f'{run}.json',r/'configs/training.json');shutil.copy2(w/'configs/hd2-reference.yaml',r/'configs/hd2-reference.yaml')
 shutil.copy2(w/f'releases/{task}/normalizer.json',r/'normalizer.json')
 camera='virtual_cam_cart_1102.npz'if task=='cart'else'virtual_cam_basket.npz'
 newcamera='virtual_cam_cart_1102_la_sharedfix1124.npz'if task=='cart'else'virtual_cam_basket_sharedfix1124.npz'
 shutil.copy2(old/'assets'/camera,r/'reports/previous-virtual-camera.npz')
 plan=dict(task=task,run=run,source_checkpoint=str(ck),source_checkpoint_sha256=expected,target_manifest=str(w/f'releases/{task}/target.json'),step=10000,seed=42,train_recordings=ntrain,valid_recordings=nvalid,port=port,normalizer_sha256=sha(r/'normalizer.json'),training_config_sha256=sha(r/'configs/training.json'),gaze_source=oldplan['gaze_source'],gaze_retrained=False,gaze_sha256={p.name:sha(p)for p in (r/'gaze').glob('*.pt')},geometry='Shared fix1124 actual cloud tg transforms; train-only fit',package=str(r),preset='TGZ10HD2'if task=='cart'else'TGZ10HD2Z5',look_at=task=='cart',camera_file=newcamera)
 (r/'reports/plan.json').write_text(json.dumps(plan,indent=2))
 for p in [r/'serve_robotpc.py',r/'check_server.py',*(r/'scripts').iterdir()]:
  if p.is_file()and p.suffix in ['.py','.sh','.sbatch']:
   s=p.read_text().replace(str(old),str(r)).replace(old.name,r.name).replace(oldplan['source_checkpoint_sha256'],expected).replace(str(oldplan['port']),str(port)).replace(camera,newcamera)
   p.write_text(s)
 p=r/'scripts/build_geometry.py';s=p.read_text()
 s=s.replace("mapping={x['uuid']:x for x in json.load(open(P/'plan.json'))}","mapping={e['uuid']:e for e in manifest['episodes']}")
 oldblock="src=Path(mapping[e['uuid']]['episode']);a=json.load(open(src/'retarget-independent.json'));assert a['passed']\n h=src/'retarget/selected.hdf5';assert sha(h)==a['selected_hdf5_sha256']\n cp=src/'clouds/camera_transform_000000.npy';T=np.load(cp).reshape(-1,4,4).astype('float64')"
 newblock="h=Path(e['retarget_hdf5']);demo=e['retarget_demo'];n=e['rows']\n cloud='clouds_hd2la'if task=='cart'else'clouds_hd2z5'\n cp=h.parent.parent/cloud/('tg_%06d.npy'%int(demo.split('_')[-1]));T=np.load(cp).reshape(-1,4,4)[:n].astype('float64')"
 assert oldblock in s;s=s.replace(oldblock,newblock)
 s=s.replace("g=f['data/demo_0'];q=g['obs/robot0_joint_pos'][:].astype('float64');ts=g['obs/aria_ts'][:]","g=f['data'][demo];q=g['obs/robot0_joint_pos'][:n].astype('float64');ts=g['obs/aria_ts'][:n]\n  sp=Path(e['arrays']['state_robot0_joint_pos']['file']);assert sha(sp)==e['arrays']['state_robot0_joint_pos']['sha256']\n  np.testing.assert_allclose(q[:,4:26],np.load(sp),atol=1e-6,rtol=0)\n  gp=h.parent.parent/cloud/('g_hd2_%06d.npy'%int(demo.split('_')[-1]));original_cloud=np.load(gp,mmap_mode='r')[:n].reshape(n,2048,6)\n  ap=Path(e['arrays']['front_pcd_1']['file']);assert sha(ap)==e['arrays']['front_pcd_1']['sha256']\n  np.testing.assert_allclose(original_cloud,np.load(ap,mmap_mode='r'),atol=1e-6,rtol=0)")
 s=s.replace(";src=Path(mapping[e['uuid']]['episode'])",";src=Path(e['retarget_hdf5'])")
 s=s.replace("with h5py.File(src/'retarget/selected.hdf5')as h:q26=h['data/demo_0/obs/robot0_joint_pos'][:3]","with h5py.File(src)as h:q26=h['data'][e['retarget_demo']]['obs/robot0_joint_pos'][:3]")
 s=s.replace("with h5py.File(src/'retarget/selected.hdf5')as h:imgs=h['data/demo_0/obs/aria_image'][:3]","with h5py.File(src)as h:imgs=h['data'][e['retarget_demo']]['obs/aria_image'][:3]")
 s=s.replace('fixed CPF/measured-wrist target','shared fix1124 target').replace('previous_camera_same_fixed_validation','previous_camera_same_shared_validation')
 # Hash each shared multi-episode HDF5 once per immutable-stat snapshot.
 s=s.replace('def sha(p):\n h=hashlib.sha256()',"HASH_CACHE={}\ndef sha(p):\n p=Path(p);st=p.stat();key=(str(p),st.st_size,st.st_mtime_ns)\n if key in HASH_CACHE:return HASH_CACHE[key]\n h=hashlib.sha256()")
 s=s.replace(' return h.hexdigest()',' digest=h.hexdigest();assert(p.stat().st_size,p.stat().st_mtime_ns)==(st.st_size,st.st_mtime_ns);HASH_CACHE[key]=digest;return digest')
 p.write_text(s)
 for p in r.rglob('*.py'):ast.parse(p.read_text())
 for p in (r/'scripts').glob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
 plans.append(plan)
(C/'packages.json').write_text(json.dumps(plans,indent=2))
for plan in plans:
 task=plan['task'];r=Path(plan['package'])
 with(C/f'{task}-test.intent').open('x')as f:f.write(json.dumps(plan))
 cmd=['/opt/slurm/Ubuntu-24.04/current/bin/sbatch','--parsable','--account=rl2-lab','--partition=rl2-lab','--qos=debug','--cpus-per-task=4','--mem=48G','--time=01:00:00','--exclude=hal',f'--job-name=deploy-sharedfix-{task}-10k',f'--output={C}/logs/%j-{task}.out',str(r/'scripts/test.sbatch')]
 p=subprocess.run(cmd,capture_output=True,text=True);receipt=dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 (C/f'{task}-test-submit.json').write_text(json.dumps(receipt,indent=2));assert p.returncode==0,receipt;print(task,p.stdout.strip(),flush=True)
