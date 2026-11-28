import pathlib,json,hashlib,datetime,subprocess
R=pathlib.Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');D=R/'drawer';d=D/'reports'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
state=subprocess.check_output(['/opt/slurm/Ubuntu-24.04/current/bin/sacct','-j','4872933','-n','-X','-P','-o','JobID,State,ExitCode'],text=True);assert '4872933|COMPLETED|0:0'in state
for i in [24,91]:
 out=d/f'reviewed-{i:03d}.json';assert not out.exists()
 p=d/f'independent-{i:03d}.json';a=json.load(open(p));assert a['passed']and sha(a['image'])==a['image_sha256']
 z=dict(passed=True,index=i,uuid=a['uuid'],time=datetime.datetime.now(datetime.timezone.utc).isoformat(),image=a['image'],image_sha256=a['image_sha256'],independent_sha256=sha(p),scope='Actually reviewed start/middle/end RGB and all three clouds; all-row numeric checks separate; no full retarget video reviewed.',observations='Coherent drawer approach and manipulation, three nonempty clouds, intended grey hand regions. Sparse dark cabinet depth and final right-local wall-dominated view after hand exits. No complete per-frame hand/object coverage claim.')
 with out.open('x')as f:json.dump(z,f,indent=2)
audit=json.load(open(d/'export-audit.json'));assert audit['passed']and audit['manifest_sha256']==sha(D/'target.json')
m=json.load(open(D/'target.json'));segments=json.load(open(D/'segment-manifest.json'));byid={e['uuid']:i for i,e in enumerate(segments)}
images={};reviews={};numeric={}
for e in m['episodes']:
 i=byid[e['uuid']];p=d/f'independent-{i:03d}.json';a=json.load(open(p));v=d/f'reviewed-{i:03d}.json';b=json.load(open(v))
 assert a['passed']and b['passed']and a['uuid']==b['uuid']==e['uuid']
 assert b['independent_sha256']==sha(p)and a['image_sha256']==b['image_sha256']==sha(a['image'])
 assert a['episode_sha256']==sha(pathlib.Path(segments[i]['segment'])/'downstream/episode.json')
 images[a['image']]=a['image_sha256'];reviews[str(v)]=sha(v);numeric[str(p)]=sha(p)
z=dict(passed=True,manifest_sha256=sha(D/'target.json'),export_audit_sha256=sha(d/'export-audit.json'),images=images,review_reports=reviews,independent_reports=numeric,accepted_segments=len(m['episodes']),sampled_frames_per_segment=3,scope='All accepted segments: actual start/middle/end RGB plus global/right/left cloud review; all-row numerical FK/hash/time/finite/action alignment checked separately. No full robot-mesh/retarget video review.',limitations=['Dark cabinet surfaces sparse in depth.','Some ending right-hand crops are wall-dominated when the hand leaves RGB view.','BS retains pre-BS camera observations as in Cart recipe; no exact post-BS reprojection claim.','Original split preserved before quality exclusions; same-day validation.'],time=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_ready=False)
with(d/'visual-review.json').open('x')as f:json.dump(z,f,indent=2)
print('MANIFEST_BOUND_VISUAL_REVIEW',len(images),'segments')
