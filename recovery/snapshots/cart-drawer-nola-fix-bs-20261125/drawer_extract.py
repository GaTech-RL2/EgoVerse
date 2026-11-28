"""Independent calibrated head audit and strict uninterrupted hand segments."""
import os,sys,json,hashlib,uuid,traceback,csv,fcntl
from pathlib import Path
import numpy as np
from projectaria_tools.core import data_provider
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');sys.path.insert(0,str(R/'drawer/source/aria_egoposer'))
from aria_vrs_reader import parse_vrs
records=json.load(open(R/'drawer/input-manifest.json'))['records']
for rec in records[int(sys.argv[1])::4]:
 e=R/'drawer/episodes'/rec['uuid'];out=e/'extract-proof.json'
 if out.exists():continue
 lock=(e/'extract.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  report=json.load(open(e/'stream-audit.json'));assert report['passed'] and not report['vio_nonstartup_outage']
  bounds=dict(valid_window_us=[report['first_valid_vio_ns']/1000,report['last_vio_ns']/1000]);(e/'reader-bounds.json').write_text(json.dumps(bounds));os.environ['ONDEVICE_AUDIT']=str(e/'reader-bounds.json')
  d=parse_vrs(rec['vrs']);n=d['n'];T=np.asarray(report['T_device_cpf']);F=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
  np.testing.assert_allclose(d['head_p'],d['dev_p']+d['dev_R']@T[:3,3],atol=1e-9)
  np.testing.assert_allclose(d['head_R'],d['dev_R']@T[:3,:3]@F,atol=1e-9)
  valid=d['left_valid']&d['right_valid'];z=np.diff(np.r_[False,valid,False].astype(int));starts=np.flatnonzero(z==1);ends=np.flatnonzero(z==-1)
  segments=[]
  for j,(aa,bb)in enumerate(zip(starts,ends)):
   if d['ts'][bb-1]-d['ts'][aa]<=1e6:continue
   sid=str(uuid.uuid5(uuid.UUID(rec['uuid']),f'ondevice-cpf-h157-{aa}-{bb}'));p=R/'drawer/segments'/sid;(p/'retarget/work').mkdir(parents=True,exist_ok=True);(p/'logs').mkdir(exist_ok=True)
   tag='drawer_'+sid.replace('-','');data={k:(v[aa:bb]if isinstance(v,np.ndarray)and v.ndim and len(v)==n else v)for k,v in d.items()if k!='vrs'};data['n']=int(bb-aa)
   assert all(np.isfinite(v).all()for v in data.values()if isinstance(v,np.ndarray))
   assert data['left_valid'].all() and data['right_valid'].all()
   f=p/'retarget/work'/(tag+'_aria60.npz');np.savez_compressed(f,**data)
   item=dict(uuid=sid,recording_id=rec['uuid'],raw_sha256=rec['sha256'],vrs=rec['vrs'],split=rec['split'],head_height_m=1.57,rows_before_retarget=int(bb-aa),start_us=float(data['ts'][0]),end_us=float(data['ts'][-1]),retarget_tag=tag,segment=str(p),source='ondevice_combined',look_at=False)
   (p/'input.json').write_text(json.dumps(item,indent=2));segments.append(item)
  assert segments,'No >1s strict bilateral-valid segment'
  dp=data_provider.create_vrs_data_provider(rec['vrs']);sid=dp.get_stream_id_from_label('eyegaze');rows=[]
  for i in range(dp.get_num_data(sid)):
   g=dp.get_eye_gaze_data_by_index(sid,i);v=np.r_[g.yaw,g.pitch,np.asarray(g.combined_gaze_origin_in_cpf)]
   if g.combined_gaze_valid and np.isfinite(v).all():rows.append([g.tracking_timestamp.total_seconds()*1e6,*v])
  with (e/'gaze/ondevice_eye_gaze-v2.csv').open('w')as f:
   w=csv.writer(f);w.writerow(['tracking_timestamp_us','yaw_rads_cpf','pitch_rads_cpf','origin_x_cpf','origin_y_cpf','origin_z_cpf']);w.writerows(rows)
  result=dict(passed=True,recording_id=rec['uuid'],cpf_position_and_rotation_audited=True,tracking_source='ondevice',mps_requested=False,segments=segments,total_rows=sum(s['rows_before_retarget']for s in segments),strict_interpolation_endpoints=True,training_ready=False)
 except Exception:result=dict(passed=False,recording_id=rec['uuid'],error=traceback.format_exc(),training_ready=False)
 out.write_text(json.dumps(result,indent=2));print('EXTRACT',rec['uuid'],result['passed'],result.get('total_rows'),flush=True)
