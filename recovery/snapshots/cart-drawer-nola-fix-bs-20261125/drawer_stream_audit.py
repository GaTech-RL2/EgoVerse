"""Read embedded streams only. No interpolation, retarget, MPS or training."""
import hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125')
index=int(sys.argv[1]);records=json.load(open(R/'drawer/input-manifest.json'))['records']
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
for rec in records[index::4]:
    e=R/'drawer/episodes'/rec['uuid'];out=e/'stream-audit.json'
    if out.exists():continue
    report=dict(uuid=rec['uuid'],head_height_m=1.57,source='ondevice',mps_requested=False,training_ready=False)
    try:
        assert sha(rec['vrs'])==rec['sha256'],'raw hash mismatch'
        dp=data_provider.create_vrs_data_provider(rec['vrs']);streams={dp.get_label_from_stream_id(s):s for s in dp.get_all_streams()}
        report['streams']={k:dp.get_num_data(s)for k,s in streams.items()}
        assert all(k in streams for k in ['eyegaze','handtracking','vio_high_frequency','vio'])
        cal=dp.get_device_calibration();T=cal.get_transform_device_cpf().to_matrix();assert T.shape==(4,4)and np.isfinite(T).all()
        report['T_device_cpf']=T.tolist()
        vt=[];valid=[]
        for i in range(dp.get_num_data(streams['vio'])):
            v=dp.get_vio_data_by_index(streams['vio'],i);vt.append(v.capture_timestamp_ns);valid.append(str(v.status)=='VioStatus.VALID')
        tt=[];both=[]
        for i in range(dp.get_num_data(streams['handtracking'])):
            s=dp.get_sensor_data_by_index(streams['handtracking'],i);h=s.hand_pose_data()
            tt.append(s.get_time_ns(TimeDomain.DEVICE_TIME));both.append(all(getattr(h,side+'_hand')is not None and getattr(h,side+'_hand').confidence>0 for side in ['left','right']))
        tt=np.asarray(tt);both=np.asarray(both);assert np.all(np.diff(tt)>0)
        # Candidate intervals only; retarget output duration and full FK/QC still required.
        intervals=[];start=None
        for i in range(len(tt)+1):
            ok=i<len(tt)and both[i]and(i==0 or tt[i]-tt[i-1]<100_000_000)
            if ok and start is None:start=i
            if not ok and start is not None:
                if(tt[i-1]-tt[start])/1e9>1:intervals.append([int(tt[start]),int(tt[i-1])])
                start=i if i<len(tt)and both[i]else None
        gv=0
        for i in range(dp.get_num_data(streams['eyegaze'])):
            g=dp.get_eye_gaze_data_by_index(streams['eyegaze'],i)
            gv+=bool(g.combined_gaze_valid and np.isfinite(np.r_[g.yaw,g.pitch,np.asarray(g.combined_gaze_origin_in_cpf)]).all())
        good=np.flatnonzero(valid);assert len(good)>0 and gv>10 and intervals
        report.update(passed=True,raw_sha256=rec['sha256'],hand_frames=len(tt),both_valid_frames=int(both.sum()),candidate_intervals_ns=intervals,gaze_valid=gv,vio_total=len(valid),vio_valid=sum(valid),vio_nonstartup_outage=not all(valid[good[0]:]),first_valid_vio_ns=int(vt[good[0]]),last_vio_ns=int(vt[-1]))
    except Exception:
        report.update(passed=False,error=traceback.format_exc())
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(out)
    print(json.dumps({k:report[k]for k in ['uuid','passed','training_ready']}),flush=True)
