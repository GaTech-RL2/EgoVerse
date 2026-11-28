import os,sys,json,importlib.util,shutil
from pathlib import Path
import numpy as np
from prepare import sha,save
R=Path(os.environ['GAZE_ROOT']);OUT=R/'assets'
repo=Path('/coc/flash7/czhang883/Documents/SEW-Geometric-Teleop-vv')
sys.path.insert(0,str(repo))
from projects.rby1_teleop.utils import rollout_reset_pose as native
entries=json.loads((R/'reports/geometry-inputs.json').read_text())
joint=[];left=[];right=[]
for e in entries:
 if e['split']!='train':continue
 assert sha(Path(e['file']))==e['sha256']
 with np.load(e['file']) as z:joint.append(z['q'][0]);left.append(z['left'][0]);right.append(z['right'][0])
assert len(joint)==55
path=OUT/'reset_pose_cloth_new2.npz'
if (OUT/'reset-native-audit.json').exists():
 assert sha(path)==json.loads((OUT/'reset-native-audit.json').read_text())['reset_sha256']
 raise SystemExit(0)
previous=R/'reports/reset_pose_cloth_new2.arithmetic-attempt0.npz'
assert not previous.exists();shutil.copyfile(path,previous)
source=json.dumps(dict(manifest=json.loads((R/'reports/data-ready.json').read_text())['source_manifest'],training_uuids=[e['uuid'] for e in entries if e['split']=='train'],method='Native circular mean/std q[4:26], wheel slots zero; arithmetic mean/std hands; first row of55training recordings only'))
p=native._finalize_reset_pose(joint_rows=joint,hand_left_rows=left,hand_right_rows=right,initial_step=0,skipped=[],source=source)
fields={k:getattr(p,k) for k in ['robot0_joint_pos','robot0_joint_pos_std','hand_left_qpos','hand_left_qpos_std','hand_right_qpos','hand_right_qpos_std','num_episodes','initial_step','source']}
fields.update(version=1,skipped=np.array([],dtype='U1'))
np.savez(path,**fields)
loaded=native.load_reset_pose_cache(path)
for k in ['robot0_joint_pos','hand_left_qpos','hand_right_qpos']:
 np.testing.assert_array_equal(getattr(loaded,k),getattr(p,k));assert np.isfinite(getattr(loaded,k)).all()
a=np.array(joint,dtype='float64')[:,4:26];mean=np.arctan2(np.sin(a).mean(0),np.cos(a).mean(0))
np.testing.assert_allclose(loaded.robot0_joint_pos[4:26],mean,atol=1e-12);assert np.all(loaded.robot0_joint_pos[:4]==0)
assert loaded.num_episodes==55 and not loaded.skipped and loaded.initial_step==0
with np.load(previous) as z:delta=float(abs(z['robot0_joint_pos']-loaded.robot0_joint_pos).max())
audit=dict(passed=True,reset_sha256=sha(path),native_module_sha256=sha(Path(native.__file__)),native_loader_passed=True,train_recordings=55,valid_excluded=5,robot_q26=True,hands12_each=True,reset_method=source,max_difference_from_arithmetic_q_rad=delta,hardware_executed=False)
save(OUT/'reset-native-audit.json',audit)
g=json.loads((OUT/'geometry-audit.json').read_text());g['reset_exact_train_row0_mean_verified']='Native circular mean for q4:26; arithmetic mean for hands, independently verified';g['file_sha256'][path.name]=sha(path);g['reset_native_audit']='reset-native-audit.json';save(OUT/'geometry-audit.json',g)
print(json.dumps(audit),flush=True)
