import os,sys,json,time,traceback
from pathlib import Path
from finish import run,save
R=Path('/coc/flash7/cxin34/egoversea-multitask-20261119')
P=Path('/coc/flash7/cxin34/human-ondevice-pilot-20261110')
SEW=P/'source/sew';AE=SEW/'projects/xhand_teleop/aria_egoposer'
EGOROOT=Path('/coc/flash7/czhang883/Documents/mobile_generalist/data/aria_egoposer_warp_handoff/aria_egoposer_warp_handoff/02_egoposer')
TRAIN='/coc/flash7/cxin34/mobilemani-pretraining/env-train/bin/python'
DEPTH='/coc/flash7/scratch/czhang883/envs/aria_fs/bin/python'
EGO='/coc/flash7/czhang883/miniconda3/envs/egoposer/bin/python'
RBY='/coc/flash7/czhang883/miniconda3/envs/rby1/bin/python'
i=int(sys.argv[1]);item=json.loads((R/'plan.json').read_text())[i];tag=item['uuid'];e=Path(item['episode']);w=e/'retarget/work'
if item.get('reused'):
 assert (e/'preprocess-proof.json').exists();print('REUSED_VERIFIED_BENCHMARK',tag,flush=True);raise SystemExit(0)
for marker in ['input-quality-quarantine.json','input-processing-quarantine.json']:
 if (e/marker).exists():print('ISOLATED_INPUT',tag,marker,flush=True);raise SystemExit(0)
assert (e/'input.json').exists(),'Record preparation missing; audit record result before recovery'
item.update(json.loads((e/'input.json').read_text()))
import shutil
from storage_interlock import require_capacity
require_capacity(R,i,e)

env=os.environ.copy();env.update(EGOPOSER_ROOT=str(EGOROOT),SEW_REPO=str(SEW),ARIA_ARTIFACTS=str(w),PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='egl',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MPLCONFIGDIR=str(R/'cache/matplotlib'),TORCH_HOME=str(R/'cache/torch'))
env.pop('PYTHONPATH',None);env.pop('VIRTUAL_ENV',None)
assert not (e/'worker.intent.json').exists(),'Existing worker requires audit'
save(e/'worker.intent.json',dict(job_id=os.environ.get('SLURM_JOB_ID'),time=time.time(),source_mode='MPS general Gen1'))
try:
 run(e,'raw-rgb',[TRAIN,str(R/'scripts/extract_upright_rgb.py'),str(i)],env)
 os.chdir(EGOROOT)
 run(e,'egoposer',[EGO,str(AE/'run_aria_egoposer.py'),'--aria_npz',str(w/(tag+'_aria60.npz')),'--out',str(w/(tag+'_smpl.npz'))],env)
 os.chdir(SEW)
 run(e,'bones',[RBY,str(AE/'write_bone_csv.py'),'--smpl_npz',str(w/(tag+'_smpl.npz')),'--out',str(w/(tag+'_human.csv'))],env)
 run(e,'retarget',[RBY,str(AE/'build_training_hdf5.py'),'--no_firm','--res','224','--tags',tag,'--out',str(e/'retarget/episode-untrimmed.hdf5')],env)
 run(e,'trim',[RBY,str(AE/'trim_hdf5.py'),'--in_hdf5',str(e/'retarget/episode-untrimmed.hdf5'),'--out_hdf5',str(e/'retarget/episode.hdf5')],env)
 os.chdir(R)
 run(e,'depth',[DEPTH,str(R/'source/fast_stereo_gen1.py'),'--vrs',item['vrs'],'--output-dir',str(e/'depth'/Path(item['vrs']).stem),'--frame-step','4','--start-ns',str(item['start_ns']),'--end-ns',str(item['end_ns']),'--fs-home','/coc/flash7/yzhu827/Fast-FoundationStereo','--ckpt','/coc/flash7/yzhu827/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth'],env)
 run(e,'finish',[TRAIN,str(R/'scripts/finish.py'),tag],env)
 if (e/'quality-quarantine.json').exists():
  save(e/'worker-result.json',dict(status='QUALITY_QUARANTINED',time=time.time(),job_id=os.getenv('SLURM_JOB_ID')))
 else:
  assert (e/'preprocess-proof.json').exists()
  run(e,'tensor',[TRAIN,str(R/'scripts/tensor_stage.py'),str(i)],env)
  save(e/'worker-result.json',dict(status='TENSORS_STAGED_PENDING_INDEPENDENT_AUDIT',time=time.time(),job_id=os.getenv('SLURM_JOB_ID')))
except Exception:
 save(e/'worker-result.json',dict(status='FAILED_REQUIRES_DIAGNOSIS',time=time.time(),job_id=os.getenv('SLURM_JOB_ID'),traceback=traceback.format_exc()));raise
