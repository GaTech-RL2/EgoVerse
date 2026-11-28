import os,sys,json,subprocess,time,shutil
from pathlib import Path
from finish import run,save
R=Path('/coc/flash7/cxin34/cloth-new2-scratch-20261121');item=json.loads((R/'plan.json').read_text())[int(sys.argv[1])];e=Path(item['episode'])
assert shutil.disk_usage(R).free>120*1024**3,'Storage below120GiB'
C=Path('/coc/flash7/czhang883/miniconda3/envs');SEW=Path('/coc/flash7/czhang883/Documents/SEW-aria-pipeline')
PY='/coc/flash7/cxin34/mobilemani-pretraining/env-train/bin/python';DEP='/coc/flash7/scratch/czhang883/envs/aria_fs/bin/python'
env=os.environ.copy();env.pop('VIRTUAL_ENV',None);env.pop('PYTHONPATH',None)
env.update(ARIA_PY=str(C/'aria2/bin/python'),EGOPOSER_PY=str(C/'egoposer/bin/python'),RBY1_PY=str(C/'rby1/bin/python'),SEW_REPO=str(SEW),EGOPOSER_ROOT='/coc/flash7/czhang883/Documents/mobile_generalist/data/aria_egoposer_warp_handoff/aria_egoposer_warp_handoff/02_egoposer',ARIA_HANDS='mps',MPS_ROOT=str(Path(item['mps']).parent),ARIA_ARTIFACTS=str(e/'retarget/work'),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',MUJOCO_GL='egl',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TMPDIR=str(e/'tmp'),MPLCONFIGDIR=str(e/'tmp/mpl'),XDG_CACHE_HOME=str(R/'cache'))
assert not (e/'worker-result.json').exists(),'Existing result requires review'
try:
 run(e,'retarget',["bash",str(SEW/'projects/xhand_teleop/aria_egoposer/vrs_to_all.sh'),str(e/'input/episode'),str(e/'retarget'),'--njobs','1','--name',item['uuid']],env)
 run(e,'select-window',[PY,str(R/'scripts/select_window.py'),sys.argv[1]],env)
 if (e/'quality-quarantine.json').exists():raise SystemExit(0)
 run(e,'depth',[DEP,'/coc/flash7/czhang883/Documents/aria_gen2_scripts/src/run_fast_stereo_depth_to_npy.py','--vrs',item['vrs'],'--output-dir',str(e/'depth'/Path(item['vrs']).stem),'--frame-step','4','--fs-home','/coc/flash7/yzhu827/Fast-FoundationStereo','--ckpt','/coc/flash7/yzhu827/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth'],env)
 run(e,'finish',[PY,str(R/'scripts/finish.py'),item['uuid']],env)
 if (e/'quality-quarantine.json').exists():raise SystemExit(0)
 run(e,'tensor',[PY,str(R/'scripts/tensor_stage.py'),sys.argv[1]],env)
 run(e,'independent-audit',[PY,str(R/'scripts/audit_item.py'),sys.argv[1]],env)
 save(e/'worker-result.json',{'status':'STAGED','job_id':os.getenv('SLURM_JOB_ID')})
except Exception as ex:
 save(e/'worker-result.json',{'status':'FAILED','error':str(ex),'job_id':os.getenv('SLURM_JOB_ID')});raise
