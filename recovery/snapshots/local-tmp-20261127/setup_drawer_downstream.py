import ast,shutil,json,hashlib
from pathlib import Path
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');D=R/'drawer/source/downstream';D.mkdir(exist_ok=True)
old=Path('/coc/flash7/cxin34/human-ondevice-pilot-20261110/source')
for n in ['rgbd.py','dino_hdf5.py']:shutil.copy2(old/n,D/n)
s=(R/'cart-novfilter/source/clouds.py').read_text()
s=s.replace('f=f"{MPS_ROOT}/mps_{vrsn[:-4]}_vrs/eye_gaze/general_eye_gaze.csv"','f=os.environ["GAZE_CSV"]')
s=s.replace('df["pitch_rads_cpf"].to_numpy(float))','df["pitch_rads_cpf"].to_numpy(float), df[["origin_x_cpf","origin_y_cpf","origin_z_cpf"]].to_numpy(float))')
s=s.replace('gts,gyaw,gpit=mps_gaze','gts,gyaw,gpit,gorig=mps_gaze')
s=s.replace('o_r=(Trd@np.append(o_dev,1.0))[:3];', 'origin_cpf=np.array([np.interp(int(ts[r]),gts,gorig[:,k]) for k in range(3)])\n            o_dev_row=T_dev_cpf[:3,3]+T_dev_cpf[:3,:3]@origin_cpf\n            o_r=(Trd@np.append(o_dev_row,1.0))[:3];')
assert 'o_dev_row' in s and 'gts,gyaw,gpit,gorig=' in s;ast.parse(s);(D/'clouds_ondevice_exact.py').write_text(s)
# Frozen model assets are only read. All writable caches belong to cxin34.
for n in ['depth','reports']:(R/'drawer'/n).mkdir(exist_ok=True)
(D/'source-provenance.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in D.glob('*.py')},indent=2))
(R/'scripts/drawer-downstream.sbatch').write_text('#!/bin/bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4\nexec /coc/flash7/cxin34/mobilemani-pretraining/env-train/bin/python '+str(R/'scripts/drawer_downstream.py')+' "${SLURM_ARRAY_TASK_ID}"\n')
