from pathlib import Path
import shutil,json,hashlib
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');out=R/'cart-novfilter'
for d in ['source','clouds_hd2','tmp']:(out/d).mkdir(parents=True,exist_ok=True)
src=Path('/coc/flash7/czhang883/warp-data/pipelines/retarget_fix_rebuild_1124/clouds/real_dp3c_tri_gz_vis_hd2_fix1124_v2.py');dst=out/'source/clouds.py'
if not dst.exists():shutil.copy2(src,dst)
shell=Path('/coc/flash7/czhang883/warp-data/pipelines/retarget_fix_rebuild_1124/clouds/build_cart_fix1124_v2.sbatch').read_text();shell=shell[shell.index('set -uo pipefail'):]
shell=shell.replace('set -uo pipefail','set -euo pipefail\nexport RUN=fix1124 NSTRIPE=107\nexport PYTHONDONTWRITEBYTECODE=1')
shell=shell.replace('export TMPDIR=/coc/flash7/czhang883/tmp PYTHONUNBUFFERED=1','export TMPDIR='+str(out/'tmp')+' PYTHONUNBUFFERED=1')
shell=shell.replace('TOY_OUTD=$OUT','TOY_OUTD='+str(out/'clouds_hd2')).replace('TOY_VIS=1','TOY_VIS=0').replace('python $P/clouds/real_dp3c_tri_gz_vis_hd2_fix1124_v2.py','python '+str(dst))
shell+='\npython '+str(R/'scripts/novfilter_validate.py')+' "${SLURM_ARRAY_TASK_ID:-0}"\n'
(R/'scripts/cart-novfilter-cloud.sbatch').write_text('#!/bin/bash\n'+shell)
(out/'recipe.json').write_text(json.dumps(dict(virtual_camera_filter=False,look_at=False,cloud_source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),outputs=str(out/'clouds_hd2'),other_settings='identical to fixed non-LA upstream Cart hd2'),indent=2))
