import ast,shutil
from pathlib import Path
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');p=R/'scripts/drawer_retarget.py';b=p.with_name('drawer_retarget.before_rgb_merge.py')
if not b.exists():shutil.copy2(p,b)
s=b.read_text();start=s.index(" run('bs-gate'");end=s.index('\nexcept Exception:',start)
s=s[:start]+" subprocess.run([py,str(R/'scripts/drawer_finalize.py'),sys.argv[1]],env=env,check=True)"+s[end:];ast.parse(s);p.write_text(s)
(R/'scripts/drawer-pilot-finalize.sbatch').write_text('#!/bin/bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl\nexec /coc/flash7/czhang883/miniconda3/envs/rby1/bin/python '+str(R/'scripts/drawer_finalize.py')+' 0\n')
