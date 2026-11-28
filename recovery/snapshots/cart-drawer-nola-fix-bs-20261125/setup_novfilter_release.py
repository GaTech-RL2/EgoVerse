from pathlib import Path
import ast
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');W=R/'training/cart/novfilter'
s=(R/'scripts/cart_release.py').read_text().replace("R/'cart/", "R/'cart-novfilter/").replace('virtual_camera_filter=True','virtual_camera_filter=False')
ast.parse(s);(R/'scripts/novfilter_release.py').write_text(s)
for name,body in {
 'cart-novfilter-export.sbatch':'source '+str(W/'scripts/env.sh')+'\nexec "$CAMPAIGN_PYTHON" '+str(R/'scripts/novfilter_export.py'),
 'cart-novfilter-release.sbatch':'source '+str(W/'scripts/env.sh')+'\n"$CAMPAIGN_PYTHON" '+str(R/'scripts/novfilter_review_gate.py')+'\nexec "$CAMPAIGN_PYTHON" '+str(R/'scripts/novfilter_release.py')+' novfilter',
 'cart-novfilter-ft-chain.sbatch':'export CAMPAIGN_ROOT='+str(W)+'\nexport RUN_ID=cart-nola-fix-bs-novfilter-finetune-MSCstrong-s42\nexec bash "$CAMPAIGN_ROOT/scripts/chain.sbatch"',
}.items():(R/'scripts'/name).write_text('#!/bin/bash\nset -euo pipefail\n'+body+'\n')
