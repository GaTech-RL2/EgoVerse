from pathlib import Path
import hashlib,json
R=Path('/coc/flash7/cxin34/cloth-new2-scratch-20261121');AE=Path('/coc/flash7/czhang883/Documents/SEW-aria-pipeline/projects/xhand_teleop/aria_egoposer')
files=[*R.glob('scripts/*.py'),*R.glob('scripts/*.sbatch'),*R.glob('source/*.py'),*AE.glob('*.py'),*AE.glob('*.sh'),Path('/coc/flash7/czhang883/Documents/aria_gen2_scripts/src/run_fast_stereo_depth_to_npy.py'),Path('/coc/flash7/czhang883/Documents/aria_gen2_scripts/src/utils/stereo_utils.py')]
rows={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(R/'reports/preprocessing-source-hashes.json').write_text(json.dumps(rows,indent=2));print('SOURCE_FILES',len(rows))
