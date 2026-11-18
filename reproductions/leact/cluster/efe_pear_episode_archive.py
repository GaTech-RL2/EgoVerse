"""Offload completed native rollout artifacts before pruning local caches."""
import json
import os
from pathlib import Path


def completed_episode(folder,identity):
    prefix=os.environ.get('PEAR_EPISODE_ARCHIVE')
    if not prefix:return None
    folder=Path(folder)
    try:
        from efe_pear_object_archive import archive
        destination=prefix.rstrip('/')+'/'+identity
        manifest=archive(folder,destination,workers=2)
        if manifest.get('readback_verified')is not True:
            raise ValueError('Native artifact archive has no verified readback; local files retained')
        # Preserve every small native JSON verdict/result locally as well as
        # the object-store manifest. Actions/reasoning are also logged by PEAR.
        retained={p for p in folder.rglob('*.json')if p.is_file()and p.stat().st_size<256<<10}
        for path in folder.rglob('*'):
            if path.is_file()and path not in retained:path.unlink()
        receipt={'archive':destination,'bytes':manifest['bytes'],'readback_verified':True,
            'retention':'Full native artifacts in private object storage; small native JSON records retained locally.'}
        (folder/'rp_jhh_native_archive.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return receipt
    except Exception as exc:
        # Storage outages do not rewrite an already-finalized native reward.
        # Keep original files so the collector can retry an offload safely.
        print('PEAR_EPISODE_ARCHIVE_RETRY '+json.dumps({'identity':identity,'error_type':type(exc).__name__}),flush=True)
        return None
