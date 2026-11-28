"""Explicit release/normalization scope; target-only scratch never changes source/FT rules."""
def release_path(root,cfg):return root/'releases'/cfg.get('release_id',cfg['task'])

def check_normalizer(cfg,norm,manifest=None,manifest_sha=None):
    expected=cfg.get('normalization_fit_partition','source_train')
    assert expected in ('source_train','target_train') and norm['fit_partition']==expected
    if expected=='target_train':
        assert cfg['task']=='cart' and cfg['stage']=='scratch' and cfg['split_role']=='target'
        assert cfg['arm'] in ('B','E') and cfg['release_id']=='cart-target-scratch'
        if manifest is not None:
            train={e['uuid'] for e in manifest['episodes'] if e['split']=='train'}
            valid={e['uuid'] for e in manifest['episodes'] if e['split']=='valid'}
            assert len(train)==30 and not train&valid
            assert set(norm['train_uuids'])==train and norm['manifest_sha256']==manifest_sha
