def release_path(root,cfg):
 assert cfg['stage']=='finetune' and cfg['split_role']=='target' and cfg['arm']=='B'
 assert cfg['release_id'] in ['cart','basket','cloth'] and cfg['formal_training_authorized']
 return root/'releases'/cfg['release_id']
def check_normalizer(cfg,norm,manifest=None,manifest_sha=None):
 assert norm['fit_partition']=='source_train' and norm['manifest_sha256']==cfg['source_identity']['manifest_sha256']
 tr={e['uuid'] for e in manifest['episodes'] if e['split']=='train'}
 va={e['uuid'] for e in manifest['episodes'] if e['split']=='valid'}
 _verify_adaptation(cfg,norm,manifest)
 assert len(tr)==cfg['train_recordings'] and len(va)==cfg['valid_recordings'] and not tr&va

# The original source manifest remains provenance; only explicitly newly active
# dimensions use target-training statistics, never validation.
def _verify_adaptation(cfg,norm,manifest):
 a=norm['adaptation'];c=cfg['normalizer_adaptation']
 assert c['passed'] and not a['valid_used'] and not c['valid_used']
 assert a['source_normalizer_sha256']==cfg['source_identity']['normalizer_sha256']==c['source_sha256']
 assert a['dimensions']==c['changes']=={'action':[3,8],'state_robot0_joint_pos':[0,5]}
 assert set(a['train_uuids'])=={e['uuid'] for e in manifest['episodes'] if e['split']=='train'}
