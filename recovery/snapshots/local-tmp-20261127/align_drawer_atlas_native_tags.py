from pathlib import Path
import json,h5py,sys
W=Path('/coc/flash7/scratch/cxin34/warp-data');N='cxin34_drawer_1125_qc';P=W/'pipelines/drawer_atlas_20261125';v=sys.argv[1]
ret=W/'collections'/N/'processed'/v/'retargeted';d=W/'datasets'/N/'releases'/v
srcmap=d/'source_map.jsonl';maps=[json.loads(s) for s in srcmap.read_text().splitlines()];meta=json.loads((ret/'metadata.json').read_text());bundle=ret/(N+'.hdf5')
backup=P/(v+'-uuid-source-map.jsonl');assert not backup.exists();backup.write_text(srcmap.read_text())
(P/(v+'-uuid-metadata.json')).write_text(json.dumps(meta,indent=2));(P/(v+'-uuid-tag-manifest.tsv')).write_text((ret/'tag_manifest.tsv').read_text())
tags=[];changes=[]
with h5py.File(bundle,'r+') as out:
 for m in maps:
  with h5py.File(m['source_hdf5']) as f:tag=str(f['data/demo_0'].attrs['source_tag'])
  old=m['source_tag'];assert tag=='drawer_'+old.replace('-','')
  for suffix in ['_aria60.npz','_rgb224.npz']:
   (ret/'work'/(tag+suffix)).symlink_to(old+suffix)
  # Wrapper attributes only: data links continue to immutable source files.
  out['data/'+m['demo']].attrs['source_tag']=tag
  m['source_tag']=tag;m['segment_uuid']=old
  md=next(x for x in meta['hdf5_files'][bundle.name]['demos'] if x['demo']==m['demo']);md['source_tag']=tag
  tags.append('\t'.join([tag,N,m['raw_vrs'],'ok']));changes.append(dict(uuid=old,source_tag=tag))
srcmap.write_text(''.join(json.dumps(x)+'\n' for x in maps));(ret/'metadata.json').write_text(json.dumps(meta,indent=2));(ret/'tag_manifest.tsv').write_text('\n'.join(tags)+'\n')
(P/(v+'-native-tags.json')).write_text(json.dumps(dict(passed=True,changes=changes,reason='Consistent original source_tag in original HDF, wrapper HDF, metadata, source_map, tag manifest and tracking/RGB caches',source_files_unchanged=True),indent=2))
print(v,'all native tags aligned',len(maps))
