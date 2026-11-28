from pathlib import Path
import h5py,json,hashlib,sys,numpy as np
W=Path('/coc/flash7/scratch/cxin34/warp-data');N='cxin34_drawer_1125_qc';P=W/'pipelines/drawer_atlas_20261125';v=sys.argv[1]
ret=W/'collections'/N/'processed'/v/'retargeted';bundle=ret/(N+'.hdf5');d=W/'datasets'/N/'releases'/v
maps=[json.loads(s) for s in (d/'source_map.jsonl').read_text().splitlines()];audit=json.loads((P/(v+'-audit.json')).read_text());assert len(maps)==len(audit['episodes'])
backup=P/(v+'-source-links-v1.hdf5');tmp=ret/(N+'.hdf5.tmp');assert not backup.exists() and not tmp.exists()
with h5py.File(tmp,'x') as out:
 for m in maps:
  source=m['source_hdf5'];name='data/'+m['demo'];g=out.create_group(name)
  with h5py.File(source) as src:
   original=src['data/demo_0']
   for k,value in original.attrs.items():g.attrs[k]=value
   g.attrs.update(source_tag=m['source_tag'],recording_id=m['recording_id'],num_samples=m['length'])
   def visit(k,obj):
    if isinstance(obj,h5py.Group):g.require_group(k)
    else:g[k]=h5py.ExternalLink(source,'/data/demo_0/'+k)
   original.visititems(visit)
with h5py.File(tmp) as f:
 for m in maps:
  with h5py.File(m['source_hdf5']) as src:
   g=f['data/'+m['demo']];assert g.attrs['source_tag']==m['source_tag']
   for k in ['obs/robot0_joint_pos','actions/joint','obs/aria_ts']:np.testing.assert_array_equal(g[k][:],src['data/demo_0/'+k][:])
bundle.rename(backup);tmp.rename(bundle)
(P/(v+'-tag-compatibility.json')).write_text(json.dumps(dict(passed=True,episodes=len(maps),backup=str(backup),change='Own wrapper group source_tag now matches source_map/tag_manifest; all data still read-only external links to immutable originals',training_authorized=False),indent=2))
print(v,'tags corrected',len(maps))
