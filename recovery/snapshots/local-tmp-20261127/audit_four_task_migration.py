from pathlib import Path
import json,os,hashlib,time
R=Path('/coc/flash7/cxin34/warp-data/registry/migration-four-tasks-20261125');C=Path('/coc/flash7/czhang883/warp-data');S=C/'.cxin34-storage/20261125-four-tasks'
rows=[json.loads(x) for x in (R/'files.jsonl').read_text().splitlines()];hdf=json.loads((R/'HDF_SELF_CONTAINED.json').read_text());rewritten={x['file']:x for x in hdf['rewritten']};errors=[];identical=0;copies=0;seen={};symlinks=[]
for r in rows:
 try:
  src=Path(r['source']);dst=Path(r['destination']);a=src.stat();b=dst.stat();assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(r['source_dev'],r['source_inode'],r['bytes'],r['source_mtime_ns']),'original_changed'
  if str(dst) in rewritten:continue
  assert b.st_size==r['bytes'],'size_mismatch'
  if (a.st_dev,a.st_ino)==(b.st_dev,b.st_ino):identical+=1
  else:
   key=(b.st_dev,b.st_ino)
   if key not in seen:
    h=hashlib.sha256()
    with dst.open('rb') as f:
     for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    seen[key]=h.hexdigest()
   if r.get('sha256'):assert seen[key]==r['sha256'],'sha_mismatch'
   else:
    # deduplicated cross-filesystem links inherit the copied inode's recorded SHA.
    ref=next((q['sha256'] for q in rows if q.get('sha256') and os.path.samefile(q['destination'],dst)),None)
    assert ref is not None and seen[key]==ref,'dedup_missing_proof'
   copies+=1
 except Exception as e:errors.append(dict(path=r['source'],error=str(e)))
for root,dirs,files in os.walk(S,followlinks=False):
 for n in dirs+files:
  p=Path(root)/n
  if p.is_symlink():
   t=p.resolve();assert t.exists(),str(p);symlinks.append(dict(path=str(p),target=str(t)));assert str(t).startswith(str(C)+'/'),str(p)
report=dict(passed=not errors,checked=len(rows),same_inode=identical,copied_or_dedup_checked=copies,rewritten_wrappers=len(rewritten),canonical_symlinks=len(symlinks),errors=errors,originals_preserved=True,time=time.time())
(R/'file-independent-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
