import os,json,time,fcntl
from pathlib import Path
C=Path('/coc/flash7/czhang883/warp-data');R=Path('/coc/flash7/cxin34/warp-data/registry/migration-four-tasks-20261125')
lock=(R/'publish.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
s=json.loads((R/'status.json').read_text());log=R/'publication.json';done=json.loads(log.read_text()) if log.exists() else []
def switch(p,target):
 if p.is_symlink() and p.resolve()==target.resolve():return
 assert not p.exists() or p.is_symlink(),str(p)
 if p.is_symlink():assert p.lstat().st_uid==os.getuid(),str(p)
 tmp=p.with_name('.'+p.name+'.cxin34-switch')
 if tmp.is_symlink():tmp.unlink()
 tmp.symlink_to(target);os.replace(tmp,p)
for e in s['done']:
 if any(x['canonical']==e['canonical'] for x in done):continue
 p=Path(e['canonical']);target=Path(e['payload']);assert target.exists()
 if p.is_symlink() or not p.exists():
  prior=os.readlink(p) if p.is_symlink() else None
  switch(p,target);rec=dict(e,prior_link=prior,published=time.time(),kind='atomic_alias')
 else:
  assert p.stat().st_uid==os.getuid(),str(p)
  added=[]
  # Existing migrated Cloth60/50 inputs remain authoritative; add only absent entries.
  for child in target.iterdir():
   q=p/child.name
   if not q.exists() and not q.is_symlink():switch(q,child);added.append(str(q))
   elif child.is_dir() and q.is_dir() and child.name in ('processed','raw'):
    for g in child.iterdir():
     qg=q/g.name
     if not qg.exists() and not qg.is_symlink():switch(qg,g);added.append(str(qg))
  rec=dict(e,published=time.time(),kind='nonoverwriting_merge',added=added)
 done.append(rec);tmp=R/'publication.tmp';tmp.write_text(json.dumps(done,indent=2));os.replace(tmp,log)
 print('PUBLISHED',str(p),rec['kind'],flush=True)
print('TOTAL',len(done),flush=True)
