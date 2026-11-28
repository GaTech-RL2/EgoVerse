import os,json,time,hashlib,shutil,fcntl,traceback
from pathlib import Path
import h5py
R=Path('/coc/flash7/cxin34/warp-data/registry/migration-four-tasks-20261125');C=Path('/coc/flash7/czhang883/warp-data');S=C/'.cxin34-storage/20261125-four-tasks'
lock=(R/'hdf.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while not (R/'MATERIALIZED.json').exists():
 st=json.loads((R/'status.json').read_text())
 if st['status']=='BLOCKED':raise RuntimeError('materialization blocked')
 time.sleep(5)
if (R/'HDF_SELF_CONTAINED.json').exists():raise SystemExit(0)
rows=[json.loads(l) for l in (R/'files.jsonl').read_text().splitlines()]
mp={x['real_source']:Path(x['destination']) for x in rows};results=[];deps=[];visited=set();active=set()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save():
 (R/'hdf-status.json').write_text(json.dumps(dict(scanned=len(visited),rewritten=len(results),dependencies=len(deps),last=results[-1] if results else None),indent=2))
def materialize(p):
 p=p.resolve()
 if str(p).startswith(str(C)+'/'):return p
 if str(p) in mp:return mp[str(p)]
 d=S/'dependencies'/hashlib.sha256(str(p).encode()).hexdigest()[:20]/p.name;d.parent.mkdir(parents=True,exist_ok=True)
 if not d.exists():
  try:os.link(p,d);v='same_inode';h=None
  except OSError:
   assert shutil.disk_usage(C).free>p.stat().st_size+150*1024**3
   tmp=d.with_name(d.name+'.partial');shutil.copy2(p,tmp);h=sha(p);assert sha(tmp)==h;os.replace(tmp,d);v='sha256_copy'
 else:
  v='same_inode' if os.path.samefile(p,d) else 'sha256_existing';h=None if v=='same_inode' else sha(p)
  if h:assert sha(d)==h
 mp[str(p)]=d;deps.append(dict(source=str(p),destination=str(d),bytes=p.stat().st_size,verification=v,sha256=h));save();return d
def scan(p):
 p=Path(p).resolve()
 assert str(p).startswith(str(S)+'/'),str(p)
 if str(p) in visited:return
 if str(p) in active:raise RuntimeError('HDF dependency cycle '+str(p))
 active.add(str(p));links=[]
 def walk(g):
  for k in g:
   link=g.get(k,getlink=True);name=g.name.rstrip('/')+'/'+k
   if isinstance(link,h5py.ExternalLink):links.append((name,link.filename,link.path))
   elif isinstance(link,h5py.HardLink) and isinstance(g[k],h5py.Group):walk(g[k])
 with h5py.File(p,'r') as f:walk(f)
 changes=[]
 for name,fn,target in links:
  orig=Path(fn) if Path(fn).is_absolute() else p.parent/fn
  dest=materialize(orig)
  if dest.suffix in ('.hdf5','.h5') and str(dest).startswith(str(S)+'/'):scan(dest)
  if str(dest)!=fn:changes.append((name,str(dest),target))
 if changes:
  # Break original hardlink before changing wrapper metadata; original bytes are retained.
  before=sha(p);tmp=p.with_name(p.name+'.canonical-links');shutil.copy2(p,tmp)
  with h5py.File(tmp,'r+') as f:
   for name,dest,target in changes:del f[name];f[name]=h5py.ExternalLink(dest,target)
  with h5py.File(tmp,'r') as f:
   for name,dest,target in changes:
    link=f.get(name,getlink=True);assert link.filename==dest and link.path==target
   # Probe representative resolved data; file identity/SHA proves all copied source values unchanged.
   for name,dest,target in changes[::max(1,len(changes)//10)]:assert f[name] is not None
  os.replace(tmp,p);results.append(dict(file=str(p),external_links=len(changes),old_sha256=before,new_sha256=sha(p)))
 active.remove(str(p));visited.add(str(p));save()
try:
 paths=sorted({x['destination'] for x in rows if x['destination'].endswith(('.hdf5','.h5'))})
 for p in paths:scan(p)
 report=dict(passed=True,scanned=len(visited),rewritten=results,dependencies=deps,original_files_unchanged=True)
 (R/'HDF_SELF_CONTAINED.json').write_text(json.dumps(report,indent=2));print('HDF_SELF_CONTAINED',len(visited),len(results),len(deps),flush=True)
except Exception:
 (R/'hdf-error.txt').write_text(traceback.format_exc());raise
