import os,json,time,hashlib,shutil,fcntl,traceback,stat
from pathlib import Path
U=Path('/coc/flash7/cxin34/warp-data');C=Path('/coc/flash7/czhang883/warp-data');R=U/'registry/migration-four-tasks-20261125';STORE=C/'.cxin34-storage/20261125-four-tasks'
lock=(R/'migration.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
plan=json.loads((R/'plan-inventory.json').read_text())
if (R/'MATERIALIZED.json').exists():print('Already materialized; no duplicate work');raise SystemExit(0)
state={'status':'MATERIALIZING','started':time.time(),'files':0,'hardlinks':0,'copied':0,'copied_bytes':0,'done':[]}
dirs={};files={};visiting=set();fout=(R/'files.jsonl').open('w');dout=(R/'dirs.jsonl').open('w')
def save():
 p=R/'status.tmp';p.write_text(json.dumps(state,indent=2));os.replace(p,R/'status.json')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def linkto(dst,target):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.is_symlink():assert dst.resolve()==Path(target).resolve();return
 assert not dst.exists(),str(dst)
 dst.symlink_to(target)
def clone(src,dst):
 s=src.stat();key=(s.st_dev,s.st_ino)
 if stat.S_ISDIR(s.st_mode):
  if key in visiting:raise RuntimeError('directory cycle '+str(src))
  if key in dirs:
   if dst!=dirs[key]:linkto(dst,dirs[key])
   return
  dirs[key]=dst;visiting.add(key);dst.mkdir(parents=True,exist_ok=True)
  dout.write(json.dumps({'source':str(src),'real_source':str(src.resolve()),'destination':str(dst)})+'\n')
  for x in sorted(src.iterdir()):
   if x.name.startswith('.staging') or '.attempt' in x.name:continue
   clone(x,dst/x.name)
  visiting.remove(key);return
 if not stat.S_ISREG(s.st_mode):raise RuntimeError('unexpected nonregular '+str(src))
 dst.parent.mkdir(parents=True,exist_ok=True)
 record=dict(source=str(src),real_source=str(src.resolve()),destination=str(dst),bytes=s.st_size,source_dev=s.st_dev,source_inode=s.st_ino,source_mtime_ns=s.st_mtime_ns)
 if not dst.exists():
  try:
   os.link(files.get(key,src.resolve()),dst)
   method='same_inode' if key not in files else 'deduplicated_link'
  except OSError:
   if shutil.disk_usage(C).free < s.st_size+150*1024**3:raise RuntimeError('150GiB capacity floor')
   tmp=dst.with_name(dst.name+'.migration-partial');h=hashlib.sha256()
   with src.open('rb') as fi,tmp.open('wb') as fo:
    for b in iter(lambda:fi.read(8*1024*1024),b''):h.update(b);fo.write(b)
    fo.flush();os.fsync(fo.fileno())
   sha=h.hexdigest();assert digest(tmp)==sha
   shutil.copystat(src,tmp);os.replace(tmp,dst);record['sha256']=sha;method='sha256_copy';state['copied']+=1;state['copied_bytes']+=s.st_size
 else:
  if os.path.samefile(src,dst):method='same_inode'
  elif key in files and os.path.samefile(files[key],dst):method='deduplicated_link'
  else:
   a=digest(src);assert digest(dst)==a;record['sha256']=a;method='sha256_existing'
 after=src.stat();assert (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)==(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns),'source changed '+str(src)
 ds=dst.stat();assert ds.st_size==s.st_size
 if method=='same_inode':assert (ds.st_dev,ds.st_ino)==key;state['hardlinks']+=1
 record['verification']=method;files.setdefault(key,dst);fout.write(json.dumps(record)+'\n');state['files']+=1
 if state['files']%1000==0:fout.flush();save();print(json.dumps(state),flush=True)
try:
 save()
 # Raw-only Drawer first helps all later references deduplicate into one canonical payload.
 entries=sorted(plan['entries'],key=lambda e:0 if 'cxin34_aria_20261125' in e['source'] else 1)
 for e in entries:
  src=Path(e['source']);dest=Path(e['destination']);relative=dest.relative_to(C);stage=STORE/relative
  state['current']=str(src);save();clone(src,stage)
  state['done'].append(dict(source=str(src),canonical=str(dest),payload=str(stage),mode=e['mode']));save();fout.flush();dout.flush()
 state['status']='MATERIALIZED_NOT_PUBLISHED';save();fout.close();dout.close();(R/'MATERIALIZED.json').write_text(json.dumps(state,indent=2));print('MATERIALIZED',json.dumps(state),flush=True)
except Exception:
 state['status']='BLOCKED';state['error']=traceback.format_exc();save();raise
