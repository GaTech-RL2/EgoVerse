import os,json,time
from pathlib import Path
from collections import Counter
U=Path('/coc/flash7/cxin34/warp-data');C=Path('/coc/flash7/czhang883/warp-data')
O=U/'registry/migration-four-tasks-20261125';O.mkdir(parents=True,exist_ok=True)
entries=[]
for cat in ('collections','datasets'):
 for p in (C/cat).iterdir():
  if p.name.startswith('cxin34') and any(s in p.name for s in ('cart','basket','cloth','drawer')) and p.is_symlink():
   entries.append((str(p.resolve()),str(p),'replace_canonical_alias'))
for n in ['cxin34_aria_20261125_0556_114','cxin34_cloth_new2_60','cxin34_cloth_new2_50_20261123','cxin34_cloth_new2_60_111_170_20261125','cxin34_cloth_new2_110','cxin34_cloth_new2_170']:
 entries.append((str(U/'collections'/n),str(C/'collections'/n),'merge_or_new'))
allfiles={};report=[];errors=[]
for src,dst,mode in entries:
 seen=set();n=0;nb=0;ext=Counter();links=[]
 for base,dirs,files in os.walk(src,followlinks=True):
  # Skip failed staging outputs; retain them in original backups.
  dirs[:]=[d for d in dirs if not d.startswith('.staging') and '.attempt' not in d]
  try:
   s=os.stat(base);i=(s.st_dev,s.st_ino)
   if i in seen:dirs[:]=[];continue
   seen.add(i)
  except OSError as e:errors.append([base,str(e)]);continue
  for f in files:
   p=Path(base)/f
   try:
    s=p.stat()
    if not p.is_file():continue
    n+=1;nb+=s.st_size;ext[p.suffix]+=s.st_size;allfiles[(s.st_dev,s.st_ino)]=s.st_size
    if p.is_symlink() and len(links)<3:links.append([str(p),str(p.resolve())])
   except OSError as e:errors.append([str(p),str(e)])
 rec=dict(source=src,destination=dst,mode=mode,files=n,logical_bytes=nb,ext_bytes=dict(ext),link_examples=links)
 report.append(rec);print(json.dumps(rec),flush=True)
d=dict(entries=report,unique_bytes=sum(allfiles.values()),unique_files=len(allfiles),errors=errors)
(O/'plan-inventory.json').write_text(json.dumps(d,indent=2));print('TOTAL',json.dumps({k:v for k,v in d.items() if k!='entries'}),flush=True)
