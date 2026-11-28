import os,json,itertools
from pathlib import Path
from collections import Counter,defaultdict
roots={u:Path('/coc/flash7')/u/'warp-data' for u in ('cxin34','czhang883')}
O=roots['cxin34']/'registry/audits/four-task-overlap-20261125';O.mkdir(parents=True,exist_ok=True)
def task(n):
 if n.startswith('egoa_'):return None
 if 'cloth' in n:return 'cloth'
 if 'drawer' in n or '0556_114' in n:return 'drawer'
 if 'basket' in n:return 'basket'
 if 'cart' in n:return 'cart'
 return None
records=[];inventory=[];errors=[]
for who,root in roots.items():
 for cat in ('collections','datasets'):
  for p in sorted((root/cat).iterdir()):
   t=task(p.name)
   if not t:continue
   if not p.is_dir():continue
   entry=dict(owner_tree=who,task=t,category=cat,name=p.name,path=str(p),resolved=str(p.resolve()),symlink=p.is_symlink(),raw_vrs=0)
   dirs=(p/'processed') if cat=='collections' else (p/'releases')
   entry['versions']=[dict(name=q.name,resolved=str(q.resolve()),symlink=q.is_symlink()) for q in sorted(dirs.iterdir())] if dirs.is_dir() else []
   if cat=='collections' and (p/'raw').is_dir():
    seen=set()
    def walk(d,depth=0):
     try:
      st=d.stat();ident=(st.st_dev,st.st_ino)
      if ident in seen:return
      seen.add(ident)
      for q in d.iterdir():
       if q.name.lower().endswith('.vrs'):
        try:
         s=q.stat();records.append(dict(tree=who,task=t,collection=p.name,path=str(q),resolved=str(q.resolve()),name=q.name,bytes=s.st_size,dev=s.st_dev,inode=s.st_ino,symlink=q.is_symlink()));entry['raw_vrs']+=1
        except OSError as e:errors.append([str(q),str(e)])
       elif q.is_dir() and depth<3 and not any(z in q.name.lower() for z in ('mps','depth','rgb','cache')):walk(q,depth+1)
     except OSError as e:errors.append([str(d),str(e)])
    walk(p/'raw')
   inventory.append(entry)
summary={}
for t in ('cart','basket','cloth','drawer'):
 by={w:[r for r in records if r['tree']==w and r['task']==t] for w in roots}
 ids={w:{(r['dev'],r['inode']) for r in rs} for w,rs in by.items()}
 named={w:{(r['name'],r['bytes']) for r in rs} for w,rs in by.items()}
 summary[t]=dict(raw_path_counts={w:len(v) for w,v in by.items()},unique_inodes={w:len(v) for w,v in ids.items()},same_physical_files_cross_trees=len(ids['cxin34']&ids['czhang883']),same_name_size_cross_trees=len(named['cxin34']&named['czhang883']))
pairs=[]
cols=defaultdict(list)
for r in records:cols[(r['tree'],r['collection'])].append(r)
for (ka,a),(kb,b) in itertools.combinations(cols.items(),2):
 if ka[0]==kb[0]:continue
 ia={(r['dev'],r['inode']) for r in a};ib={(r['dev'],r['inode']) for r in b};na={(r['name'],r['bytes']) for r in a};nb={(r['name'],r['bytes']) for r in b}
 same=ia&ib;name=na&nb
 if same or name:pairs.append(dict(a=list(ka),b=list(kb),same_inode=len(same),same_name_size=len(name)))
result=dict(scope='task-name-matched collections and datasets; EgoVerse fold_clothes excluded; raw walk depth<=3 skipping MPS/depth/RGB/cache',summary=summary,inventory=inventory,raw_records=records,cross_collection_pairs=pairs,errors=errors)
(O/'inventory.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(summary=summary,errors=errors,artifact=str(O/'inventory.json')),indent=2))
print('COLLECTIONS')
for e in inventory:
 if e['category']=='collections':print(e['owner_tree'],e['task'],e['name'],e['raw_vrs'],'ALIAS' if e['symlink'] else '')
print('OVERLAP_PAIRS')
for p in pairs:print(json.dumps(p))
