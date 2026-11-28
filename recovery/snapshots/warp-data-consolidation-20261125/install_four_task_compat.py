from pathlib import Path
import os,json,fcntl,time
U=Path('/coc/flash7/cxin34/warp-data');T=Path('/coc/flash7/scratch/cxin34/warp-data');C=Path('/coc/flash7/czhang883/warp-data');R=U/'registry/migration-four-tasks-20261125'
lock=(R/'compatibility.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert json.loads((R/'file-independent-audit.json').read_text())['passed']
if (R/'COMPATIBILITY.json').exists():print('Compatibility already complete');raise SystemExit
entries=json.loads((R/'publication.json').read_text());journal=(R/'compatibility.jsonl').open('a');counts=dict(symbolic_links=0,hardlinks_kept=0,independent_originals_kept=0,already_canonical=0);seen=set()
def link(p,d):
 if str(p) in seen:return
 seen.add(str(p));assert str(p).startswith((str(U)+'/',str(T)+'/'));assert d.exists(),str(d)
 if p.is_symlink() and str(p.resolve()).startswith(str(C)+'/'):counts['already_canonical']+=1;return
 if not p.is_symlink():
  if os.path.samefile(p,d):counts['hardlinks_kept']+=1;return
  if p.stat().st_dev==d.stat().st_dev:counts['independent_originals_kept']+=1;return
 assert p.lstat().st_uid==os.getuid(),str(p)
 root=T if str(p).startswith(str(T)+'/') else U;backup=root/'registry/migration-four-tasks-20261125/original-paths'/p.relative_to(root);backup.parent.mkdir(parents=True,exist_ok=True)
 rec=dict(old=str(p),canonical=str(d),backup=str(backup),prior_link=os.readlink(p) if p.is_symlink() else None)
 if not backup.exists() and not backup.is_symlink():
  if p.is_symlink():backup.symlink_to(p.resolve())
  else:os.link(p,backup)
 assert backup.exists();journal.write(json.dumps(dict(rec,event='INTENT'))+'\n');journal.flush()
 tmp=p.with_name('.'+p.name+'.canonical-link');assert not tmp.exists() and not tmp.is_symlink();tmp.symlink_to(d);os.replace(tmp,p);assert p.exists();counts['symbolic_links']+=1;journal.write(json.dumps(dict(rec,event='COMPLETED'))+'\n');journal.flush()
for e in entries:
 root=Path(e['source']);payload=Path(e['payload'])
 if root.is_symlink():link(root,payload);continue
 for r,dirs,files in os.walk(root,followlinks=False):
  dirs[:]=[d for d in dirs if not d.startswith('.staging') and '.attempt' not in d]
  for name in dirs+files:
   p=Path(r)/name;d=payload/p.relative_to(root)
   if d.exists() and (p.is_symlink() or p.is_file()):link(p,d)
(R/'COMPATIBILITY.json').write_text(json.dumps(dict(passed=True,counts=counts,all_replaced_originals_preserved=True,time=time.time(),method='Atomic symlink replacement with original file hardlinks retained in same-filesystem registry backup; same-filesystem data hardlinks remain valid at old paths'),indent=2));print(json.dumps(counts))
