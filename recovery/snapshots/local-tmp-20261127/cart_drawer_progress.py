import pathlib,json,subprocess,datetime,collections,shutil
R=pathlib.Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');S='/opt/slurm/Ubuntu-24.04/current/bin/'
now=datetime.datetime.now(datetime.timezone.utc);out=dict(time=now.isoformat(),formal={})
for task in ['cart','drawer']:
 for stage in ['finetune','scratch','lora','novfilter']:
  w=R/'training'/task/stage
  for p in w.glob('runs/*/metrics.jsonl'):
   name=p.parent.name
   if name.startswith('egoa-')or name.endswith('-smoke'):continue
   ls=p.read_text().splitlines()
   if ls:out['formal'][task+'/'+stage]=dict(latest=json.loads(ls[-1]),ckpt10k=(p.parent/'step_010000.pt').exists(),path=str(p.parent))
ls=(R/'logs/cart-novfilter-export-4871888.log').read_text().splitlines();out['novfilter_export_count']=sum(x.startswith('NOVFILTER_EXPORT')for x in ls);out['novfilter_tail']=ls[-3:]
q=[json.load(open(p))for p in R.glob('drawer/segments/*/retarget-proof-v2.json')];out['drawer_qc']=dict(terminal=len(q),passed=sum(x['passed']for x in q),pass_indices=sorted(x['index']for x in q if x['passed']))
for f in ['drawer-controller-live.json','drawer-downstream-dispatch-live.json']:
 p=R/'reports'/f
 if p.exists():out[f]=json.load(open(p))
out['queue']=subprocess.check_output([S+'squeue','-u','cxin34','-h','-o','%i|%j|%T|%M|%N|%R'],text=True).splitlines()
out['scratch_free_GiB']=round(shutil.disk_usage(R).free/2**30,2)
p=R/'reports'/('progress-'+now.strftime('%Y%m%dT%H%M%SZ')+'.json');p.write_text(json.dumps(out,indent=2))
compact=dict(time=out['time'],formal=out['formal'],novfilter_export=out['novfilter_export_count'],drawer_qc=out['drawer_qc'],controller=out.get('drawer-controller-live.json'),scratch_free_GiB=out['scratch_free_GiB'],snapshot=str(p),queue=[x for x in out['queue']if any(t in x for t in ['drawer','cart-novfilter'])])
print(json.dumps(compact),flush=True)
