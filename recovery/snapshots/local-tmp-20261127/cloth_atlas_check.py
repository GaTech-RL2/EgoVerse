import json,urllib.request,urllib.parse,concurrent.futures,time
from pathlib import Path
BASE='http://100.76.236.119:8080'
D='cxin34_cloth_new2_h168@r1_hd2'
def get(p,timeout=55):
 with urllib.request.urlopen(BASE+p,timeout=timeout) as r:return r.read()
eps=json.loads(get('/api/episodes?'+urllib.parse.urlencode({'dataset':D})))
Path('/tmp/cloth-atlas-episodes.json').write_text(json.dumps(eps,indent=2))
assert len(eps)==60,len(eps)
def check(e):
 try:
  m=json.loads(get('/api/episodes/'+e['id']+'/manifest?format=10'))
  assert m['n_frames']==e['total_frames']
  assert m['point_counts']==[2048,1024,1024],m['point_counts']
  assert len(m['time_s'])==m['n_frames']
  assert all(b>a for a,b in zip(m['time_s'],m['time_s'][1:]))
  v={'id':e['id'],'ordinal':e['ordinal'],'frames':m['n_frames'],'points':m['point_counts'],'adapter':m['adapter_version'],'passed':True}
  if e['ordinal'] in [0,30,55,59]:
   Path('/tmp/cloth-atlas-manifest-'+str(e['ordinal'])+'.json').write_text(json.dumps(m))
   chunks=sorted(set([0,m['n_frames']//2//m['chunk_size'],(m['n_frames']-1)//m['chunk_size']]))
   v['chunk_bytes']=[len(get('/api/episodes/'+e['id']+'/chunks/'+str(i)+'?v='+str(m['adapter_version'])+'-'+m['source_fingerprint'])) for i in chunks]
   assert min(v['chunk_bytes'])>1000
  return v
 except Exception as ex:return {'id':e['id'],'ordinal':e['ordinal'],'passed':False,'error':str(ex)}
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 for r in ex.map(check,eps):
  results.append(r);print(json.dumps(r),flush=True)
  Path('/tmp/cloth-atlas-api-check.json').write_text(json.dumps({'dataset':D,'results':results,'passed':len(results)==60 and all(x['passed'] for x in results)},indent=2))
