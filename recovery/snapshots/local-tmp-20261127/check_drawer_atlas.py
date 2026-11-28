import json,urllib.request,urllib.parse,sys,pathlib,concurrent.futures
B='http://100.76.236.119:8080';V=sys.argv[1];N='cxin34_drawer_1125_qc@'+V;P=pathlib.Path('/tmp/drawer-atlas-check');P.mkdir(exist_ok=True)
def read(path):
 with urllib.request.urlopen(B+path,timeout=55) as f:return f.read()
cat=json.loads(read('/api/datasets'));entry=next((e for e in cat if e['id']==N),None)
print('CATALOG',json.dumps(entry),flush=True)
if entry is None:sys.exit(2)
assert entry['catalog_state']=='ready';eps=json.loads(read('/api/episodes?dataset='+urllib.parse.quote(N)));(P/(V+'-episodes.json')).write_text(json.dumps(eps,indent=2));print('FIRST',json.dumps(eps[0]),flush=True)
assert len(eps)==(23 if V.startswith(('r3','r4')) else 145)
indices=[0,1,len(eps)-1] if V.startswith(('r3','r4')) else [7,63,125]
def check(i):
 e=eps[i];m=json.loads(read('/api/episodes/'+e['id']+'/manifest?format=10'));assert m['n_frames']==e['total_frames'];assert all(b>a for a,b in zip(m['time_s'],m['time_s'][1:]))
 if V.startswith(('r3','r4')):assert m['point_counts']==[2048,1024,1024]
 chunks=[]
 for k in sorted({0,(m['n_frames']//2)//m['chunk_size'],(m['n_frames']-1)//m['chunk_size']}):
  data=read('/api/episodes/'+e['id']+'/chunks/'+str(k)+'?v='+str(m['adapter_version'])+'-'+m['source_fingerprint']);assert len(data)>1000;chunks.append(dict(index=k,bytes=len(data)))
 r=dict(id=e['id'],ordinal=i,frames=m['n_frames'],point_counts=m['point_counts'],chunks=chunks);print('PASS',json.dumps(r),flush=True);return r
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=list(ex.map(check,indices))
(P/(V+'-api-check.json')).write_text(json.dumps(dict(passed=True,catalog=entry,samples=results),indent=2))
