import json,urllib.request,urllib.parse,urllib.error,concurrent.futures,pathlib,time
B='http://100.76.236.119:8080';out=pathlib.Path('/tmp/atlas-migration-verification.json')
def read(p):
 for attempt in range(8):
  try:
   with urllib.request.urlopen(B+p,timeout=55) as f:return f.read()
  except urllib.error.HTTPError as ex:
   if ex.code!=503 or attempt==7:raise
   time.sleep(2)
cat=json.loads(read('/api/datasets'));ids=['cxin34_pretrain_cart_retarget@r2_retarget_pcd','cxin34_pretrain_basket_retarget@r2_retarget_pcd','cxin34_cloth_new2_h168@r1_hd2','cxin34_cloth_fix1124@r1_hd2_fix1124','cxin34_drawer_1125_qc@r4_fix_bs_display23','cxin34_cloth_new50_fix1124_bs@r1_hd2'];results=[]
def test(n):
 e=next(x for x in cat if x['id']==n);eps=json.loads(read('/api/episodes?dataset='+urllib.parse.quote(n)));r=dict(dataset=n,catalog=e,episodes=len(eps),samples=[]);assert eps,(n,e.get('catalog_message'))
 for i in sorted(set([0,len(eps)//2,len(eps)-1])):
  a=eps[i];m=json.loads(read('/api/episodes/'+a['id']+'/manifest?format=10'));assert m['n_frames']==a['total_frames'];assert all(b>a for a,b in zip(m['time_s'],m['time_s'][1:]));assert m['point_counts']==[2048,1024,1024],m['point_counts'];chunks=[]
  for k in sorted(set([0,m['n_frames']//2//m['chunk_size'],(m['n_frames']-1)//m['chunk_size']])):
   x=read('/api/episodes/'+a['id']+'/chunks/'+str(k)+'?v='+str(m['adapter_version'])+'-'+m['source_fingerprint']);assert len(x)>1000;chunks.append(dict(index=k,bytes=len(x)))
  r['samples'].append(dict(id=a['id'],ordinal=i,frames=m['n_frames'],point_counts=m['point_counts'],chunks=chunks))
 r['passed']=True;return r
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 fs={ex.submit(test,n):n for n in ids}
 for f in concurrent.futures.as_completed(fs):
  try:r=f.result()
  except Exception as e:r=dict(dataset=fs[f],passed=False,error=str(e))
  results.append(r);out.write_text(json.dumps(dict(time=time.time(),results=results),indent=2));print(r['dataset'],r['passed'],r.get('error',''),flush=True)
