from pathlib import Path
import ast
R=Path('/coc/flash7/cxin34/cloth-valid-recovery-fix1124-20261124');p=R/'scripts/visual.py';s=p.read_text()
s=s.replace('import matplotlib.pyplot as plt','import matplotlib.pyplot as plt\nfrom retarget_tensor_state import StateConverter\nconv=StateConverter();offs=[np.array([.0265,.0065,-.1077,1]),np.array([-.0265,.0065,-.1077,1])]')
s=s.replace("g=f['data/demo_0'];n=len(g['obs/aria_ts']);ts=g['obs/aria_ts'][:]/1e9;fig=plt.figure(figsize=(12,8))",'''g=f['data/demo_0'];n=len(g['obs/aria_ts']);ts=g['obs/aria_ts'][:]/1e9;fig=plt.figure(figsize=(12,8))
  q=g['obs/robot0_joint_pos'][:];bp=g['obs/robot_base_pose'][:];er=g['obs/eef_right_proprio'][:].reshape(-1,4,4);el=g['obs/eef_left_proprio'][:].reshape(-1,4,4)
  def se2(i):
   T=np.eye(4);ang=bp[i,2];T[:2,:2]=[[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]];T[:2,3]=bp[i,:2];return T
  shift=np.mean([er[i,:3,3]-(se2(i)@conv.fk(q[i])[1])[:3,3]for i in np.linspace(0,n-1,20).astype(int)],0);Ts=np.eye(4);Ts[:3,3]=shift;metrics=[]''')
s=s.replace("ax=fig.add_subplot(3,4,r*4+1)","th,tr=conv.fk(q[i]);gw=np.linalg.inv(Ts@se2(i)@th@conv.mount);targets=[(gw@er[i]@offs[0])[:3],(gw@el[i]@offs[1])[:3]]\n   left=conv.d.body('left_eef');tl=np.eye(4);tl[:3,:3]=left.xmat.reshape(3,3);tl[:3,3]=left.xpos;actual=[(np.linalg.inv(th@conv.mount)@tr@offs[0])[:3],(np.linalg.inv(th@conv.mount)@tl@offs[1])[:3]];metrics.append(dict(row=int(i),fk_to_stored_tcp_m=[float(np.linalg.norm(a-b))for a,b in zip(actual,targets)]))\n   ax=fig.add_subplot(3,4,r*4+1)")
s=s.replace("else:ax.text2D(.1,.5,'NOT RELEASED',transform=ax.transAxes)","else:ax.text2D(.1,.5,'NOT RELEASED',transform=ax.transAxes)\n    if c==0:\n     for k,color in enumerate(['red','blue']):ax.scatter(*targets[k],c=color,s=30,marker='x');ax.scatter(*actual[k],c=color,s=12,marker='o')\n    else:ax.scatter(*offs[c-1][:3],c='red'if c==1 else'blue',s=30,marker='x')")
s=s.replace("fig.suptitle(f\"take", "stats={}\n  for name in ['hdn_hd2','gaze_hit']:\n   pp=e/'clouds'/(name+'_000000.npy')\n   if pp.exists():\n    aa=np.load(pp);stats[name]=dict(mean=np.mean(aa,axis=0).tolist(),median=np.median(aa,axis=0).tolist(),fraction_lt64=(aa<64).mean(axis=0).tolist()) if name=='hdn_hd2' else dict(hit_fraction=float(aa.mean()))\n  (out/(str(x['index']).zfill(2)+'.json')).write_text(json.dumps(dict(uuid=x['uuid'],take=x['take'],sampled_fk=metrics,coverage=stats),indent=2))\n  fig.suptitle(f\"take")
ast.parse(s);p.write_text(s);print('Added TCP/FK overlays and coverage reports to final visual audit')
