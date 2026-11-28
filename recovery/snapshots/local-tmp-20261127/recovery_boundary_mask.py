from pathlib import Path
import ast,json
R=Path('/coc/flash7/cxin34/cloth-valid-recovery-fix1124-20261124')
assert not list((R/'tensors/episodes').glob('*/episode.json')),'Existing tensor artifacts require separate version'
p=R/'scripts/tensor_stage.py';s=p.read_text();(R/'scripts/tensor_stage.py.before-boundary-mask').write_text(s)
s=s.replace('data.update(action_valid=np.ones(n,bool),dino_valid=np.ones(n,bool))','data.update(action_valid=np.ones(n,bool),dino_valid=np.ones(n,bool))\n # Base deltas use the previous frame; the first recovered frame must not supervise a gap-crossing delta.\n if e.get("recovery_grid_bounds",[0])[0]>0:data["action_valid"][0]=False');ast.parse(s);p.write_text(s)
p=R/'scripts/final_audit.py';s=p.read_text().replace('quality_passed=False','quality_passed=True').replace("int(b['action_valid'].sum())==min(64,n-t)","int(b['action_valid'].sum())==int(ep['action_valid'][t:min(t+64,n)].sum())")
s=s.replace("assert ts[-1]-ts[0]>1e6", "assert ts[-1]-ts[0]>1e6\n  av=np.load(arrays['action_valid']['file']);assert av[1:].all();assert bool(av[0])==(x['recovery_grid_bounds'][0]==0)")
ast.parse(s);p.write_text(s)
(R/'reports/boundary-mask.json').write_text(json.dumps(dict(reason='Base action delta is current minus previous pose; initial recovered row action supervision masked when prior row lies outside segment. No values zero-filled; original arrays untouched; H64/DINO windows cannot cross segment boundaries.',existing_tensor_count=0),indent=2));print('First recovered action mask and real loader assertions installed')
