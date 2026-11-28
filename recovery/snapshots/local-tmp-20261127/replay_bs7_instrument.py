import os,sys,importlib.util,inspect,json
from pathlib import Path
import numpy as np
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125')
P=R/'drawer/segments/a3dfa5f8-c243-5670-bf5f-9f87c73b74ec'
O=R/'reports/bs-example7-replay-20261125'
O.mkdir(exist_ok=False)
os.environ.update(SEW_REPO='/coc/flash7/czhang883/Documents/SEW-aria-pipeline',ARIA_ARTIFACTS=str(P/'retarget/work'),BASEFIX_LOG=str(O/'basefix.jsonl'))
p=R/'drawer/source/cpu/bs/build_hdf5_basefix.py'
spec=importlib.util.spec_from_file_location('trace_bs',p);M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
import projects.rby1_teleop.geometric_kinematics_rby1 as G
orig=G.SEW_IK_SP_2_1_1_Euler_3
src=inspect.getsource(orig)
src=src.replace('    # === STAGE 2:', '    _trace = {}\n    # === STAGE 2:')
for j in range(1,8):
    needle=f'q{j} = np.clip(q{j}, q_lower[{j-1}], q_upper[{j-1}])'
    src=src.replace(needle,f'_trace["q{j}_pre"] = float(q{j}); '+needle)
# capture after every bounded solve, without changing arithmetic
lines=src.splitlines();idx=next(i for i,l in enumerate(lines) if l.strip().startswith('return '))
lines.insert(idx,'    _trace.update(Q=np.asarray(Q).tolist(), prev=None if q_prev is None else np.asarray(q_prev).tolist(), limited=bool(limited), LS=np.asarray(is_LS_vec).tolist()); TRACE.append(_trace)')
G.__dict__['TRACE']=[]
exec('\n'.join(lines),G.__dict__)
sys.argv=[str(p),'--no_firm','--res','8','--tags','drawer_a3dfa5f8c2435670bf5f9f87c73b74ec','--out',str(O/'diagnostic_bs.hdf5')]
try:M.B.main()
finally:(O/'ik-trace.json').write_text(json.dumps(G.TRACE))
print('TRACE_CALLS',len(G.TRACE),flush=True)
