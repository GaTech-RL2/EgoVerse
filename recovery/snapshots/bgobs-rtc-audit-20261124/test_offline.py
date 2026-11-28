import ast
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

R = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rtc', R/'rtc_client.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m._selftest()

def filter_from_source(name, chunk, rtc):
    tree = ast.parse((R/name).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
        and ast.unparse(n.test) == 'args.chunk_filter > 0')
    env = dict(chunk=chunk.copy(), rtc=rtc, args=SimpleNamespace(chunk_filter=1), smooth_chunk=m.smooth_chunk)
    exec(compile(ast.Module(body=node.body, type_ignores=[]), name, 'exec'), env)
    return env['chunk']

rng = np.random.default_rng(42)
old = m.smooth_chunk(rng.normal(size=(64,49)))
next_chunk = rng.normal(size=(64,49)); next_chunk[:12] = old[13:25]
rtc = m.RTCController(freeze_steps=12, period=13, soft_horizon=21)
rtc.on_chunk(old,0)
for i in range(13):
    assert not rtc.should_submit(i)
    rtc.on_execute()
assert rtc.should_submit(13)
payload = {}; rtc.attach(payload,13)
assert payload['rtc_prev_actions'].shape == (51,49)
assert np.allclose(payload['rtc_prev_actions'],old[13:])
assert not rtc.should_submit(30)
before = filter_from_source('rollout_rby1_policy_real_bgobs.py',next_chunk,rtc)
after = filter_from_source('rollout_rby1_policy_real_bgobs_fixed.py',next_chunk,rtc)
before_error = float(abs(before[:12]-next_chunk[:12]).max())
assert before_error > 0 and np.array_equal(after[:12],next_chunk[:12])
cases=[]
for delta in (0,1,5,11,12,13,20):
    c=m.RTCController(12,13,21); c.on_chunk(old,0)
    for _ in range(13): c.on_execute()
    c.attach({},13)
    for _ in range(delta): c.on_execute()
    rows=c.on_chunk(after)
    assert np.array_equal(rows,after[delta:]) and c.base_idx==delta
    assert c.n_late == int(delta>12)
    if delta<12: assert np.array_equal(rows[0],old[13+delta])
    cases.append(dict(delta=delta,remaining=len(rows),late=bool(c.n_late)))

# Execute the actual nested worker definition without importing the hardware client.
tree=ast.parse((R/'rollout_rby1_policy_real_bgobs_fixed.py').read_text())
worker=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_bg_worker')
lh=np.arange(12.);rh=lh+20
received={}
def build(*a):
    received['hands']=a[-2]
    received['sensors']=a[-1]
    return {},None
env=dict(np=np,step=13,_bg_pre_state=None,_bg_pre_odom=None,_bg_rtc_keys={},
    _bg_sensors=('pair_at_submit','rgb_at_submit',123.0),
    last_hand_left=lh,last_hand_right=rh,time=time,_bg_build_obs=build,
    args=SimpleNamespace(viz_num_samples=1),client=SimpleNamespace(infer=lambda p: {'actions':after}))
exec(compile(ast.Module(body=[worker],type_ignores=[]),'actual_worker','exec'),env)
lh[:]=999;rh[:]=888
env['_bg_sensors']=('later_pair','later_rgb',999.)
env['_bg_worker']()
assert np.array_equal(received['hands'][0],np.arange(12.))
assert np.array_equal(received['hands'][1],np.arange(12.)+20)
assert received['sensors']==('pair_at_submit','rgb_at_submit',123.0)
report=dict(passed=True,rtc=[12,13,21],old_filter_frozen_max_error=before_error,
    fixed_filter_frozen_max_error=0.,splice_cases=cases,worker_hand_snapshot_passed=True,
    scope='actual pure RTC and extracted worker/filter; no robot/camera imports')
(R/'offline-results.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
