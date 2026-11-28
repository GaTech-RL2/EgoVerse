"""Patch an isolated generated bgobs copy; never edit the installed client."""
from pathlib import Path
import ast
import hashlib
import json

root = Path(__file__).resolve().parent
p = root / 'rollout_rby1_policy_real_bgobs.py'
s = p.read_text()
original_sha = hashlib.sha256(s.encode()).hexdigest()
def replace(old, new):
    global s
    assert s.count(old) == 1, (old, s.count(old))
    s = s.replace(old, new)

replace('import threading', 'import threading\nfrom types import SimpleNamespace')
replace('def _bg_build_obs(step, prefetch_now, _pre_state, _pre_odom, _in_bg):',
        'def _bg_build_obs(step, prefetch_now, _pre_state, _pre_odom, _in_bg, _pre_hands=None, _pre_sensors=None):')
replace('last_hand_left=last_hand_left,',
        'last_hand_left=(_pre_hands[0] if _pre_hands is not None else last_hand_left),')
replace('last_hand_right=last_hand_right,',
        'last_hand_right=(_pre_hands[1] if _pre_hands is not None else last_hand_right),')
replace('_bg_pre_state = RBY1.robot.get_state() if RBY1 is not None else None',
        '_bg_state_time = time.time()\n'
        '                                        _bg_pre_state = SimpleNamespace(position=np.array(RBY1.robot.get_state().position, copy=True)) if RBY1 is not None else None')
replace('_bg_pre_odom = RBY1.get_current_odometry() if RBY1 is not None else None',
        '_bg_pre_odom = np.array(RBY1.get_current_odometry(), copy=True) if RBY1 is not None else None')
replace('_bg_rtc_keys = {}',
        '_bg_sensors = None\n'
        '                                        if uses_live_policy_camera and percep is not None:\n'
        '                                            _bg_pair = pair_reader.wait_fresh(last_slam_seq, timeout_s=0.0, max_pair_dt_s=args.slam_pair_max_dt_ms * 1e-3)\n'
        '                                            _bg_raw = read_aria_raw_rgb(aria_streaming_shared)\n'
        '                                            _bg_pair = (np.array(_bg_pair[0], copy=True), np.array(_bg_pair[1], copy=True), *_bg_pair[2:]) if _bg_pair is not None else None\n'
        '                                            _bg_raw = (np.array(_bg_raw[0], copy=True), *_bg_raw[1:]) if _bg_raw is not None else None\n'
        '                                            _bg_sensors = (_bg_pair, _bg_raw, time.time(), _bg_state_time)\n'
        '                                        _bg_rtc_keys = {}')
replace('\n                                                pair = pair_reader.wait_fresh(', '\n                                                pair = _pre_sensors[0] if _pre_sensors is not None else pair_reader.wait_fresh(')
replace('_raw_pre = read_aria_raw_rgb(aria_streaming_shared)',
        '_raw_pre = _pre_sensors[1] if _pre_sensors is not None else read_aria_raw_rgb(aria_streaming_shared)')
replace('_raw = _raw_pre if _raw_pre is not None else read_aria_raw_rgb(aria_streaming_shared)',
        '_raw = _pre_sensors[1] if _pre_sensors is not None else (_raw_pre if _raw_pre is not None else read_aria_raw_rgb(aria_streaming_shared))')
replace('_dump_meta = {"t_snapshot": time.time()}',
        '_dump_meta = {"t_snapshot": _pre_sensors[3] if _pre_sensors is not None else time.time(), "t_obs_build": time.time()}')
replace('t_slam_read=time.time())', 't_slam_read=(_pre_sensors[2] if _pre_sensors is not None else time.time()))')
replace('t_rgb_read=time.time())',
        't_rgb_read=(_pre_sensors[2] if _pre_sensors is not None else time.time()))')
replace('def _bg_worker(_s=step, _st=_bg_pre_state, _od=_bg_pre_odom, _rk=_bg_rtc_keys):',
        'def _bg_worker(_s=step, _st=_bg_pre_state, _od=_bg_pre_odom, _rk=_bg_rtc_keys,\n'
        '                                                       _hands=(np.array(last_hand_left, copy=True), np.array(last_hand_right, copy=True)), _sensors=_bg_sensors):')
replace('_res = _bg_build_obs(_s, True, _st, _od, True)',
        '_res = _bg_build_obs(_s, True, _st, _od, True, _hands, _sensors)')
# Filter only the unconstrained region: server-frozen values already describe
# the previously filtered executing trajectory. Filtering those again breaks RTC.
replace('chunk = smooth_chunk(chunk, args.chunk_filter)',
        '_frozen = min(rtc.d, len(chunk)) if rtc is not None and rtc.pending else 0\n'
        '                                        _prefix = chunk[:_frozen].copy()\n'
        '                                        chunk = smooth_chunk(chunk, args.chunk_filter)\n'
        '                                        chunk[:_frozen] = _prefix')
ast.parse(s)
(root / 'rollout_rby1_policy_real_bgobs_fixed.py').write_text(s)
(root / 'patch-manifest.json').write_text(json.dumps(dict(original_sha256=original_sha,
    fixed_sha256=hashlib.sha256(s.encode()).hexdigest(), installed_client_modified=False), indent=2))
print('isolated patch prepared')
