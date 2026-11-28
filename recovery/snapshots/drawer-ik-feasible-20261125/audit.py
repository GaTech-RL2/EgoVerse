"""Independent final-joint diagnostics for the two offline pilots; no READY writer."""
import json, hashlib, os
from pathlib import Path
import numpy as np
import mujoco

R = Path(os.environ.get('PILOT_ROOT', '/coc/flash7/scratch/cxin34/drawer-ik-feasible-20261125'))
C = Path('/coc/flash7/czhang883/warp-data/pipelines/drawer_base_check_1125')
model = mujoco.MjModel.from_xml_path('/coc/flash7/czhang883/Documents/SEW-aria-pipeline/projects/xhand_teleop/rby1_with_xhand/model_v1.3_xhand_act.xml')


def summary(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return dict(zip(['median', 'p95', 'max'], map(float, np.percentile(x, [50, 95, 100])))) if len(x) else None


result = {'training_ready': False, 'scope': 'offline feasibility patch only', 'segments': []}
for line in (R/'inputs.tsv').read_text().splitlines():
    index, tag, work = line.split('\t')
    entry = {'index': int(index), 'tag': tag, 'variants': {}}
    for mode in ['off', 'bs_head']:
        path = R/'out'/mode/(tag+'.npz')
        with np.load(path) as f:
            a = {k: f[k] for k in f.files}
        ev = json.loads((R/'reports'/(mode+'_'+tag+'.json')).read_text())['events']
        assert ev
        n = len(a['right'])
        valid = np.ones(n, bool)
        seen = np.zeros(n, int)
        for e in ev:
            i = e['frame']
            if i is not None and 0 <= i < n:
                seen[i] += 1
                valid[i] &= e['valid']
        valid &= seen >= 2
        violations = {}
        for group, prefix, size in [('right', 'right_arm', 7), ('left', 'left_arm', 7), ('torso', 'torso', 6)]:
            limits = np.array([model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'{prefix}_{i}')] for i in range(size)])
            q = a[group]
            bad = (~np.isfinite(q)).any(1) | (q < limits[:, 0]-1e-6).any(1) | (q > limits[:, 1]+1e-6).any(1)
            valid &= ~bad
            violations[group] = int(bad.sum())
        err = np.maximum(a['errR'], a['errL'])
        valid &= np.isfinite(err) & (err <= .02)
        item = {'frames': n, 'invalid_ik_calls': sum(not e['valid'] for e in ev),
                'unique_invalid_frames': len({e['frame'] for e in ev if not e['valid']}),
                'final_joint_limit_violation_frames': violations,
                'final_smoothed_FK_target_error_cm': summary(err*100),
                'diagnostic_valid_frames_at_2cm': int(valid.sum()),
                'max_arm_step_deg': float(np.degrees(max(np.abs(np.diff(a['right'], axis=0)).max(), np.abs(np.diff(a['left'], axis=0)).max()))),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        oldpath = C/'out2'/mode/(tag+'.npz')
        if oldpath.exists():
            with np.load(oldpath) as b:
                item['old_final_smoothed_FK_target_error_cm'] = summary(np.maximum(b['errR'], b['errL'])*100)
                m = min(n, len(b['right']))
                item['changed_arm_rows'] = int((np.maximum(np.max(np.abs(a['right'][:m]-b['right'][:m]),1), np.max(np.abs(a['left'][:m]-b['left'][:m]),1)) > 1e-8).sum())
        np.save(R/'reports'/(mode+'_'+tag+'_diagnostic_valid.npy'), valid)
        entry['variants'][mode] = item
    result['segments'].append(entry)
(R/'reports/pilot-audit.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
