import csv
import json
from pathlib import Path

root = Path('/coc/flash7/czhang883/warp-data/collections/cxin34_cloth_new2_165')
rows = json.loads((root / 'manifests/all165.json').read_text())['recordings']
required = {'slam': ['summary.json', 'closed_loop_trajectory.csv', 'online_calibration.jsonl', 'semidense_points.csv.gz'], 'hand_tracking': ['summary.json', 'hand_tracking_results.csv'], 'eye_gaze': ['summary.json', 'general_eye_gaze.csv']}
assert len(rows) == len({r['uuid'] for r in rows}) == 165
assert {r['take'] for r in rows} == set(range(1, 171)) - {123,132,152,157,160}
records = []
for r in rows:
    p = Path(r['mps_path'])
    assert str(p.resolve()).startswith('/coc/flash7/czhang883/warp-data/')
    assert Path(r['raw_path']).is_file()
    for feature, names in required.items():
        for name in names:
            f = p / feature / name
            assert f.is_file() and f.stat().st_size > 0, str(f)
        json.loads((p / feature / 'summary.json').read_text())
    for v in json.loads((p / 'slam/summary.json').read_text()).values():
        if isinstance(v, dict) and 'status' in v:
            assert v['status'] in ['SUCCESS', 'WARNING'] and not v.get('errors'), (r['take'], v)
    counts = {}
    for feature, name in [('slam','closed_loop_trajectory.csv'), ('hand_tracking','hand_tracking_results.csv'), ('eye_gaze','general_eye_gaze.csv')]:
        with (p / feature / name).open() as f:
            counts[feature] = sum(1 for _ in csv.DictReader(f))
        assert counts[feature] > 0
    records.append(dict(take=r['take'], rows=counts))
assert len(list((root / 'raw').glob('*.vrs'))) == 165
assert all(p.exists() for p in (root / 'raw').iterdir())
report = dict(passed=True, count=165, raw_links=165, mps_links=165, records=records, training_ready=False, atlas_playback_ready=False, note='Raw+MPS input; no retargeted Atlas/training release generated')
(root / 'provenance/migration_20261125/content-audit.json').write_text(json.dumps(report, indent=2)+'\n')
print('CONTENT AUDIT PASSED: 165 VRS, 165 MPS, required files + parsed summaries + nonempty CSV rows, no broken links')
