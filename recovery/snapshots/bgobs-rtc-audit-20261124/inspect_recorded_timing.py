"""Read a bounded, already-recorded sample. Never opens a sensor or robot."""
from pathlib import Path
import json
import numpy as np
r=Path('/home/aloha/RB_Y1_workspace/rollout_logs/egoa180_cloth50_prefix_FT10k_bgobs_20261124_152642')
rows=[]
for p in sorted(r.glob('frame_*.npz'))[:15]:
    with np.load(p,allow_pickle=False) as z:
        rows.append(dict(file=p.name,step=int(z['step']),prefetch=bool(z['rtc_prefetch']),
            rgb_slam_delta_ms=float((z['raw_rgb_ts']-z['slam_ts'])*1000),
            slam_age_at_build_ms=float((z['t_snapshot']-z['slam_ts'])*1000),
            build_to_dump_ms=float((z['t_wall']-z['t_snapshot'])*1000)))
bg=[r for r in rows if r['prefetch']]
stats={k:dict(median=float(np.median([abs(r[k]) for r in bg])),
    max=float(max(abs(r[k]) for r in bg)))
    for k in ('rgb_slam_delta_ms','slam_age_at_build_ms','build_to_dump_ms')}
report=dict(source=str(r),sample_frames=len(rows),prefetch_frames=len(bg),rows=rows,stats=stats,
    caveats=['Cloth active-run sample, not Cart validation',
        'Logged sensor timestamps are not proven exposure-clock aligned',
        'Old t_snapshot is worker build start, not main-thread state sample',
        'Old t_rgb_read is after compute even when RGB was read before compute'])
out=Path(__file__).resolve().parent/'recorded-timing.json'
out.write_text(json.dumps(report,indent=2));print(json.dumps(stats))
