"""Audit recorded native RGB frames without rerunning a policy or simulator."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

from efe_pear_bridge import image_blocks,observation_payload
from efe_pear_observation_check import validate_frames


def audit(item):
    try:
        root=Path(item['native_trial_path']).parents[3];frames={};sources={}
        for index in [1,2]:
            path=root/f'observation-{index:04d}.json';raw=path.read_bytes();body=json.loads(raw)
            visible,content=observation_payload(body)
            if visible['action_contract']!=item['action_contract']:
                raise ValueError('Recorded action contract differs from its native qualification')
            frames[str(index)]=validate_frames([b['image']for b in image_blocks(content)if b['type']=='image'],
                                               item['action_contract']['cameras'])
            sources[str(index)]=hashlib.sha256(raw).hexdigest()
        return item['task'],{'passed':True,'scene_sha256':item['scene_sha256'],
            'native_trial_path':item['native_trial_path'],'observation_sha256':sources,'camera_evidence':frames}
    except Exception as exc:return item['task'],{'passed':False,'error':repr(exc)}


def main():
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    tasks={};unavailable=[]
    for name in spec['check_files']:
        path=Path(name)
        if not path.exists():unavailable.append(name);continue
        for attempt in range(3):
            try:checks=json.loads(path.read_text());break
            except json.JSONDecodeError:
                if attempt==2:raise
                time.sleep(.2)
        for key,item in checks.items():
            if item.get('success'):
                if item.get('qualification')!='passed':raise ValueError('Successful source check is not admitted')
                tasks[key]=item
    with ThreadPoolExecutor(max_workers=4)as pool:results=dict(pool.map(audit,tasks.values()))
    value={'scope':'Recorded native reset and post-action public RGB evidence; no policy inference or optimizer update.',
        'primary_rl_updates':0,'checked':len(results),'passed':sum(v['passed']for v in results.values()),
        'unavailable_check_files':unavailable,'tasks':results}
    output=Path(spec['output']);temporary=output.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(output)
    print('PEAR_OBSERVATION_AUDIT '+json.dumps({k:v for k,v in value.items()if k!='tasks'}),flush=True)


if __name__=='__main__':main()
