"""Fingerprint pinned native reset states without task names or instructions.

These are reset-identity checks, not evidence that a robot adapter runs. Native
reset/action/reward checks and source-scene ancestry remain separate receipts.
"""
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def scene_identity(reference,row):
    import numpy as np
    directory=Path(reference)/row['path'];sim=directory/'environment/sim'
    if not sim.exists():sim=directory/'environment'  # privileged native layout
    if hashlib.sha256((directory/'task.toml').read_bytes()).hexdigest()!=row['task_toml_sha256']:
        raise ValueError('Pinned task metadata changed: '+row['task'])
    if (directory/'DIGEST').read_text().strip().removeprefix('sha256:')!=row['upstream_digest']:
        raise ValueError('Pinned task digest changed: '+row['task'])
    suite=row['suite'];published=None
    if suite in {'metaworldplus','robocasa','dexjoco','abc-sim'}:
        path=sim/'instance.npz';physical={}
        with np.load(path,allow_pickle=False)as values:
            if suite in {'metaworldplus','robocasa'}:
                keys=sorted(k for k in values.files if k in {'integration','xml'}or k.startswith('model_'))
                if 'integration'not in keys or not any(k.startswith('model_')for k in keys):
                    raise ValueError('Missing frozen MuJoCo state')
                published=str(values['instance_sha'].item())
            elif suite=='dexjoco':
                keys=sorted(k for k in values.files if k in {'physics','mocap_pos','mocap_quat'}or k.startswith('model__'))
                if 'physics'not in keys or not any(k.startswith('model__')for k in keys):
                    raise ValueError('Missing frozen DexJoCo physical state')
            else:
                keys=['qpos0']
                if 'scene_xml'not in values.files:raise ValueError('Missing ABC-Sim scene geometry')
                tree=ET.fromstring(str(values['scene_xml'].item()));tree.attrib.pop('model',None)
                physical['scene_xml']=hashlib.sha256(ET.tostring(tree)).hexdigest()
            for key in keys:
                array=values[key]
                physical[key]={'shape':array.shape,'dtype':str(array.dtype),
                               'sha256':hashlib.sha256(array.tobytes()).hexdigest()}
        identity=digest(physical);method='canonical initial physical arrays and frozen model geometry'
    elif suite in {'libero-max','robodojo','robolab','vlabench','yamlab'}:
        path=sim/'instance.json';values=json.loads(path.read_text())
        key={'libero-max':'initial_state_sha','vlabench':'initial_state_sha256'}.get(suite,'state_sha')
        published=values[key]
        if not isinstance(published,str)or not re.fullmatch('[a-f0-9]{16}|[a-f0-9]{64}',published):
            raise ValueError('Missing native frozen-state digest: '+suite)
        if suite in {'robodojo','robolab','yamlab'}:
            # These releases publish truncated digests. Hash the actual frozen
            # layout/state and its model/asset declarations at full precision.
            keys=(['layout']if suite=='robodojo'else
                  ['initial_state','objects','assets','robot_joint_positions','physics','table_top_z'])
            physical={k:values[k]for k in keys if k in values}
            if not physical:raise ValueError('Missing actual frozen native state')
            identity=digest(physical);method='canonical frozen layout/state and declared physical asset configuration'
        else:
            identity=published;method='published frozen native initial-state SHA256; native verifier qualification required'
    else:raise ValueError('No reviewed reset identity method: '+suite)
    return {'task':row['task'],'suite':suite,'scene_sha256':identity,
        'upstream_instance_sha256':published,'method':method,
        'source_files':{path.name:hashlib.sha256(path.read_bytes()).hexdigest()},
        'task_toml_sha256':row['task_toml_sha256'],'upstream_digest':row['upstream_digest']}


def audit(reference,training,protected):
    """Keep protected reset IDs for matching suites; no held-out outcomes read."""
    suites={r['suite']for r in training};reserved={};errors=[]
    for row in protected:
        if row['suite']not in suites:continue
        try:
            scene=scene_identity(reference,row)
            reserved.setdefault(scene['scene_sha256'],[]).append(row['task'])
        except Exception as exc:errors.append({'task':row['task'],'error':repr(exc)})
    results={}
    for row in training:
        try:
            scene=scene_identity(reference,row);overlap=reserved.get(scene['scene_sha256'],[])
            results[row['task']]={'scene':scene,'protected_reset_overlap':overlap,
                                  'source_scene_check_passed':not overlap and not errors}
        except Exception as exc:results[row['task']]={'source_scene_check_passed':False,'error':repr(exc)}
    return {'tasks':results,'protected_state_errors':errors,
            'scope':'Pinned source and exact initial-state identity. No policy rollouts or complete asset-ancestry claim.',
            'protected_outcomes_read':False}
