"""Family previews and task membership for the PEAR planning workspace.

Images illustrate simulator families. Frozen split records determine membership;
an image is never evidence that a task has been used for training or evaluation.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


NAMES={'abc-sim':'ABC Sim','dexjoco':'DexJoCo','libero-max':'LIBERO-Max',
    'metaworldplus':'MetaWorldPlus','robocasa':'RoboCasa','robodojo':'RoboDojo',
    'robolab':'RoboLab','vlabench':'VLABench','yamlab':'YAMLab (EFE)',
    'native-gr00t-yam':'Native GR00T YAM','pandadyn':'PandaDyn',
    'robopaint':'RoboPaint','robotwin-2':'RoboTwin-2','robowits':'RoboWits'}


def simulator_families(inventory,review,images=None):
    images=images or {};entries=inventory['entries'];by_id={r['id']:r for r in entries}
    train=defaultdict(list);seen=defaultdict(list);unseen=defaultdict(list)
    for row in entries:
        if row['role']=='PEAR training candidate':train[row['suite']].append(row)
    for row in review.get('core_entries',[]):
        if row['cohort']=='same_sim_new_task':seen[row['suite']].append(by_id[row['task']])
    for row in review.get('unseen_reserve_entries',[]):
        unseen[row['suite']].append(by_id[row['task']])
    if seen.keys()-train.keys()or unseen.keys()&train.keys():
        raise ValueError('Evaluation simulator categories disagree with the proposed training families')
    train_ids={r['id']for rows in train.values()for r in rows}
    eval_ids={r['id']for rows in [*seen.values(),*unseen.values()]for r in rows}
    if train_ids&eval_ids:raise ValueError('A task cannot be in both train and evaluation lists')
    def task(row):
        return {k:row.get(k)for k in ['id','description','semantic_task']}
    families=[]
    for suite in sorted(train.keys()|seen.keys()|unseen.keys()):
        families.append({'id':suite,'name':NAMES.get(suite,suite),'image':images.get(suite),
            'train':[task(r)for r in sorted(train[suite],key=lambda r:r['id'])],
            'eval_seen':[task(r)for r in sorted(seen[suite],key=lambda r:r['id'])],
            'eval_unseen':[task(r)for r in sorted(unseen[suite],key=lambda r:r['id'])],
            'pending':[],'split_status':'training family'if train[suite]else'unseen simulator'})
    native={}
    for row in entries:
        if row['source']=='Native GR00T':
            native.setdefault(row['semantic_task'],{'id':row['semantic_task'],'description':row['description']})
    if native:
        families.append({'id':'native-gr00t-yam','name':NAMES['native-gr00t-yam'],
            'image':images.get('native-gr00t-yam'),'train':[],'eval_seen':[],'eval_unseen':[],
            'pending':[native[key]for key in sorted(native)],'split_status':'split pending'})
    families.sort(key=lambda r:(bool(r['eval_unseen']),r['name'].lower()))
    return {'families':families,'counts':{'train':sum(len(r['train'])for r in families),
        'eval_seen':sum(len(r['eval_seen'])for r in families),
        'eval_unseen':sum(len(r['eval_unseen'])for r in families),
        'pending':sum(len(r['pending'])for r in families)},
        'basis':'Proposed PEAR training membership; seen-simulator evaluation uses core same_sim_new_task entries; unseen evaluation uses the frozen simulator reserve.',
        'note':'Family images are illustrative. Native GR00T YAM is distinct from EFE YAMLab and awaits a frozen train/eval split. Scene and exposure audits remain provisional.'}


def main():
    from .reasoning import BASE,build,read,write
    base=BASE/'artifacts/reasoning-workspace';media=base/'media';media.mkdir(parents=True,exist_ok=True)
    value=build();families=value['simulators']['families']
    def fetch(family):
        suite=family['id']
        if suite=='native-gr00t-yam':
            metadata=read(BASE/'artifacts/embodied-first-exam/pear-scaled-v1/sources/gr00t-sim-source.json',{})
            path='docs/images/task_creation/place_objects.png'
            data=(Path(metadata['source'])/path).read_bytes();extension='png'
            source='https://gitlab-master.nvidia.com/GR00T/sim/gr00t-sim/-/blob/'+metadata['revision']+'/'+path
            caption='GR00T scene authoring view'
        else:
            index_url='https://huggingface.co/api/datasets/MLL-Lab/embodied-exam-media/tree/main/site/'+suite+'?recursive=false&limit=1000'
            with urlopen(index_url,timeout=40)as response:items=json.load(response)
            candidates=[r['path']for r in items if r['type']=='file'and r['path'].endswith('.jpg')]
            if not candidates:raise ValueError('No official family preview for '+suite)
            selected=next((p for p in candidates if p.endswith('/_teaser.jpg')),candidates[0])
            source='https://huggingface.co/datasets/MLL-Lab/embodied-exam-media/resolve/main/'+selected
            with urlopen(source,timeout=60)as response:data=response.read(5_000_001)
            extension='jpg';caption='EFE reference image'
        if len(data)>5_000_000 or not(data.startswith(b'\xff\xd8\xff')or data.startswith(b'\x89PNG\r\n\x1a\n')):
            raise ValueError('Invalid preview image for '+suite)
        digest=hashlib.sha256(data).hexdigest();name=digest+'.'+extension
        (media/name).write_bytes(data)
        return suite,{'url':'/reasoning-media/'+name,'file':name,'sha256':digest,
            'source_url':source,'caption':caption,'kind':'illustrative family preview'}
    with ThreadPoolExecutor(max_workers=4)as pool:images=dict(pool.map(fetch,families))
    write(base/'family-images.json',images);build()
    print(json.dumps({'families':len(images),'bytes':sum((media/r['file']).stat().st_size for r in images.values())}))


if __name__=='__main__':main()
