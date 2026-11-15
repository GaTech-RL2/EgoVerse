"""Standalone, checksum-verified direct-control teacher and legacy student replays."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .report import ROOT, read, write
from ..tracking.local_service import mirror_viewer


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    root=ROOT/'direct-control-v1'
    plan=read(root/'plan.json',{})
    scene_audit=read(root/'scene-audit/result.json',{})
    scene_collisions={r['task']:r['protected_task'] for r in scene_audit.get('collisions',[])}
    sources=[]
    for item in plan.get('builds',[]):
        folder=root/item['name']/'public-episodes'
        for row in read(folder/'manifest.json',{'episodes':[]})['episodes']:
            if row.get('id') and all((folder/row['id']/n).is_file() and sha(folder/row['id']/n)==row[k]
                for n,k in [('trace.json','trace_sha256'),('replay.mp4','video_sha256')]):
                sources.append((item['name'],row,folder))
    if not sources:return
    conditions=[
        {'id':'legacy-warm96','label':'Warmstart 96 · original prompt','matched_prompt':False,
         'folder':ROOT/'training-evaluation-v1/warm96/public-episodes'},
        {'id':'base-direct','label':'Base Qwen · direct-control prompt','matched_prompt':True,
         'folder':root/'students/base-direct/public-episodes'},
        {'id':'warm96-direct','label':'Warmstart 96 · direct-control prompt','matched_prompt':True,
         'folder':root/'students/warm96-direct/public-episodes'},
        {'id':'base-aligned','label':'Base Qwen · verified image preprocessing','matched_prompt':True,
         'folder':root/'students/base-aligned/public-episodes'},
        {'id':'curated-warm64','label':'Curated warmstart · 64K training','matched_prompt':True,'curated_training':True,
         'folder':root/'students/curated-warm64/public-episodes'},
        {'id':'curated-leact','label':'Curated warmstart + LeAct R1','matched_prompt':True,'curated_training':True,
         'folder':root/'students/curated-leact/public-episodes'},
        {'id':'qwen36-base-v2','label':'Qwen3.6 · original Responses parser (type errors)','matched_prompt':True,
         'folder':ROOT/'structured-control-v2/qwen36-base-r2/public-episodes'},
        {'id':'qwen36-base','label':'Qwen3.6 · corrected tool argument types','matched_prompt':True,
         'folder':ROOT/'structured-control-v2/qwen36-base-r3/public-episodes'},
        {'id':'qwen36-fixed-movement','label':'Qwen3.6 · directed movement diagnostic','matched_prompt':False,
         'folder':ROOT/'fixed-action-v1/runs/v3/qwen36-movement/public-episodes'},
        {'id':'astra-fixed-movement','label':'Astra · directed movement diagnostic','matched_prompt':False,
         'folder':ROOT/'fixed-action-v1/runs/v3/astra-movement/public-episodes'},
        {'id':'qwen36-fixed-task','label':'Qwen3.6 · constrained output, native actions','matched_prompt':False,
         'folder':ROOT/'fixed-action-v1/runs/v3/qwen36-task/public-episodes'},
        {'id':'astra-fixed-task','label':'Astra · constrained output, native actions','matched_prompt':False,
         'folder':ROOT/'fixed-action-v1/runs/v3/astra-task/public-episodes'},
        {'id':'qwen36-wrapper-movement','label':'Qwen3.6 · JSON wrapper adapter · movement','matched_prompt':False,
         'folder':ROOT/'fixed-action-v1/runs/v4/qwen36-movement/public-episodes'},
        {'id':'qwen36-wrapper-task','label':'Qwen3.6 · JSON wrapper adapter · tasks','matched_prompt':False,
         'folder':ROOT/'fixed-action-v1/runs/v4/qwen36-task/public-episodes'},
        {'id':'qwen36-constrained-warm32','label':'Qwen3.6 · constrained warmstart LoRA · step 32','matched_prompt':False,
         'folder':ROOT/read(ROOT/'fixed-action-v1/active-warmstart.json',
             {'directory':'fixed-action-v1/warmstart-lora-v1'})['directory']/'evaluation/public-episodes'},
        {'id':'qwen36-constrained-leact','label':'Qwen3.6 · warmstart + LeAct R1','matched_prompt':False,
         'folder':ROOT/'fixed-action-v1/leact-lora-v1/evaluation/public-episodes'},
    ]
    for arm in ['warmstart','leact']:
        for seed in [101,202]:
            conditions.append({'id':f'qwen36-{arm}-seed{seed}',
                'label':f'Qwen3.6 · {arm} · repeat seed {seed} (3 training tasks)','matched_prompt':False,
                'folder':ROOT/f'fixed-action-v1/leact-followup-v1/{arm}-seed{seed}/public-episodes'})
    students={c['id']:{r['task']:r for r in read(c['folder']/'manifest.json',{'episodes':[]})['episodes']} for c in conditions}
    template=Path(__file__).with_suffix('.html')
    signature=hashlib.sha256(json.dumps({'teacher':[r for _,r,_ in sources],
        'student':students,'scene_collisions':scene_collisions,
        'source':sha(Path(__file__)),'template':sha(template)},sort_keys=True).encode()).hexdigest()
    if read(root/'viewer-published.json',{}).get('signature')==signature:return
    def episode(folder,row):
        trace=read(folder/row['id']/'trace.json')
        return {'trace':trace,'video':base64.b64encode((folder/row['id']/'replay.mp4').read_bytes()).decode()}
    cases=[]
    for batch,row,folder in sources:
        teacher=episode(folder,row);student=None;matched=None
        comparisons={}
        for condition in conditions:
            warm=condition['folder'];baseline=students[condition['id']].get(row['task'])
            if baseline and all((warm/baseline['id']/n).is_file() and sha(warm/baseline['id']/n)==baseline[k]
                for n,k in [('trace.json','trace_sha256'),('replay.mp4','video_sha256')]):
                current=episode(warm,baseline)
                def initial(ep):return next((e.get('state_sha') for e in ep['trace']['simulator_events'] if e.get('ev')=='start'),None)
                current['initial_state_match']=bool(initial(teacher) and initial(teacher)==initial(current))
                comparisons[condition['id']]=current
                if condition['id']=='legacy-warm96':student=current;matched=current['initial_state_match']
        cases.append({'id':batch+'/'+row['task'],'task':row['task'],'batch':batch,'teacher':teacher,
            'student':student,'initial_state_match':matched,'method_pass':row['direct_method_pass'],
            'students':comparisons,'candidate_episode':row.get('candidate_episode'),
            'scene_collision':scene_collisions.get(row['task']),
            'robot_submits':row['robot_submits'],'eligible_targets':row['eligible_targets']})
    value={'updated_utc':datetime.now(timezone.utc).isoformat(),'cases':cases,
           'conditions':[{k:v for k,v in c.items() if k!='folder'} for c in conditions]}
    data=json.dumps(value,ensure_ascii=False).replace('<','\\u003c').replace('&','\\u0026').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    document=template.read_text().replace('<!-- DATA -->','<script type="application/json" id="data">'+data+'</script>')
    output=ROOT/'efe-direct-control.html';temporary=output.with_suffix('.html.tmp');temporary.write_text(document);temporary.replace(output)
    mirror_viewer(output)
    write(root/'viewer-published.json',{'signature':signature,'updated_utc':value['updated_utc'],'episodes':len(cases),'html_sha256':sha(output)})


if __name__=='__main__':build()
