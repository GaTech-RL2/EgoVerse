"""Retarget the separately requested L40 workers without replacing allocations."""
from pathlib import Path

import yaml

from .scale_efe import read,write,command
from .launch_efe_pear_scaled import DEST


def tick(state):
    path=DEST/'additional-l40-03-20261119.json';plan=read(path,{})
    if plan.get('model_workflow')!=state.get('workflow'):return
    for row in plan.get('jobs',[]):
        old_run=row.get('model_run',plan['model_run'])
        if old_run==state['run']:continue
        query=command(['osmo','workflow','query',row['workflow'],'--format-type','json'])
        if query['status']!='RUNNING':continue
        from .efe_pear_allocation_jobs import running_tasks,submit
        from .launch_efe_pear_primary import worker_build
        from .recover_efe_pear_graphics import frozen_files
        running_tasks(query)
        old=Path(row['build']);spec=read(old/'spec.json')
        if spec['transport_root']!=old_run+'/transport':raise ValueError('Extra worker has an unexpected transport owner')
        for key in ['output','transport_root','finished','head_heartbeat']:
            if key in spec:
                if not spec[key].startswith(old_run+'/'):raise ValueError('Extra worker paths cross run owners')
                spec[key]=state['run']+spec[key][len(old_run):]
        files=frozen_files(old)
        if read(files['native-membership.json'])!=read(Path(state['build'])/'native-membership.json'):
            raise ValueError('Extra workers require the same admitted native environment membership')
        workflow=yaml.safe_load((old/'workflow.yaml').read_text())['workflow'];task=workflow['tasks'][0]
        build=worker_build(spec,files,spec['allocation_job_entry'],task['image'],task['command'][0],**workflow['resources']['default'])
        queue=read(DEST/'primary-native-queue.json')
        candidates=[v for v in queue.values()if v['workflow']==row['workflow']and not v.get('superseded')]
        if len(candidates)!=1:raise ValueError('Extra worker has ambiguous allocation ownership')
        item=candidates[0];receipt=submit(item,build,replace_running=True,job_id='retarget-'+build.name.rsplit('-',1)[-1])
        for value in queue.values():
            if value['workflow']==row['workflow']:value['superseded']=True
        queue[build.name]={**item,**receipt,'build':str(build),'allocation_build':item.get('allocation_build',str(old)),
            'superseded':False,'model_job_id':state['job_id']}
        write(DEST/'primary-native-queue.json',queue)
        row.update(build=str(build),model_run=state['run'],job_id=receipt['job_id']);write(path,plan)
