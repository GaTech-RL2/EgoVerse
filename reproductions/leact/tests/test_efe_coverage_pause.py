import json

from reproductions.leact.cluster import launch_efe_coverage as coverage
from reproductions.leact.cluster import launch_efe_coverage_train as training


def test_user_paused_stage_cannot_query_or_resubmit_its_cancelled_attempt(tmp_path,monkeypatch):
    monkeypatch.setattr(coverage,'DEST',tmp_path)
    monkeypatch.setattr(training,'node_limit',lambda:8)
    monkeypatch.setattr(training,'reserved_nodes',lambda active:0)
    def forbidden(*args,**kwargs):raise AssertionError('Paused training must not query or submit a replacement')
    monkeypatch.setattr(training,'command',forbidden)
    monkeypatch.setattr(training,'submit',forbidden)
    stage=tmp_path/'leact-r1';stage.mkdir()
    (stage/'pause.json').write_text(json.dumps({'active':True,'reason':'Prioritize PEAR'}))
    (tmp_path/'training-queue.json').write_text(json.dumps({'stages':{'leact-r1':{
        'attempts':[{'workflow':'cancelled-attempt'}],'retry':True}}}))
    evidence={'training':{'restart-probe':{'success':True},'warmstart':{'complete':{'success':True}},
        'leact-r1':{'progress':{'completed':1092,'total':2048}}}}
    for _ in range(2):
        result=training.tick({},evidence,collection_gate=True,active=[],preparation_ready=True)
        row=next(v for v in result['jobs']if v['stage']=='leact-r1')
        assert row['status']=='PAUSED_BY_USER'
        assert row['progress']['completed']==1092
        queue=json.loads((tmp_path/'training-queue.json').read_text())
        assert queue['stages']['leact-r1']['retry']is False
