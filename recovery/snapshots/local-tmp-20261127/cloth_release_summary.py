import pathlib,json,collections,time
R=pathlib.Path('/coc/flash7/cxin34/cloth-valid-recovery-fix1124-20261124')
a=json.load(open(R/'reports/final-numeric-audit.json'));m=json.load(open(R/'release/target.json'));p=json.load(open(R/'parents.json'))
out=[]
for parent in p:
 rid=parent['uuid'];es=[e for e in m['episodes']if e['recording_id']==rid];old=[e for e in es if e['uuid']==rid];new=[e for e in es if e['uuid']!=rid]
 out.append(dict(take=parent['take'],recording_id=rid,split=es[0]['split'],original_rows=sum(e['rows']for e in old),recovered_rows=sum(e['rows']for e in new),recovered_segments=len(new),recovered_sample_duration_s=sum(e['rows']for e in new)/60,intervals=[e['recovery_grid_bounds']for e in new],status='recovered'if new else'additional_interval_quality_excluded'))
d=dict(time=time.time(),original_recordings=43,episodes=a['episodes'],rows=a['rows'],added_rows=a['added_rows'],added_segments=a['added'],recovered_recordings=sum(x['recovered_segments']>0 for x in out),recording_split=dict(train=38,valid=5),row_split=a['loader'],per_recording=out,training=dict(job=4855714,root=str(R/'ft'),parent_step=180000,target_steps=60000,seed=42,recipe='B+strongMSC'),note='Disjoint segments; sample count/60 is additive sample-duration, not elapsed recording duration; old43 artifacts unchanged. Validation now has extra valid frames; loss not directly comparable to old row set.')
(R/'reports/release-summary.json').write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items()if k!='per_recording'}))
