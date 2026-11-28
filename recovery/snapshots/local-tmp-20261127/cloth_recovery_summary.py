import json, pathlib, collections
R=pathlib.Path('/coc/flash7/cxin34/cloth-valid-recovery-fix1124-20261124')
rows=[]
for x in json.load(open(R/'plan.json')):
    e=pathlib.Path(x['episode'])
    p=e/'reviewed-disposition.json'
    if not p.exists(): p=e/'recovery-result.json'
    d=json.load(open(p)) if p.exists() else {}
    rows.append(dict(index=x['index'],take=x['take'],uuid=x['uuid'],bounds=x['recovery_grid_bounds'],status=d.get('status','PENDING')))
print(json.dumps(rows,indent=2))
print(dict(collections.Counter(x['status'] for x in rows)))
