import drawer_diag as d
ready=[i for i,e in enumerate(d.segments) if (d.P/'segments'/e['uuid']/'done.json').exists()]
missing=[i for i in range(len(d.segments)) if i not in ready]
d.write(d.P/'completion.json',dict(ready=ready,missing=missing,all_complete=not missing,training_authorized=False))
if ready:d.publish('r2_diagnostic_'+str(len(ready))+'segments',ready)
assert not missing,missing
