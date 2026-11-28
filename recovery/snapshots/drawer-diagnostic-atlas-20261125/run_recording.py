import sys,json,traceback
import drawer_diag as d
kind=sys.argv[1];idx=int(sys.argv[2]);rid=json.loads((d.P/(kind+'-recordings.json')).read_text())[idx]
d.make_depth(rid)
errors=[]
for i,e in enumerate(d.segments):
 if e['recording_id']!=rid:continue
 try:d.build(i)
 except Exception as ex:
  traceback.print_exc();d.write(d.P/'segments'/e['uuid']/'error.json',dict(error=repr(ex)));errors.append(i)
assert not errors,errors
