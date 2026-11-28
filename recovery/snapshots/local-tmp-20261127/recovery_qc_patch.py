from pathlib import Path
import ast
R=Path('/coc/flash7/cxin34/cloth-valid-recovery-fix1124-20261124');p=R/'scripts/final_audit.py';s=p.read_text()
s=s.replace("if status['status'] in ['QUALITY_QUARANTINED_RETARGET','QUALITY_QUARANTINED_FK']:excluded.append(dict(x,status=status));continue",'''if status['status'] in ['QUALITY_QUARANTINED_RETARGET','QUALITY_QUARANTINED_FK']:
  if status['status']=='QUALITY_QUARANTINED_FK':assert sha(status['evidence'])==status['sha256']
  else:
   assert not json.load(open(e/'retarget-independent.json'))['passed']
   status.update(evidence=str(e/'logs/fixed-retarget-qc.log'),sha256=sha(e/'logs/fixed-retarget-qc.log'))
  excluded.append(dict(x,status=status));continue''')
ast.parse(s);p.write_text(s)
print('Quarantine evidence checks installed')
