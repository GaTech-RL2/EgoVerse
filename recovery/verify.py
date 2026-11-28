"""Run local recovery checks without starting training, services, or remote jobs."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-package',type=Path)
    parser.add_argument('--report',type=Path,default=Path('recovery/runtime/validation.json'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    runtime=root/'recovery/runtime';runtime.mkdir(parents=True,exist_ok=True)
    checks=[]
    manifest=root/'recovery/file-manifest.json'
    if manifest.exists():
        entries=json.loads(manifest.read_text())['files']
        mismatches=[]
        for rel,expected in entries.items():
            p=root/rel
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
                mismatches.append(rel)
        checks.append(dict(name='recovered_file_integrity',passed=not mismatches,
                           files=len(entries),mismatches=mismatches))
    imported=json.loads((root/'recovery/import-manifest.json').read_text())['files']
    paths={x['destination'] for x in imported if x['destination'].endswith('.py')}
    paths.update(str(p.relative_to(root)) for d in ['campaign','recovery','tests']
                 for p in (root/d).rglob('*.py') if '__pycache__' not in p.parts)
    syntax=[]
    for rel in sorted(paths):
        try:ast.parse((root/rel).read_text(),filename=rel)
        except SyntaxError as exc:syntax.append(dict(file=rel,error=str(exc)))
    checks.append(dict(name='python_syntax',passed=not syntax,files=len(paths),errors=syntax))
    env=os.environ.copy()
    env.update(PYTHONPATH=str(root),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               MPLCONFIGDIR=str(runtime/'matplotlib'),CHECK_BATCH='1')
    commands=[('behavior_tests',['-m','pytest','-q','tests/test_local_recovery.py']),
              ('model_ABCD',['tests/recovered/check_model.py']),
              ('objectives_fk',['tests/recovered/check_objectives.py']),
              ('paired_transfer_contracts',['tests/recovered/check_contracts.py']),
              ('gaze_preprocessing',['tests/recovered/check_gaze_preprocessing.py']),
              ('gaze_model',['tests/recovered/check_gaze.py'])]
    commands += [(name+'_help',['-m','campaign.'+name,'--help'])
                 for name in ['train','train_lora','train_cloth','train_rgb']]
    if args.checkpoint_package:
        commands.append(('deployed_checkpoint',['-m','recovery.check_deployed',
                         str(args.checkpoint_package.resolve()),'--report',str(runtime/'checkpoint.json')]))
    for name,command in commands:
        started=time.monotonic()
        result=subprocess.run([sys.executable,*command],cwd=root,env=env,
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        log=runtime/(name+'.log');log.write_text(result.stdout)
        checks.append(dict(name=name,passed=result.returncode==0,exit_code=result.returncode,
                           seconds=round(time.monotonic()-started,3),log=str(log.relative_to(root))))
        print(name,'PASS' if result.returncode==0 else 'FAIL',flush=True)
        if result.returncode:print(result.stdout[-4000:],flush=True)
    report=dict(passed=all(c['passed'] for c in checks),network_used=False,hardware_used=False,
                production_training_run=False,checks=checks)
    out=args.report if args.report.is_absolute() else root/args.report
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],report=str(out))))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
