"""Candidate-only workers and isolated timing runs; never update model weights."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from efe_training_data import sha,write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--execution',type=Path,required=True)
    parser.add_argument('--child',action='store_true')
    args=parser.parse_args()
    spec=json.loads(args.spec.read_text());execution=json.loads(args.execution.read_text())
    root=Path(execution['output_root']);root.mkdir(parents=True,exist_ok=True)
    if not args.child:
        manifest=json.loads(Path(spec['parent_manifest']).read_text())
        if manifest['revision']!=spec['base_revision']:raise ValueError('Parent revision mismatch')
        for name,row in manifest['files'].items():
            if sha(Path(spec['base_model'])/name)!=row['sha256']:raise ValueError('Parent weight checksum mismatch')
        if execution.get('benchmark'):
            for concurrency in (1,8):
                child={**execution,'benchmark':False,'max_num_seqs':concurrency,
                    'output_root':str(root/('seq'+str(concurrency))),
                    'worker_root':str(root/('seq'+str(concurrency)))}
                path=root/('seq'+str(concurrency)+'.json');write(path,child)
                with (root/('seq'+str(concurrency)+'.log')).open('a') as log:
                    subprocess.run([sys.executable,'-u',__file__,'--child','--spec',str(args.spec),
                        '--execution',str(path)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
            write(root/'benchmark-complete.json',{'success':True,'seconds_epoch':time.time()})
            return
    from leact import select
    try:
        select(spec,root,{**execution,'candidate_only':True})
    except Exception as error:
        write(Path(execution['worker_root'])/'error.json',{'error':str(error),'workflow':os.environ.get('EFE_WORKFLOW_ID')})
        raise


if __name__=='__main__':main()
