"""Fixed 40-question paired capability screen; no primary training launch."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil

import yaml

from .launch_efe_realworldqa_retention import ROOT
from .launch_efe_training import freeze_sources
from .scale_efe import BASE,BUILD_ROOT,read,submit,write


def prepare(reference_build,qualification,*,attempt=16,questions=40):
    if questions!=40:raise ValueError('This preliminary study has a frozen panel of 40 questions')
    if not qualification.get('success')or qualification.get('batch_size')!=8 or len(qualification.get('batches',[]))!=2:
        raise ValueError('A successful batch-eight serving qualification is required')
    reference_build=Path(reference_build)
    for name,digest in read(reference_build/'source-manifest.json').items():
        if hashlib.sha256((reference_build/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen reference evaluation changed')
    source_spec=read(reference_build/'evaluation-spec.json')
    if qualification['batching_profile']!=source_spec['serving_profile']or any(
            batch['aggregate_output_tokens_per_second']<20 for batch in qualification['batches']):
        raise ValueError('Serving qualification differs from the requested evaluation profile')
    if source_spec['batch_size']!=8 or source_spec['max_output_tokens']!=32768:
        raise ValueError('The preliminary screen retains the qualified inference settings')
    folder=BUILD_ROOT/('rp-jhh-warm496-realworldqa-v4-r'+str(attempt));folder.mkdir(exist_ok=True)
    if (folder/'source-manifest.json').exists():return folder
    spec=dict(source_spec)
    seed=20261118
    spec.update(id='rp_jhh_warm496_realworldqa_preliminary40_v4',study_version=4,
        shared_root='/mnt/amlfs-07/shared/rpunamiya/leact/studies/rp_jhh_warm496_realworldqa_preliminary40_v4',
        purpose='User-requested preliminary paired screen, not the full benchmark score',
        preliminary=True,question_ids=random.Random(seed).sample(range(765),questions),
        question_subset_seed=seed,sampling_method='uniform_seeded_without_replacement',
        execution_order=['warmstart496','base'],
        serving_qualification=qualification,
        runner_sha256=hashlib.sha256((BASE/'cluster/efe_realworldqa_retention.py').read_bytes()).hexdigest())
    write(folder/'spec.json',spec)
    shutil.copyfile(BASE/'cluster/efe_realworldqa_retention.py',folder/'runner.py')
    shutil.copyfile(reference_build/'efe_model_cache.py',folder/'efe_model_cache.py')
    workflow=yaml.safe_load((reference_build/'workflow.yaml').read_text())
    workflow['workflow']['name']='leact-'+folder.name
    workflow['workflow']['labels']={'reasoning_eval_questions':'40','reasoning_eval_scope':'preliminary'}
    workflow['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'12h'}
    task=workflow['workflow']['tasks'][0]
    task['args']=['-u','/tmp/rp-retention/runner.py','--spec','/tmp/rp-retention/spec.json']
    task['files']=[{'localpath':name,'path':'/tmp/rp-retention/'+name}
                   for name in ['runner.py','efe_model_cache.py','spec.json']]
    (folder/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
    freeze_sources(folder)
    return folder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit',action='store_true')
    parser.add_argument('--pool',choices=['groot-h100-01','groot-h100-02'],default='groot-h100-02')
    parser.add_argument('--priority',choices=['HIGH','LOW'],default='HIGH')
    args=parser.parse_args()
    change=read(ROOT/'preliminary-scope-change.json')
    qualification=read(ROOT/'serving-qualification-result.json')
    build=prepare(change['previous_launch']['build'],qualification)
    record={'build':str(build),'attempt':16,'pool':args.pool,'priority':args.priority,
            'questions_per_model':40,'preliminary':True,'scope':'40-question preliminary retention evaluation only',
            'pear_launched':False}
    if args.submit:record.update(submit(build,priority=args.priority,pool=args.pool,nodes=1))
    write(ROOT/'preliminary-launch.json',record)
    if args.submit:write(ROOT/'launch.json',record)
    print(json.dumps(record))


if __name__=='__main__':main()
