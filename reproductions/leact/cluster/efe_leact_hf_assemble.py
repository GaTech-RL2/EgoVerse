"""Prepare corrected R1 data and atomically replace its never-trained admission."""
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def guard_untrained(root):
    if (root/'train/complete.json').exists()or(root/'complete.json').exists():
        raise ValueError('Cannot replace admission after completed optimization')
    metrics=root/'train/metrics.jsonl'
    if metrics.exists()and metrics.read_text().strip():
        raise ValueError('Cannot replace admission after a production optimizer update')
    if list((root/'restart-states').glob('checkpoint-*/training-state.json')):
        raise ValueError('Cannot replace admission after a committed optimizer update')


ARCHIVE_NAMES=['dataset-manifest.json','selection.json','selection-complete.json','scoring-probes.json',
    'membership.json','candidate-report.json','training-lineage.json','spec.json','modality-parity.json',
    'smoke','train','smoke-spec.json','smoke-loss-replay.json','smoke-adapter-ready.json',
    'adapter-ready.json','tensors','tensor-blobs','probes','candidates']
INSTALL_NAMES=['dataset-manifest.json','selection.json','selection-complete.json','scoring-probes.json',
    'membership.json','candidate-report.json','tensors','tensor-blobs','probes']


def activate_prepared(root,prepared,scores,original_dataset_sha256,score_membership_sha256):
    """Journal every directory move so an interrupted promotion can finish safely."""
    from efe_training_data import sha,write
    done=root/'hf-score-activation.json'
    if done.exists():
        receipt=json.loads(done.read_text())
        if sha(root/'dataset-manifest.json')!=receipt['dataset_sha256']:raise ValueError('Activated admission changed')
        return receipt
    guard_untrained(root)
    path=root/'hf-score-activation-pending.json'
    backup=root/'attempts/admission-before-hf-rescore-20261117'
    if path.exists():
        journal=json.loads(path.read_text())
        if(journal['original_dataset_sha256']!=original_dataset_sha256
           or journal['score_membership_sha256']!=score_membership_sha256):raise ValueError('Recovery inputs changed')
    else:
        if sha(root/'dataset-manifest.json')!=original_dataset_sha256:raise ValueError('Original stopped-run dataset changed')
        manifest=json.loads((prepared/'dataset-manifest.json').read_text())
        journal={'phase':'archiving','original_dataset_sha256':original_dataset_sha256,
            'score_membership_sha256':score_membership_sha256,'dataset_sha256':sha(prepared/'dataset-manifest.json'),
            'selection':manifest['selection'],'backup':str(backup)}
        write(path,journal)
    backup.mkdir(parents=True,exist_ok=True)
    if journal['phase']=='archiving':
        for name in ARCHIVE_NAMES:
            original=root/name;target=backup/name
            if target.exists()or target.is_symlink():continue
            if original.exists()or original.is_symlink():shutil.move(str(original),target)
        journal['phase']='installing';write(path,journal)
    for name in INSTALL_NAMES:
        source=prepared/name;target=root/name
        if source.exists()or source.is_symlink():
            if target.exists()or target.is_symlink():raise ValueError('Ambiguous promotion target: '+name)
            shutil.move(str(source),target)
        elif not target.exists():raise ValueError('Missing promoted data: '+name)
    target=root/'candidates'
    if target.is_symlink():
        if target.resolve()!=(scores/'candidates').resolve():raise ValueError('Promoted candidate reference changed')
    elif target.exists():raise ValueError('Unexpected promoted candidate directory')
    else:target.symlink_to(scores/'candidates',target_is_directory=True)
    if sha(root/'dataset-manifest.json')!=journal['dataset_sha256']:raise ValueError('Promoted dataset checksum differs')
    receipt={**journal,'phase':'complete','success':True,'production_optimizer_updates_before_recovery':0,
        'completed_utc':datetime.now(timezone.utc).isoformat()}
    write(done,receipt);return receipt


def main():
    from efe_training_data import sha,write
    from leact import select
    lane=Path(__file__).parent;request=json.loads((lane/'assembly.json').read_text())
    spec=json.loads((lane/'scientific-spec.json').read_text());root=Path(spec['archive_root'])
    scores=Path(request['scores']);prepared_name=request.get('prepared_directory','prepared')
    if Path(prepared_name).name!=prepared_name or not prepared_name.startswith('prepared'):
        raise ValueError('Invalid preparation directory')
    prepared=scores/prepared_name;prepared.mkdir(parents=True,exist_ok=True)
    out=scores/'assembly';out.mkdir(exist_ok=True)
    def status(phase,**extra):
        row={'phase':phase,'updated_utc':datetime.now(timezone.utc).isoformat(),'workflow':os.environ['EFE_WORKFLOW_ID'],
            'production_optimizer_updates':0,**extra};write(out/'status.json',row)
        print('EFE_HF_ASSEMBLY '+json.dumps(row),flush=True)
    if(root/'hf-score-activation.json').exists()or(root/'hf-score-activation-pending.json').exists():
        with(root/'writer.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            receipt=activate_prepared(root,prepared,scores,request['original_dataset_sha256'],sha(out/'score-membership.json'))
            status('activated',receipt=receipt);return
    deadline=time.monotonic()+12*3600
    while True:
        receipts=[scores/f'worker-{i}/complete.json'for i in range(request['shards'])]
        if all(p.exists()for p in receipts):break
        if time.monotonic()>deadline:raise TimeoutError('HF rescoring workers did not finish')
        status('waiting_for_hf_scores',completed_shards=sum(p.exists()for p in receipts),total_shards=len(receipts))
        time.sleep(30)
    for i,path in enumerate(receipts):
        receipt=json.loads(path.read_text());check=json.loads((scores/f'worker-{i}/reference-check.json').read_text())
        if not(receipt.get('success')is True and receipt.get('optimizer_updates')==0 and check.get('passed')is True
                and check.get('frozen_input_tensors_equal')is True):
            raise ValueError('HF scorer did not pass exact parent/input qualification')
    members=json.loads((root/'membership.json').read_text())['states']
    if len(members)!=2048:raise ValueError('The frozen 2048-state membership changed')
    identities={}
    for sid in members:
        path=scores/'candidates'/(sid+'.json');value=json.loads(path.read_text())
        if (value.get('score_backend')!='exact_training_fsdp_fp32_master_bf16_forward'
            or value['original_vllm_record_sha256']!=sha(root/'candidates'/path.name)):
            raise ValueError('HF score provenance differs from frozen original explanations')
        identities[sid]=sha(path)
    write(out/'score-membership.json',{'states':identities,'optimizer_updates':0})
    with (root/'writer.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        done=root/'hf-score-activation.json'
        if done.exists():
            receipt=json.loads(done.read_text())
            if sha(root/'dataset-manifest.json')!=receipt['dataset_sha256']:raise ValueError('Activated HF admission changed')
            status('activated',receipt=receipt);return
        guard_untrained(root)
        if sha(root/'dataset-manifest.json')!=request['original_dataset_sha256']:
            raise ValueError('Original stopped-run dataset changed before recovery')
        status('assembling_frozen_hf_admission',states=len(members),preparation_root=str(prepared))
        if not(prepared/'selection-complete.json').exists():
            select(spec,prepared,{'frozen_training_scores_only':True,'scored_candidates_root':str(scores/'candidates'),
                'original_training_root':str(root),'max_num_seqs':8,
                'reuse_original_tensors':request.get('reuse_original_tensors',False),
                'original_dataset_sha256':request['original_dataset_sha256']})
        result=json.loads((prepared/'selection-complete.json').read_text())
        if not(result.get('success')and not result.get('no_admissions')):raise ValueError('HF assembly did not produce admitted training examples')
        manifest=json.loads((prepared/'dataset-manifest.json').read_text())
        if manifest['selection'].get('score_backend')!='exact_training_fsdp_fp32_master_bf16_forward':
            raise ValueError('Prepared corpus has the wrong scoring backend')
        manifest['scoring_recovery']={'original_dataset_sha256':request['original_dataset_sha256'],
            'score_membership_sha256':sha(out/'score-membership.json'),
            'reason':'Original inference and training scorers differed on identical probes. Reuse every sampled explanation and compute all admission deltas with the actual training implementation.',
            'generated_new_explanations':False,'production_optimizer_updates_before_recovery':0}
        write(prepared/'dataset-manifest.json',manifest)
        guard_untrained(root)
        status('activating_hf_admission',original_dataset_sha256=request['original_dataset_sha256'])
        receipt=activate_prepared(root,prepared,scores,request['original_dataset_sha256'],sha(out/'score-membership.json'))
        status('activated',receipt=receipt)


if __name__=='__main__':main()
