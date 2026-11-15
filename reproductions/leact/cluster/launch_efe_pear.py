"""Freeze and start the separate PEAR qualification without changing LeAct."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import yaml

from .scale_efe import BASE, ROOT, BUILD_ROOT, read, write, submit
from .launch_efe_training import freeze_sources
from .launch_efe_coverage import DURABLE, DEST as COVERAGE

DEST=ROOT/'pear-v1'
UPSTREAM='f32aa1f7c2ec9c3f45630cc9f8541b3201e8e087'
PAYLOAD_VERSION=2


def freeze_membership():
    admission=read(COVERAGE/'evidence.json')['admission']
    coverage=read(COVERAGE/'plan.json')
    allowed={r['task'] for r in coverage['tasks']}
    protected={r['task'] for r in coverage['protected_tasks']}
    tasks={}
    for row in admission['episodes']:
        if row['task'] not in allowed or row['task'] in protected:
            raise ValueError('Training membership overlaps protected tasks')
        tasks.setdefault(row['task'],set()).add((row['scene_sha256'],row['initial_state_sha256']))
    if len(tasks)!=admission['unique_tasks'] or not admission['gate_passed']:
        raise ValueError('Frozen warmstart membership is incomplete')
    value={'source_decisions_sha256':admission['decisions_sha256'],
           'training_tasks':{k:[{'scene_sha256':s,'initial_state_sha256':i} for s,i in sorted(v)]
                             for k,v in sorted(tasks.items())},
           'scope':'Primary RL comparison uses exactly the tasks and resets represented in the admitted warmstart; broader training coverage is a separate ablation.'}
    path=DEST/'training-membership.json'
    if path.exists() and read(path)!=value:raise ValueError('RL training membership changed')
    if not path.exists():write(path,value)
    return {'training_tasks':len(tasks),'training_membership_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_decisions_sha256':admission['decisions_sha256']}


def prepare():
    DEST.mkdir(exist_ok=True)
    membership=freeze_membership()
    build=BUILD_ROOT/f'efe-pear-qualification-20261114-v{PAYLOAD_VERSION}'
    if (build/'source-manifest.json').exists():return build
    build.mkdir(exist_ok=True)
    spec={'phase':'sampler_and_loss_qualification','model':'Qwen/Qwen3.6-35B-A3B',
          'coverage_root':DURABLE,'seed':20261114,'context':65536,'max_tokens':8192,
          'capture_top_k':32,'samples':2,'upstream_pear_commit':UPSTREAM,
          'uses_base_only_for_runtime_check':True,'native_rollouts':0,'optimizer_updates':0}
    write(build/'spec.json',spec)
    mapping={'efe_pear_qualification.py':'qualification.py','efe_pear_loss.py':'efe_pear_loss.py',
             'efe_model_cache.py':'efe_model_cache.py'}
    for src,dst in mapping.items():shutil.copyfile(BASE/'cluster'/src,build/dst)
    lane='/tmp/efe-pear-qualification'
    workflow={'workflow':{'name':f'leact-efe-pear-qualification-v{PAYLOAD_VERSION}',
        'resources':{'default':{'gpu':8,'cpu':48,'memory':'512Gi','storage':'240Gi','platform':'dgx-h100'}},
        'timeout':{'queue_timeout':'20m','exec_timeout':'2h'},
        'tasks':[{'name':'qualify',
          'image':'docker.io/vllm/vllm-openai@sha256:7a0f0fdd2771464b6976625c2b2d5dd46f566aa00fbc53eceab86ef50883da90',
          'command':['python3'],'args':['-u',lane+'/qualification.py','--spec',lane+'/spec.json','--output','{{output}}'],
          'files':[{'localpath':n,'path':lane+'/'+n} for n in ['spec.json',*mapping.values()]]}]}}
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
    for p in build.glob('*.py'):compile(p.read_text(),str(p),'exec')
    freeze_sources(build)
    sources=DEST/'sources/pear'
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'qualification_prepared',
      'question':'Does native simulator reward RL improve the same constrained EFE student more than LeAct, and does score centering stabilize it?',
      'upstream':{'url':'https://mit-han-lab.github.io/pear-blog/','code':'https://github.com/mit-han-lab/pear',
                  'commit':UPSTREAM,'score_centering':'https://arxiv.org/abs/2609.20807',
                  'training_config_sha256':hashlib.sha256((sources/'pear/configs/training.yaml').read_bytes()).hexdigest()},
      'common_initialization':{'stage':'coverage warmstart final','ready':DURABLE+'/warmstart/warmstart-ready.json',
         'required_optimizer_steps':737,'model':spec['model'],'require_same_manifest_all_arms':True,
         'intermediate_checkpoints_are_not_final_comparators':True},
      'conditions':[
        {'id':'warmstart','label':'Warmstart only','baseline':'base','status':'training_recovery'},
        {'id':'leact-r1','label':'Warmstart + LeAct','baseline':'warmstart','status':'existing_dependency_gate'},
        {'id':'pear-sc','label':'Warmstart + PEAR-style RL + score centering','baseline':'warmstart','status':'integration_qualification'},
        {'id':'pear-no-sc','label':'Warmstart + PEAR-style RL without centering','baseline':'pear-sc','status':'integration_qualification'}],
      'matched_controls':{**membership,'coverage_plan_sha256':hashlib.sha256((COVERAGE/'plan.json').read_bytes()).hexdigest(),
        'protocol':'efe-constrained-action-v4-task','evaluation':'Existing frozen 36-task panel, seen scenes and new tasks reported separately',
        'train_partition_only':True,'same_episode_and_output_budgets_as_existing_eval':True,
        'rl_arms_identical_except_score_centering':True,'heldout_never_used_for_group_filter_or_curriculum':True,
        'same_lora_rank':32,'same_lora_alpha':64},
      'rl_pilot':{'retained_groups_per_update':4,'rollouts_per_group':8,'optimizer_steps':32,
        'max_candidate_groups_per_update':64,'empty_group_policy':'record and stop at budget; do not fabricate success or loop forever',
        'optimizer':'AdamW','learning_rate':1e-6,'updates_per_batch':1,'loss_reduction':'global generated-token mean',
        'advantage':'GRPO group-centered reward with sample standard deviation',
        'reward':'native simulator final grade only','group_filter':'drop zero-variance groups; log them in sample cost',
        'sampling':{'temperature':1.0,'top_p':1.0,'top_k':-1,'topk_log_probs':32},
        'ppo_clipping':False,'importance_sampling':False,'kl_penalty':0,'entropy_bonus':0,
        'initial_sync_mode':'synchronous rounds to validate policy version and token alignment before async',
        'max_nodes_per_active_rl_arm':2,'shared_study_node_cap':8,
        'qualification_gate':'GPU gradient checks, exact multimodal tokens and sampler probabilities, native reset-group identity, one finite real update with replay'},
      'metrics':['native success rate by split and suite','invalid actions','truncation rate','motion and valid control',
         'successes per GPU-hour','native episodes including filtered groups','generated tokens','seconds per update',
         'sampler/trainer logprob mismatch','score-centering correction magnitude','gradient norms','policy-version lag'],
      'deviations_from_paper':['Qwen3.6-35B-A3B with shared Astra warmstart rather than Qwen3.8-27B from base',
        'EFE and its constrained robot-specific action interface rather than LIBERO/RoboDojo',
        'LoRA and bounded synchronous pilot rather than full-model fully asynchronous 32-node training',
        '4 retained groups per update for the pilot rather than 128; same 8 samples per group',
        'existing EFE context/output limits rather than 160k/256k full-episode context'],
      'next_dependencies':['Finish common warmstart','Complete grouped EFE native rollout adapter and policy-update integration',
                           'Pass one-update replay before running the two RL arms'],
      'qualification_build':str(build),'performance_claim':None}
    write(DEST/'plan.json',plan)
    return build


def main():
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');args=p.parse_args()
    build=prepare()
    if args.submit:
        receipt=submit(build,priority='HIGH')
        write(DEST/'status.json',{'phase':'qualification_submitted','workflow':receipt['name'],
            'updated_utc':datetime.now(timezone.utc).isoformat(),'optimizer_updates':0,'native_rollouts':0,
            'summary':'H100 job checks numerical loss gradients and actual Qwen sampler top-32 probabilities on frozen EFE training observations. RL training is not yet launched.'})
        print(json.dumps(receipt))
    else:print(build)


if __name__=='__main__':main()
