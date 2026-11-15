"""Freeze a training-only native PEAR rollout qualification on one H100 node."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import yaml

from .scale_efe import BASE, BUILD_ROOT, read, write, submit
from .launch_efe_training import freeze_sources
from .launch_efe_pear import DEST, COVERAGE, freeze_membership

VERSION = 1
LANE = '/tmp/efe-training-evaluation-v1/warm96'
TEMPLATE = BUILD_ROOT/'efe-coverage-eval-20261105-v1-base-b01-r12'
REMOTE = '/mnt/amlfs-07/shared/rpunamiya/leact/studies/efe-pear-20261114-v1/native'


def prepare():
    membership = freeze_membership()
    qualification = read(DEST/'qualification-result.json')
    if not qualification.get('success'):
        raise ValueError('Actual sampler and numerical loss qualification must pass first')
    folder = BUILD_ROOT/f'efe-pear-native-qualification-20261114-v{VERSION}'
    if (folder/'source-manifest.json').exists():
        return folder
    folder.mkdir(exist_ok=True)
    for name in read(TEMPLATE/'source-manifest.json'):
        shutil.copyfile(TEMPLATE/name, folder/name)
    spec = read(folder/'spec.json')
    allowed = read(DEST/'training-membership.json')['training_tasks']
    # Deterministic training-only selection before any new outcome. Qualification
    # does not choose tasks from the held-out evaluation's model successes.
    tasks = [r for r in read(COVERAGE/'plan.json')['tasks'] if r['task'] in allowed]
    chosen = min(tasks, key=lambda r:hashlib.sha256(('pear-native-v1/'+r['task']).encode()).hexdigest())
    rows = [dict(chosen, partition='train_candidates', collection_attempt=i+1,
                 pear_episode_index=i) for i in range(8)]
    for name in ['resume_completed_trials','retry_of','training_parent','instruction_suffix_file']:
        spec.pop(name, None)
    spec.update(tasks=rows, archive_root=REMOTE, action_durable_root='',
        coverage_evaluation=False, exclude_from_task_leaderboard=True,
        model_id='pear-base-native-qualification', agent_slug='pear-base-native-qualification',
        agent_label='PEAR native group qualification from base', export_role='pear_native_qualification',
        local_results_directory='pear-v1/native-qualification',
        purpose='Validate eight native same-task/reset samples with exact model probabilities; zero optimizer updates.',
        scope='Training-only runtime qualification from base. No warmstart or RL performance claim.',
        training_admission='Qualification only; primary RL training requires common final warmstart.',
        pear={'role':'native_sampler_qualification', 'seed':20261114, 'capture_top_k':32,
              'membership_sha256':membership['training_membership_sha256'], 'group_size':8,
              'sample_temperature':1., 'sample_top_p':1., 'sample_top_k':-1,
              'zero_variance_group_counts_as_valid_qualification':True},
        sampling={'temperature':1., 'top_p':1., 'top_k':-1})
    write(folder/'spec.json', spec)
    mapping = {'efe_pear_native_worker.py':'native-worker.py', 'efe_pear_sampler.py':'efe_pear_sampler.py',
               'efe_pear_groups.py':'efe_pear_groups.py', 'efe_pear_loss.py':'efe_pear_loss.py',
               'efe_training_data.py':'efe_training_data.py', 'efe_model_cache.py':'efe_model_cache.py'}
    for source, target in mapping.items():
        shutil.copyfile(BASE/'cluster'/source, folder/target)
    shutil.copyfile(DEST/'training-membership.json', folder/'training-membership.json')
    source = (BASE/'embodied_first_exam/scene_admission.py').read_text()
    # Use the exact canonicalizer from corpus admission, without local dashboard imports.
    begin, end = source.index('def digest('), source.index('\ndef audit(')
    (folder/'scene_identity.py').write_text('import hashlib,json\nfrom pathlib import Path\n'
        'def read(path): return json.loads(Path(path).read_text())\n\n'+source[begin:end])
    pilot = (folder/'native-pilot.py').read_text()
    before = 'for _ in range(120):'
    if pilot.count(before) != 1:
        raise ValueError('Native pilot release contract changed')
    (folder/'native-pilot.py').write_text(pilot.replace(before, 'for _ in range(0):'))
    workflow = yaml.safe_load((folder/'workflow.yaml').read_text())
    workflow['workflow']['name'] = f'leact-efe-pear-native-v{VERSION}'
    workflow['workflow']['resources']['default'].update(cpu=48, memory='512Gi', storage='800Gi')
    workflow['workflow']['timeout'] = {'queue_timeout':'20m', 'exec_timeout':'12h'}
    task = workflow['workflow']['tasks'][0]
    task.pop('credentials', None)
    paths = {r['localpath'] for r in task['files']}
    for name in [*mapping.values(), 'training-membership.json', 'scene_identity.py']:
        if name not in paths:
            task['files'].append({'localpath':name, 'path':LANE+'/'+name})
    (folder/'workflow.yaml').write_text(yaml.safe_dump(workflow, sort_keys=False))
    for path in folder.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    freeze_sources(folder)
    write(DEST/'native-qualification-plan.json', {'build':str(folder), 'task':chosen['task'],
        'planned_rollouts':8, 'nodes':1, 'gpus_per_node':8, 'optimizer_updates':0,
        'policy':spec['student_revision'], 'role':spec['pear']['role'],
        'selection':'Deterministic SHA order over the 40 frozen admitted training tasks; no evaluation outcomes used.',
        'transport':'Compiled native-serving settings, original top-32 probabilities, T=1/top_p=1/top_k=-1.',
        'native_budget':spec['fixed_action_config'], 'archive_root':REMOTE,
        'training_budget_exhaustion':'Output-length truncations remain model outcomes and sample cost; infrastructure failures stop the group.'})
    return folder


def main():
    p = argparse.ArgumentParser();p.add_argument('--submit', action='store_true');args = p.parse_args()
    folder = prepare()
    if args.submit:
        receipt = submit(folder, priority='HIGH')
        write(DEST/'native-qualification-queue.json', {'attempts':[{'workflow':receipt['name'],
            'build':str(folder), 'priority':'HIGH', 'submitted_utc':datetime.now(timezone.utc).isoformat()}]})
        write(DEST/'status.json', {'phase':'native_qualification_submitted', 'workflow':receipt['name'],
            'updated_utc':datetime.now(timezone.utc).isoformat(), 'optimizer_updates':0, 'native_rollouts':0,
            'summary':'Sampler/loss qualification passed. One H100 node is submitted for eight native training-only rollouts with exact probabilities; RL optimizer training has not started.'})
        print(json.dumps({'workflow':receipt['name'], 'build':str(folder)}), flush=True)
    else:
        print(folder)


if __name__ == '__main__':
    main()
