"""Resume registered trials without rerunning completed episodes or hiding outages."""

import copy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .worker import safe_relative, sha256, write_json


def trial_key(row):
    return row['case'], row['arm'], row['attempt']


def pending_trials(cases, rows):
    """Validate a partial cohort and preserve the original case/attempt ordering."""
    indexed = {}
    allowed = {c['id'] for c in cases}
    for row in rows:
        key = trial_key(row)
        if (key in indexed or key[0] not in allowed or key[1] not in ('native', 'astra')
                or type(key[2]) is not int or key[2] not in (1, 2)
                or row.get('completed') is not True or type(row.get('success')) is not bool):
            raise ValueError('Invalid completed trial in continuation')
        indexed[key] = row
    pending = []
    for case in cases:
        for arm in ('native', 'astra'):
            first, second = (indexed.get((case['id'], arm, n)) for n in (1, 2))
            if second and (not first or first['success']):
                raise ValueError('Continuation violates stop-on-success or first-attempt ordering')
        for attempt in (1, 2):
            for arm in ('native', 'astra'):
                first = indexed.get((case['id'], arm, 1))
                key = case['id'], arm, attempt
                if key not in indexed and not (first and first['success']):
                    pending.append(key)
    return pending


def retained_path(relative, protocol):
    """Keep parent episodes byte-for-byte, isolating its interrupted attempt."""
    path = safe_relative(relative)
    amendment = protocol['continuation']
    interrupted = amendment['interrupted_trial']
    prefix = Path('evaluation') / interrupted['case'] / f"{interrupted['arm']}{interrupted['attempt']}"
    if path.is_relative_to(prefix):
        return Path('interrupted') / amendment['parent_workflow'] / path.relative_to('evaluation')
    if path.parts[0] == 'history':
        return path
    if len(path.parts) >= 3 and path.parts[0] == 'evaluation' and path.parts[1] in {c['id'] for c in protocol['cases']}:
        return path
    return Path('continuation/parent') / path


def restore_parent(client, protocol, output):
    amendment = protocol['continuation']
    parent = amendment['parent_workflow']
    if not parent.startswith('astra-complex-20261006-robocasa-xiaomi-selector-'):
        raise ValueError('Unexpected continuation parent')
    prefix = f'experiments/astra-complex-20261006/{parent}/results/'
    payload = client.get_object(Bucket='rldb', Key=prefix+'archive_receipt.json')['Body'].read()
    if hashlib.sha256(payload).hexdigest() != amendment['parent_receipt_sha256']:
        raise ValueError('Parent archive changed after continuation registration')
    receipt = json.loads(payload)
    if sum(item['bytes'] for item in receipt['files'].values()) > 512*1024**2:
        raise ValueError('Unexpected parent archive size')
    mapping = {}
    for relative, item in receipt['files'].items():
        retained = retained_path(relative, protocol)
        target = output / retained
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            # A partial network download must never be published as evidence.
            temporary = Path('/opt/astra-resume-download')
            client.download_file('rldb', prefix+relative, str(temporary))
            if temporary.stat().st_size != item['bytes'] or sha256(temporary) != item['sha256']:
                raise ValueError('Parent evidence checksum differs: '+relative)
            temporary.replace(target)
        if target.stat().st_size != item['bytes'] or sha256(target) != item['sha256']:
            raise ValueError('Retained parent evidence differs: '+relative)
        mapping[relative] = dict(retained_relative=str(retained), **item)
    evidence = output / 'continuation'
    evidence.mkdir(exist_ok=True)
    (evidence / 'parent_archive_receipt.json').write_bytes(payload)
    write_json(evidence / 'retained_files.json', mapping)
    # Validate the completed cohort before launching an evaluator.
    load_parent(output / 'evaluation', protocol)


def load_parent(evaluation, protocol):
    if 'continuation' not in protocol:
        return [], []
    amendment = protocol['continuation']
    parent = evaluation.parent / 'continuation/parent'
    old_protocol = json.loads((parent / 'protocol.json').read_text())
    # Only the explicit amendment may change; original policy settings stay fixed.
    if old_protocol != {k:v for k,v in protocol.items() if k != 'continuation'}:
        raise ValueError('Continuation changed the original experiment protocol')
    summary = json.loads((parent / 'evaluation/summary.json').read_text())
    if [list(trial_key(r)) for r in summary['episodes']] != amendment['retained_completed_trials']:
        raise ValueError('Completed parent cohort differs from registered continuation')
    pending_trials(protocol['cases'], summary['episodes'])
    rows = copy.deepcopy(summary['episodes'])
    for row in rows:
        folder = evaluation / row['case'] / f"{row['arm']}{row['attempt']}"
        if json.loads((folder / 'episode.json').read_text()) != row:
            raise ValueError('Parent episode and parent summary disagree')
        row.update(origin_workflow=amendment['parent_workflow'],
                   origin_source_revision=amendment['parent_source_revision'])
    resets = copy.deepcopy(summary['resets'])
    for row in resets:
        row['origin_workflow'] = amendment['parent_workflow']
    return rows, resets


def load_anchor(root):
    with gzip.open(root / 'anchor_model.xml.gz', 'rt') as stream:
        xml = stream.read()
    with np.load(root / 'anchor_state.npz', allow_pickle=False) as archive:
        state = archive['integration_state']
    rng = json.loads((root / 'anchor_rng.json').read_text())
    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, list) else value
    anchor = dict(xml=xml, integration_state=state, environment_rng=rng['environment_rng'],
        numpy_rng=(rng['numpy_rng'][0], np.asarray(rng['numpy_rng'][1], np.uint32), *rng['numpy_rng'][2:]),
        python_rng=tuples(rng['python_rng']))
    return anchor, json.loads((root / 'anchor.json').read_text())
