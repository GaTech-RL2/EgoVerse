import json
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from reproductions.leact.tracking.server import ThreadingHTTPServer, handler, ingest, ingest_records, latest, trace_api


def test_snapshot_import_is_idempotent_and_rejects_nonfinite_metrics(tmp_path):
    db = tmp_path/'tracker.sqlite3'
    value = {'schema_version': 1, 'runs': [], 'training': [{'loss': 0.04}]}
    first = ingest(db, value)
    assert ingest(db, value) == first
    with pytest.raises(ValueError):
        ingest(db, {**value, 'loss': float('nan')})
    assert latest(db)['training'][0]['loss'] == 0.04
    assert latest(db)['storage']['snapshot_id'] == 1


def test_snapshot_compression_retention_and_legacy_payloads(tmp_path):
    from reproductions.leact.tracking import server
    db=tmp_path/'tracker.sqlite3'
    for i in range(server.SNAPSHOT_RETENTION+5):
        ingest(db,{'schema_version':1,'runs':[],'training':[{'step':i}]})
    with server.connect(db) as connection:
        assert connection.execute('SELECT COUNT(*) FROM snapshots').fetchone()[0]==server.SNAPSHOT_RETENTION
        row=connection.execute('SELECT id,payload FROM snapshots ORDER BY id DESC LIMIT 1').fetchone()
        assert isinstance(row[1],bytes)
        connection.execute('UPDATE snapshots SET payload=? WHERE id=?',(server.unpack_payload(row[1]),row[0]))
    assert latest(db)['training'][0]['step']==server.SNAPSHOT_RETENTION+4


def test_compaction_keeps_all_trace_versions_and_checksums(tmp_path):
    from reproductions.leact.tracking.compact import backup,verify,restore
    # Reuse a fixture that creates two versions and verifies immutable experts.
    test_trace_provenance_join_rollback_and_version_history(tmp_path)
    source=tmp_path/'tracker.sqlite3'
    for i in range(4):ingest(source,{'schema_version':1,'runs':[],'training':[{'step':i}]})
    old=trace_api(source,'/api/trace',{'id':['1']})
    archive=tmp_path/'archive.xz';destination=tmp_path/'compact.sqlite3'
    exported=backup(source,archive,keep=2);checked=verify(archive)
    assert {k:exported[k] for k in checked}==checked
    restored=restore(archive,destination)
    assert {k:restored[k] for k in checked}==checked
    assert checked['counts']=={'snapshots':2,'experts':1,'trace_versions':2,'traces':1}
    assert latest(source)==latest(destination)
    assert trace_api(destination,'/api/trace',{'id':['1']})==old
    assert trace_api(destination,'/api/traces',{'run':['run/seed-0']})['rows'][0]['id']==2
    with pytest.raises(FileExistsError):restore(archive,source)
    archive.write_bytes(archive.read_bytes()[:-20])
    with pytest.raises((EOFError,ValueError)):verify(archive)


def test_export_preserves_training_records_and_links_actual_initial_checkpoint(tmp_path):
    from reproductions.leact.data import write_json
    from reproductions.leact.tracking.export import snapshot
    write_json(tmp_path/'artifacts/coldstart-training-evidence.json', {'global_step': 3})
    stage = tmp_path/'cluster/.build/example/collected/latest/leact/round-2/sft'
    stage.mkdir(parents=True)
    (stage/'train.yaml').write_text('model_name_or_path: /measured/round-1/checkpoint-256\nnum_train_epochs: 3\n')
    write_json(stage/'training_complete.json', {'complete': True})
    log = stage/'checkpoints/trainer_log.jsonl'
    log.parent.mkdir()
    log.write_text(json.dumps({'loss': .2, 'current_steps': 600, 'total_steps': 600})+'\n')
    value = snapshot(tmp_path)
    assert isinstance(value['training'], list)
    assert value['training'][0]['steps'] == 3
    row = value['status']['training_lineage'][0]
    assert row['initial_checkpoint'] == '/measured/round-1/checkpoint-256'
    assert row['stage'] == 'leact/round-2'
    assert row['complete'] and row['total_steps'] == 600


def test_trace_provenance_join_rollback_and_version_history(tmp_path):
    db = tmp_path/'tracker.sqlite3'
    expert = {'kind': 'expert', 'oracle_sha256': 'oracle', 'state_id': 's1',
              'state': {'info_state': 'player 0', 'expert_policy': {'call': 1.0}}}
    trace = {'kind': 'trace', 'run': 'run/seed-0', 'state_id': 's1', 'seed': 0,
             'oracle_sha256': 'oracle', 'label': 'fixture', 'method': 'leact', 'scope': 'diagnostic',
             'candidates': [{'selected': True, 'complete': True, 'reasoning': '<script>literal model output</script>'}]}
    with pytest.raises(ValueError, match='matching imported expert'):
        ingest_records(db, [trace])
    first = ingest_records(db, [expert, trace])
    assert ingest_records(db, [expert, trace]) == first
    changed = {**expert, 'state': {'info_state': 'changed'}}
    with pytest.raises(ValueError, match='immutable'):
        ingest_records(db, [changed])
    groups = trace_api(db, '/api/traces/groups', {})
    assert groups['expert_states'] == 1 and groups['groups'][0]['states'] == 1
    old = trace_api(db, '/api/trace', {'id': ['1']})
    assert old['expert']['state']['expert_policy'] == {'call': 1.0}
    assert '<script>' in old['candidates'][0]['reasoning']
    ingest_records(db, [{**trace, 'candidates': [{'selected': False, 'complete': False}]}])
    assert trace_api(db, '/api/traces', {'run': ['run/seed-0']})['rows'][0]['id'] == 2
    assert trace_api(db, '/api/trace', {'id': ['1']}) == old
    assert trace_api(db, '/api/traces', {'run': ["' OR 1=1 --"]})['total'] == 0


def test_http_exposes_only_known_assets_and_read_endpoints(tmp_path):
    db = tmp_path/'tracker.sqlite3'
    ingest(db, {'schema_version': 1, 'runs': [], 'training': []})
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler(db))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(root+'/api/health') as response:
            assert json.load(response)['service'] == 'leact-tracker'
            assert "script-src 'self'" in response.headers['Content-Security-Policy']
        for path in ['/../server.py', '/tracker.sqlite3', '/api/trace?id=bad']:
            with pytest.raises(HTTPError) as error:
                urlopen(root+path)
            assert error.value.code in {400, 404}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_collector_streams_large_shards_and_preserves_incomplete_tail(tmp_path, monkeypatch, capsys):
    import base64
    import hashlib
    import time
    from reproductions.leact.cluster.collect_results import REMOTE
    path = tmp_path/'eval/checkpoint-1/seed-0/shard-00.jsonl'
    path.parent.mkdir(parents=True)
    complete = b''.join((json.dumps({'state_id': str(i), 'text': 'x'*25000})+'\n').encode() for i in range(100))
    path.write_bytes(complete+b'{"state_id":"still-writing"')
    monkeypatch.setattr(time, 'sleep', lambda seconds: None)
    exec(REMOTE, {'ROOT': str(tmp_path), 'DURATION': 0, 'INTERVAL': 1,
                  'GATE_ONLY': False, 'PROMPT_DIAGNOSTIC': False, 'INCLUDE_ROLLOUTS': True})
    chunks = []
    for line in capsys.readouterr().out.splitlines():
        if not line.startswith('LEACT_FILE '):
            continue
        value = json.loads(line.removeprefix('LEACT_FILE '))
        data = base64.b64decode(value['base64'])
        assert hashlib.sha256(data).hexdigest() == value['sha256']
        assert '.jsonl.parts/' in value['relative']
        chunks.append((value['relative'], data))
    assert len(chunks) > 1
    assert b''.join(data for _, data in sorted(chunks)) == complete


def test_reproduction_collector_keeps_metrics_after_preview_limit(tmp_path, monkeypatch, capsys):
    import base64
    import time
    from reproductions.leact.cluster.collect_results import REMOTE
    directory = tmp_path/'leact/leact/round-1/generation'
    directory.mkdir(parents=True)
    row = (json.dumps({'state_id': 's', 'text': 'x'*25000})+'\n').encode()
    (directory/'shard-00.jsonl').write_bytes(row*100)
    progress = tmp_path/'leact/progress.json'
    progress.write_text('{"phase":"reasoning_sft"}')
    monkeypatch.setattr(time, 'sleep', lambda seconds: None)
    exec(REMOTE, {'ROOT': str(tmp_path), 'DURATION': 0, 'INTERVAL': 1,
                  'GATE_ONLY': False, 'PROMPT_DIAGNOSTIC': False, 'INCLUDE_ROLLOUTS': True,
                  'REPRODUCTION_ROLE': 'leact', 'MAX_ROLLOUT_BYTES': 500_000})
    records = [json.loads(line.removeprefix('LEACT_FILE '))
               for line in capsys.readouterr().out.splitlines() if line.startswith('LEACT_FILE ')]
    assert any(r['relative'] == 'leact/progress.json' for r in records)
    raw = b''.join(base64.b64decode(r['base64']) for r in records if '.parts/' in r['relative'])
    assert 500_000 <= len(raw) < 500_000+len(row)
    assert raw.endswith(b'\n')


def test_publisher_prunes_buffers_only_after_a_matching_receipt(tmp_path, monkeypatch):
    import hashlib
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from reproductions.leact.tracking import server, sync
    expert = {'kind': 'expert', 'oracle_sha256': 'oracle', 'state_id': 's1',
              'state': {'info_state': 'fixture', 'expert_policy': {'call': 1.0}}}
    root = tmp_path/'collected/latest'
    chunk = root/'eval/shard-00.jsonl.parts/000000000000.jsonl'
    chunk.parent.mkdir(parents=True)
    content = b'{"state_id":"s1"}\n'
    digest = hashlib.sha256(content).hexdigest()
    history = root.parent/'history'/digest/chunk.relative_to(root)
    history.parent.mkdir(parents=True)
    chunk.write_bytes(content)
    history.write_bytes(content)
    def records(*, observed_chunks):
        yield expert
        observed_chunks.append((chunk, digest))
    monkeypatch.setattr(sync, 'records', records)
    db = tmp_path/'remote.sqlite3'
    monkeypatch.setattr(sync, 'ssh_command', lambda args, mode: [sys.executable, server.__file__,
        'import-records', '--db', str(db), '--snapshot', '-'])
    args = SimpleNamespace(state=tmp_path/'state.json', host='fixture', prune_uploaded_chunks=True)
    sync.publish_records(args)
    assert not chunk.exists() and not history.exists()
    assert server.trace_api(db, '/api/traces/groups', {})['expert_states'] == 1
    # A wrong acknowledgment must preserve the transfer buffer for a retry,
    # even when the remote database may already have received the record.
    chunk.write_bytes(content)
    args.host = 'another-destination'
    real_run = sync.subprocess.run
    def mismatched_receipt(*args, **kwargs):
        result = real_run(*args, **kwargs)
        value = json.loads(result.stdout)
        value['batch_sha256'] = 'wrong'
        result.stdout = json.dumps(value).encode()
        return result
    monkeypatch.setattr(sync.subprocess, 'run', mismatched_receipt)
    with pytest.raises(ValueError, match='checksum'):
        sync.publish_records(args)
    assert chunk.read_bytes() == content
