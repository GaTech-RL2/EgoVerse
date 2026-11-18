"""Read-only experiment dashboard with idempotent, SSH-driven SQLite imports."""
from __future__ import annotations

import argparse
import base64
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sqlite3
import sys
import zlib
from urllib.parse import parse_qs, urlparse

ASSETS = Path(__file__).resolve().parent
EFE_ROLLOUT_VIEWER = ASSETS.parent / 'artifacts/embodied-first-exam/efe-rollouts.html'
REASONING_MEDIA = ASSETS/'reasoning-media'
if not REASONING_MEDIA.is_dir():REASONING_MEDIA=ASSETS.parent/'artifacts/reasoning-workspace/media'
COMPRESSED_PAYLOAD = b'LEACT_ZLIB_V1\0'
SNAPSHOT_RETENTION = 32


def pack_payload(payload):
    return COMPRESSED_PAYLOAD + zlib.compress(payload.encode())


def unpack_payload(payload):
    if isinstance(payload, bytes):
        if payload.startswith(COMPRESSED_PAYLOAD):
            return zlib.decompress(payload[len(COMPRESSED_PAYLOAD):]).decode()
        return payload.decode()
    return payload


def connect(path):
    db = sqlite3.connect(path, timeout=15)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS snapshots (id INTEGER PRIMARY KEY, sha256 TEXT UNIQUE NOT NULL, imported_utc TEXT NOT NULL, payload TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS pear_runs (run_id TEXT PRIMARY KEY, updated_utc TEXT NOT NULL, payload TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS experts (oracle_sha TEXT, state_id TEXT, info TEXT, sha256 TEXT, payload TEXT, PRIMARY KEY(oracle_sha,state_id))')
    db.execute('CREATE TABLE IF NOT EXISTS trace_versions (id INTEGER PRIMARY KEY, sha256 TEXT UNIQUE, payload TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS traces (run TEXT, state_id TEXT, seed INTEGER, oracle_sha TEXT, label TEXT, method TEXT, scope TEXT, candidates INTEGER, selected INTEGER, truncated INTEGER, version_id INTEGER, PRIMARY KEY(run,state_id,seed))')
    db.commit()
    return db


def public_reasoning(value):
    private=lambda text:re.search('leact|yam-coverage-rpunamiya',text,re.I)is not None
    if isinstance(value,dict):return {k:public_reasoning(v)for k,v in value.items()
        if not private(k)and not(isinstance(v,str)and private(v))}
    if isinstance(value,list):return [public_reasoning(v)for v in value if not(isinstance(v,str)and private(v))]
    return value


def ingest_pear(path,value):
    workspace=latest(path).get('status',{}).get('reasoning_workspace',{})
    allowed={r['id']for r in workspace.get('pear',{}).get('arms',[])}
    run_id=value.get('run_id');data=value.get('data')
    if value.get('schema_version')!=1 or run_id not in allowed or not isinstance(data,dict):
        raise ValueError('Only registered PEAR runs may upload telemetry')
    fields={'history','latest_metrics','optimizer_steps_observed','latest_task_rewards','evaluations',
            'checkpoints','recent_graded_groups','heartbeat','metadata','rollout_diagnostics'}
    if data.keys()-fields:raise ValueError('Telemetry cannot modify catalog or run registration')
    data=public_reasoning(data);json.dumps(data,allow_nan=False)
    if value.get('dry_run'):return {'accepted':True,'dry_run':True,'run_id':run_id}
    now=datetime.now(timezone.utc).isoformat()
    with closing(connect(path))as db,db:
        previous=db.execute('SELECT payload FROM pear_runs WHERE run_id=?',(run_id,)).fetchone()
        old=json.loads(unpack_payload(previous[0]))if previous else {}
        for key in ['history','evaluations','checkpoints']:
            records={row['step']:row for row in [*old.get(key,[]),*data.get(key,[])]}
            if any(type(k)is not int or k<0 for k in records):raise ValueError('Invalid optimizer step')
            data[key]=[row for _,row in sorted(records.items())][-512:]
        merged={**old,**data}
        db.execute('INSERT OR REPLACE INTO pear_runs VALUES(?,?,?)',(run_id,now,pack_payload(json.dumps(merged,allow_nan=False))))
    return {'accepted':True,'run_id':run_id,'updated_utc':now}


def reasoning_workspace(path):
    value=latest(path).get('status',{}).get('reasoning_workspace',{})
    with closing(connect(path))as db:
        rows=db.execute('SELECT run_id,updated_utc,payload FROM pear_runs').fetchall()
    if rows:
        live=value.setdefault('live_pear',{});live['collector_state']='hosted run history';runs=live.setdefault('runs',{})
        for run_id,updated,payload in rows:
            if run_id=='__retention496__':value['warmstart_retention']=json.loads(unpack_payload(payload))
            else:runs[run_id]=json.loads(unpack_payload(payload))
        live['observed_utc']=max(r[1]for r in rows)
    status=Path(path).parent/'collector-status.json'
    if status.is_file():
        try:
            record=json.loads(status.read_text())
            value['hosted_collector']={k:v for k,v in record.items()if k in {'success','updated_utc','read_only','workflows_checked'}}
        except (ValueError,OSError):pass
    return public_reasoning(value)


def ingest(path, value):
    if value.get('schema_version') != 1 or not isinstance(value.get('runs'), list) or not isinstance(value.get('training'), list):
        raise ValueError('Expected a version 1 experiment snapshot')
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    with closing(connect(path)) as db, db:
        db.execute('INSERT OR IGNORE INTO snapshots(sha256, imported_utc, payload) VALUES(?,?,?)',
                   (digest, datetime.now(timezone.utc).isoformat(), pack_payload(payload)))
        row = db.execute('SELECT id, imported_utc FROM snapshots WHERE sha256=?', (digest,)).fetchone()
        # Bound future growth, including when an old database is migrated. Small
        # batches avoid a multi-gigabyte WAL on machines already low on space.
        db.execute('DELETE FROM snapshots WHERE id IN '
                   '(SELECT id FROM snapshots ORDER BY id DESC LIMIT 8 OFFSET ?)',
                   (SNAPSHOT_RETENTION,))
    return {'id': row[0], 'sha256': digest, 'imported_utc': row[1]}


def latest(path):
    with closing(connect(path)) as db:
        row = db.execute('SELECT id, sha256, imported_utc, payload FROM snapshots ORDER BY id DESC LIMIT 1').fetchone()
    if row is None:
        return {'schema_version': 1, 'empty': True}
    value = json.loads(unpack_payload(row[3]))
    value['storage'] = {'snapshot_id': row[0], 'sha256': row[1], 'imported_utc': row[2]}
    return value


def ingest_records(path, records):
    count = 0
    digest = hashlib.sha256()
    with closing(connect(path)) as db, db:
        for value in records:
            payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
            sha = hashlib.sha256(payload.encode()).hexdigest()
            digest.update((sha+'\n').encode())
            if value['kind'] == 'expert':
                key = (value['oracle_sha256'], value['state_id'])
                previous = db.execute('SELECT sha256 FROM experts WHERE oracle_sha=? AND state_id=?', key).fetchone()
                if previous and previous[0] != sha:
                    raise ValueError('An immutable expert state changed under the same oracle checksum')
                db.execute('INSERT OR IGNORE INTO experts VALUES(?,?,?,?,?)',
                           (*key, value['state']['info_state'], sha, pack_payload(payload)))
            elif value['kind'] == 'trace':
                key = (value['oracle_sha256'], value['state_id'])
                if not db.execute('SELECT 1 FROM experts WHERE oracle_sha=? AND state_id=?', key).fetchone():
                    raise ValueError('Trace has no matching imported expert state')
                db.execute('INSERT OR IGNORE INTO trace_versions(sha256,payload) VALUES(?,?)', (sha, pack_payload(payload)))
                version = db.execute('SELECT id FROM trace_versions WHERE sha256=?', (sha,)).fetchone()[0]
                candidates = value['candidates']
                db.execute('INSERT OR REPLACE INTO traces VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                           (value['run'], value['state_id'], value['seed'], value['oracle_sha256'],
                            value['label'], value['method'], value['scope'], len(candidates),
                            sum(bool(c['selected']) for c in candidates),
                            sum(not c['complete'] for c in candidates), version))
            else:
                raise ValueError('Unknown record kind')
            count += 1
    return {'records': count, 'batch_sha256': digest.hexdigest()}


def trace_api(database, path, query):
    def arg(key, default=''):
        return query.get(key, [default])[0]
    with closing(connect(database)) as db:
        db.row_factory = sqlite3.Row
        if path == '/api/traces/groups':
            rows = db.execute('SELECT run AS id, label, method, scope, COUNT(*) AS states, '
                              'SUM(candidates) AS candidates, SUM(selected) AS selected, SUM(truncated) AS truncated '
                              'FROM traces GROUP BY run ORDER BY run').fetchall()
            return {'groups': [dict(r) for r in rows],
                    'expert_states': db.execute('SELECT COUNT(*) FROM experts').fetchone()[0]}
        if path == '/api/trace':
            row = db.execute('SELECT payload, sha256 FROM trace_versions WHERE id=?', (int(arg('id')),)).fetchone()
            if not row:
                return None
            value = json.loads(unpack_payload(row['payload']))
            state = db.execute('SELECT payload FROM experts WHERE oracle_sha=? AND state_id=?',
                               (value['oracle_sha256'], value['state_id'])).fetchone()
            value['expert'] = json.loads(unpack_payload(state['payload']))
            value['record_sha256'] = row['sha256']
            return value
        limit, offset = min(100, max(1, int(arg('limit', '30')))), max(0, int(arg('offset', '0')))
        search = '%'+arg('q')[:200]+'%'
        if path == '/api/experts':
            condition = 'WHERE (state_id LIKE ? OR info LIKE ?)'
            params = (search, search)
            total = db.execute('SELECT COUNT(*) FROM experts '+condition, params).fetchone()[0]
            rows = db.execute('SELECT payload FROM experts '+condition+' ORDER BY state_id LIMIT ? OFFSET ?',
                              (*params, limit, offset)).fetchall()
            return {'total': total, 'offset': offset, 'limit': limit, 'rows': [json.loads(unpack_payload(r[0])) for r in rows]}
        condition = 'WHERE run=? AND (t.state_id LIKE ? OR e.info LIKE ?)'
        params = (arg('run'), search, search)
        relation = 'FROM traces t JOIN experts e ON e.oracle_sha=t.oracle_sha AND e.state_id=t.state_id '
        total = db.execute('SELECT COUNT(*) '+relation+condition, params).fetchone()[0]
        rows = db.execute('SELECT t.state_id, e.info, t.candidates, t.selected, t.truncated, t.version_id AS id '
                          +relation+condition+' ORDER BY t.state_id LIMIT ? OFFSET ?', (*params, limit, offset)).fetchall()
        return {'total': total, 'offset': offset, 'limit': limit, 'rows': [dict(r) for r in rows]}


def handler(database,home_page='experiments',*,media_root=None):
    media_root=Path(media_root or REASONING_MEDIA).resolve()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            csp = "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'"
            if home_page=='reasoning' and not (path in {'/','/reasoning','/rpunamiya-reasoning',
                '/reasoning.js','/reasoning.css','/api/reasoning','/api/health'}or path.startswith('/reasoning-media/')):
                self.send_error(404)
                return
            if path in {'/api/traces/groups', '/api/traces', '/api/trace', '/api/experts'}:
                try:
                    value = trace_api(database, path, parse_qs(parsed.query))
                except (ValueError, TypeError):
                    self.send_error(400)
                    return
                if value is None:
                    self.send_error(404)
                    return
                data = json.dumps(value, allow_nan=False).encode()
                content_type = 'application/json'
            elif path=='/api/reasoning':
                value=reasoning_workspace(database)
                data=json.dumps(value,allow_nan=False).encode()
                content_type='application/json'
            elif path in {'/api/snapshot', '/export.json'}:
                data = json.dumps(latest(database), allow_nan=False).encode()
                content_type = 'application/json'
            elif path == '/api/health':
                value = latest(database)
                data = json.dumps({'service': 'reasoning-tracker'if home_page=='reasoning'else'leact-tracker', 'status': 'ok',
                                   'snapshot': value.get('storage'),
                                   'runs': len(value.get('runs', []))}).encode()
                content_type = 'application/json'
            elif path in {'/efe-rollouts.html','/efe-direct-control.html'}:
                viewer = EFE_ROLLOUT_VIEWER.with_name(path.lstrip('/'))
                if not viewer.is_file():
                    self.send_error(404)
                    return
                data = viewer.read_bytes()
                content_type = 'text/html'
                # Permit only the exact embedded code/style in the frozen report.
                def hashes(tag):
                    return ' '.join("'sha256-" + base64.b64encode(hashlib.sha256(block).digest()).decode() + "'"
                        for block in re.findall(rb'<' + tag + rb'>(.*?)</' + tag + rb'>', data, re.S))
                csp = ("default-src 'none'; script-src " + hashes(b'script') +
                       "; style-src " + hashes(b'style') +
                       "; img-src data:; media-src blob: data:; object-src 'none'; "
                       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
            elif path.startswith('/reasoning-media/'):
                name=path.removeprefix('/reasoning-media/')
                source=media_root/name
                if not re.fullmatch(r'[a-f0-9]{64}\.(jpg|png)',name)or not source.is_file()or not source.resolve().is_relative_to(media_root):
                    self.send_error(404)
                    return
                data=source.read_bytes()
                content_type='image/jpeg'if name.endswith('.jpg')else'image/png'
            elif path in {'/', '/experiments', '/app.js', '/traces.js', '/leaderboards.js', '/style.css',
                          '/rpunamiya-reasoning','/reasoning','/reasoning.js','/reasoning.css'}:
                names={'/':'reasoning.html'if home_page=='reasoning'else'index.html',
                    '/experiments':'index.html','/app.js':'app.js','/traces.js':'traces.js',
                    '/leaderboards.js':'leaderboards.js','/style.css':'style.css',
                    '/rpunamiya-reasoning':'reasoning.html','/reasoning':'reasoning.html',
                    '/reasoning.js':'reasoning.js','/reasoning.css':'reasoning.css'}
                name=names[path]
                content_type={'.html':'text/html','.js':'text/javascript','.css':'text/css'}[Path(name).suffix]
                data = (ASSETS / name).read_bytes()
            else:
                self.send_error(404)
                return
            accepts_gzip=any(part.split(';')[0].strip()=='gzip'and
                not re.search(r';\s*q=0(?:\.0*)?(?:;|$)',part)
                for part in self.headers.get('Accept-Encoding','').lower().split(','))
            compressed=home_page=='reasoning'and path=='/api/reasoning'and len(data)>16384 and accepts_gzip
            if compressed:data=gzip.compress(data,compresslevel=3)
            self.send_response(200)
            self.send_header('Content-Type', content_type+(''if content_type.startswith('image/')else'; charset=utf-8'))
            self.send_header('Content-Length', str(len(data)))
            if path=='/api/reasoning':self.send_header('Vary','Accept-Encoding')
            if compressed:self.send_header('Content-Encoding','gzip')
            self.send_header('Cache-Control', 'public, max-age=31536000, immutable'if content_type.startswith('image/')else'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', csp)
            self.end_headers()
            self.wfile.write(data)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['serve', 'import', 'import-records','import-pear'])
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--snapshot', help='JSON file or - for standard input')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8126)
    parser.add_argument('--home-page',choices=['experiments','reasoning'],default='experiments')
    args = parser.parse_args()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    if args.command == 'import-records':
        if not args.snapshot:
            parser.error('import-records requires --snapshot')
        stream = sys.stdin if args.snapshot == '-' else open(args.snapshot)
        def items():
            for line in stream:
                if len(line) > 2_000_000:
                    raise ValueError('One trace record exceeds size limit')
                yield json.loads(line)
        try:
            print(json.dumps(ingest_records(args.db, items())))
        finally:
            if stream is not sys.stdin:
                stream.close()
    elif args.command in {'import','import-pear'}:
        if not args.snapshot:
            parser.error('import requires --snapshot')
        raw = sys.stdin.read(8_000_001) if args.snapshot == '-' else Path(args.snapshot).read_text()
        if len(raw) > 8_000_000:
            raise ValueError('Snapshot exceeds size limit')
        print(json.dumps((ingest_pear if args.command=='import-pear'else ingest)(args.db,json.loads(raw))))
    else:
        connect(args.db).close()
        server = ThreadingHTTPServer((args.host, args.port), handler(args.db,args.home_page))
        print(f'LeAct tracker listening on {args.host}:{args.port}', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
