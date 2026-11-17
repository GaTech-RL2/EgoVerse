"""Package, checksum, deploy and verify the isolated LeAct service on ITSS."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from .export import snapshot
from .traces import records

ASSETS = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host')
    parser.add_argument('--control', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--package-only', action='store_true')
    parser.add_argument('--home-page',choices=['experiments','reasoning'])
    args = parser.parse_args()
    reasoning=args.home_page=='reasoning'
    args.host=args.host or ('rpunamiya@rpunamiya.nvidia.com'if reasoning else 'rpunamiya@yam-coverage-rpunamiya.nvidia.com')
    args.control=args.control or Path('/tmp/'+('reasoning'if reasoning else 'leact')+'-itss-ssh-rpunamiya/control')
    args.output.mkdir(parents=True, exist_ok=True)
    data = snapshot(reasoning_only=args.home_page=='reasoning')
    snapshot_path = args.output/'snapshot.json'
    snapshot_path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    records_path = args.output/'records.jsonl'
    counts = {'expert': 0, 'trace': 0}
    with records_path.open('w') as stream:
        for record in ([]if args.home_page=='reasoning'else records()):
            stream.write(json.dumps(record, allow_nan=False)+'\n')
            counts[record['kind']] += 1
    archive = args.output/'tracker.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for name in ['server.py', 'index.html', 'app.js', 'traces.js', 'leaderboards.js', 'style.css',
                     'reasoning.html','reasoning.js','reasoning.css','install_remote.sh']:
            bundle.add(ASSETS/name, arcname=name)
        bundle.add(snapshot_path, arcname='snapshot.json')
        bundle.add(records_path, arcname='records.jsonl')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    record = {'host': args.host, 'archive': str(archive), 'sha256': digest,
              'bytes': archive.stat().st_size, 'runs': len(data['runs']),
              'audit_rows': len(data.get('audit',{}).get('rows',[])), 'records': counts, 'deployed': False}
    (args.output/'deployment.json').write_text(json.dumps(record, indent=2)+'\n')
    if args.package_only:
        print(json.dumps(record))
        return
    ssh = ['ssh', '-S', str(args.control), '-o', 'BatchMode=yes',
           '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10', args.host]
    preparation = '''set -eu
umask 077
base="$HOME/leact-tracker"
if [ -e "$base" ] && [ ! -f "$base/.leact-tracker-managed" ]; then
  echo 'Refusing to replace an existing unmanaged directory' >&2
  exit 1
fi
mkdir -p "$base/releases"
touch "$base/.leact-tracker-managed"
cat > "$base/incoming.tar.gz"
'''
    subprocess.run(ssh+[preparation], input=archive.read_bytes(), check=True, timeout=45)
    installation = '''set -eu
umask 077
base="$HOME/leact-tracker"
cd "$base"
printf '%s  incoming.tar.gz\\n' DIGEST | sha256sum -c -
mkdir -p "releases/DIGEST"
tar -xzf incoming.tar.gz -C "releases/DIGEST"
bash "releases/DIGEST/install_remote.sh" "$base/releases/DIGEST"
'''.replace('DIGEST', digest)
    if args.home_page:
        installation='export LEACT_TRACKER_HOME_PAGE='+args.home_page+'\n'+installation
    result = subprocess.run(ssh+[installation], text=True, capture_output=True, timeout=90)
    (args.output/'install.log').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError(f'ITSS installation exited {result.returncode}; inspect {args.output}/install.log')
    health_result = subprocess.run(ssh+['curl --fail --silent http://127.0.0.1:8126/api/health'],
                                   capture_output=True, text=True, timeout=20, check=True)
    health = json.loads(health_result.stdout)
    canonical = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if health.get('service') != 'leact-tracker' or health['snapshot']['sha256'] != hashlib.sha256(canonical).hexdigest():
        raise ValueError('ITSS health or imported snapshot checksum did not match')
    record.update(deployed=True, verified_utc=datetime.now(timezone.utc).isoformat(), health=health,
                  url='http://'+args.host.split('@')[-1]+':8126/')
    (args.output/'deployment.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
