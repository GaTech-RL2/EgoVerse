"""Configure a scoped upload key and a per-user background publisher on macOS."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import plistlib
import shlex
import subprocess

REMOTE = '''import json, os, pathlib, sys
value = json.load(sys.stdin)
base = pathlib.Path.home()/'leact-tracker'
assert (base/'.leact-tracker-managed').is_file()
receiver = base/'receive.sh'
receiver.write_text(value['receiver'])
receiver.chmod(0o700)
ssh = pathlib.Path.home()/'.ssh'
ssh.mkdir(mode=0o700, exist_ok=True)
authorized = ssh/'authorized_keys'
public = value['public'].strip()
assert public.startswith('ssh-ed25519 ') and '\\n' not in public
entry = 'restrict,command="'+str(receiver)+'" '+public
existing = authorized.read_text().splitlines() if authorized.exists() else []
same_key = [line for line in existing if public.split()[1] in line]
if same_key and same_key != [entry]:
    raise SystemExit('Existing key has a different access restriction; refusing to change it')
if not same_key:
    with authorized.open('a') as stream:
        stream.write('\\n'+entry+'\\n')
authorized.chmod(0o600)
print(json.dumps({'receiver':str(receiver),'restricted_key_installed':True}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host')
    parser.add_argument('--control', type=Path)
    parser.add_argument('--identity', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reasoning-only',action='store_true',help='Use a separate publisher for the reasoning workspace')
    args = parser.parse_args()
    args.host=args.host or ('rpunamiya@rpunamiya.nvidia.com'if args.reasoning_only else 'rpunamiya@yam-coverage-rpunamiya.nvidia.com')
    args.control=args.control or Path('/tmp/'+('reasoning'if args.reasoning_only else 'leact')+'-itss-ssh-rpunamiya/control')
    args.identity=args.identity or Path.home()/('.ssh/reasoning_tracker_itss'if args.reasoning_only else '.ssh/leact_tracker_itss')
    repo = Path(__file__).resolve().parents[3]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.identity.parent.mkdir(mode=0o700, exist_ok=True)
    if not args.identity.exists():
        subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(args.identity),
                        '-C', 'leact-tracker-sync'], check=True)
    public = args.identity.with_suffix('.pub').read_text()
    message = json.dumps({'public': public, 'receiver': Path(__file__).with_name('receive.sh').read_text()})
    ssh = ['ssh', '-S', str(args.control), '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
           '-o', 'ConnectTimeout=10', args.host]
    command = '. "$HOME/leact-tracker/emimic/bin/activate"\npython -c '+shlex.quote(REMOTE)
    result = subprocess.run(ssh+[command], input=message, capture_output=True, text=True, check=True, timeout=30)
    print(result.stdout.strip())
    scoped = ['ssh', '-S', 'none', '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
              '-o', 'ConnectTimeout=10', '-i', str(args.identity), args.host]
    health = json.loads(subprocess.check_output(scoped+['leact-health'], timeout=20))
    denied = subprocess.run(scoped+['uname -a'], capture_output=True, text=True, timeout=20)
    if denied.returncode != 64 or 'only LeAct tracker' not in denied.stderr:
        raise RuntimeError('Scoped key did not enforce the expected command restriction')
    publisher = args.output/'publish.sh'
    state = args.output/'sync-state.json'
    script=('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(repo))+
                         '\nsource emimic/bin/activate\nexec python -m reproductions.leact.tracking.sync '+
                         '--host '+shlex.quote(args.host)+' --identity '+shlex.quote(str(args.identity))+
                         ' --state '+shlex.quote(str(state)))
    if args.reasoning_only:
        script=script.replace('exec python -m reproductions.leact.tracking.sync',
            'python -m reproductions.leact.tracking.reasoning_live\npython -m reproductions.leact.tracking.reasoning\nexec python -m reproductions.leact.tracking.sync')
        script+=' --reasoning-only\n'
    else:script+=' --watch --interval 30 --prune-uploaded-chunks\n'
    publisher.write_text(script)
    publisher.chmod(0o700)
    label = 'com.rpunamiya.reasoning-tracker-sync'if args.reasoning_only else 'com.rpunamiya.leact-tracker-sync'
    plist = Path.home()/'Library/LaunchAgents'/f'{label}.plist'
    plist.parent.mkdir(parents=True, exist_ok=True)
    spec = {'Label': label, 'ProgramArguments': ['/bin/bash', str(publisher)], 'RunAtLoad': True,
            'KeepAlive': True, 'ThrottleInterval': 15 if args.reasoning_only else 30, 'WorkingDirectory': str(repo),
            'StandardOutPath': str(args.output/'sync.log'), 'StandardErrorPath': str(args.output/'sync-error.log'),
            'EnvironmentVariables': {'PATH': '/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'}}
    plist.write_bytes(plistlib.dumps(spec))
    domain = f'gui/{os.getuid()}'
    subprocess.run(['launchctl', 'bootout', domain+'/'+label], capture_output=True)
    subprocess.run(['launchctl', 'bootstrap', domain, str(plist)], check=True)
    record = {'host': args.host, 'key': str(args.identity), 'key_scope': 'forced tracker import commands; no shell/forwarding',
              'restricted_command_test': 'passed', 'health': health, 'launch_agent': str(plist),
              'label': label, 'publisher': str(publisher), 'interval_seconds': 15 if args.reasoning_only else 30,
              'rollout_buffers': 'not exported'if args.reasoning_only else 'pruned only after checksum-acknowledged import',
              'logging_requires': 'Hosted read-only OSMO collector; Mac publisher refreshes the catalog only'if args.reasoning_only else
                                  'Mac publisher and the OSMO result collector must be running; dashboard remains hosted on ITSS'}
    (args.output/'sync-configuration.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
