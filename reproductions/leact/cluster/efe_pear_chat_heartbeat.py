"""Queue an hourly progress check into the user's existing Codex conversation."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import plistlib
import re
import shlex
import shutil
import subprocess
import time

APP=Path.home()/'Library/Application Support/GEAR/pear-chat-heartbeat'
LABEL='com.rpunamiya.pear-chat-heartbeat'


def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.chmod(0o600);temporary.replace(path)


def install():
    workspace=Path(__file__).resolve().parents[3];thread=os.environ['CODEX_THREAD_ID']
    if not re.fullmatch(r'[0-9a-f-]{36}',thread):raise ValueError('Expected the current conversation UUID')
    binary=shutil.which('codex')
    if not binary:raise RuntimeError('Codex CLI is not installed')
    prompt=('Hourly PEAR progress heartbeat requested by the user. In '+str(workspace)+
        ', read reproductions/leact/AGENTS.md and source emimic/bin/activate. Check the current primary-launch.json '
        'in reproductions/leact/artifacts/embodied-first-exam/pear-scaled-v1, live OSMO child jobs and telemetry. '
        'Post a concise timestamped update here: completed episodes/successes, usable groups versus target, actual optimizer '
        'updates, checkpoints/evals, blockers/recovery and revised ETA; say when unchanged. Keep prior-run results separate. '
        'Continue authorized recovery inside retained allocations; never cancel allocations or silently change recipes. '
        'Respect later user instructions. Do not create another heartbeat. Dashboard: http://rpunamiya.nvidia.com:8126/.')
    APP.mkdir(mode=0o700,parents=True,exist_ok=True)
    config=APP/'config.json';plist=Path.home()/'Library/LaunchAgents'/(LABEL+'.plist')
    if plist.exists()and not(APP/'installed.json').exists():raise RuntimeError('Refusing to overwrite an unmanaged LaunchAgent')
    write(config,{'enabled':True,'thread_id':thread,'workspace':str(workspace),'codex':binary,
        'interval_seconds':3600,'prompt':prompt,'user_requested':True})
    shutil.copyfile(__file__,APP/'heartbeat.py')
    runner=APP/'run.sh';runner.write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(workspace))+
        '\nsource emimic/bin/activate\nexec python -u '+shlex.quote(str(APP/'heartbeat.py'))+' --run\n');runner.chmod(0o700)
    spec={'Label':LABEL,'ProgramArguments':['/bin/bash',str(runner)],'WorkingDirectory':str(workspace),
        'StartInterval':3600,'RunAtLoad':False,'ThrottleInterval':60,
        'StandardOutPath':str(APP/'heartbeat.log'),'StandardErrorPath':str(APP/'heartbeat.log')}
    loaded=subprocess.run(['launchctl','print',f'gui/{os.getuid()}/{LABEL}'],capture_output=True).returncode==0
    if loaded:subprocess.run(['launchctl','bootout',f'gui/{os.getuid()}/{LABEL}'],check=True,capture_output=True)
    plist.write_bytes(plistlib.dumps(spec));plist.chmod(0o600)
    subprocess.run(['launchctl','bootstrap',f'gui/{os.getuid()}',str(plist)],check=True,capture_output=True)
    subprocess.run(['launchctl','print',f'gui/{os.getuid()}/{LABEL}'],check=True,capture_output=True)
    value={'installed':True,'thread_id':thread,'interval_seconds':3600,'plist':str(plist),
        'installed_utc':datetime.now(timezone.utc).isoformat(),'runtime':'local Mac; requires wakefulness and Codex service'}
    write(APP/'installed.json',value);print(json.dumps(value))


def run():
    with(APP/'heartbeat.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=json.loads((APP/'config.json').read_text());state_path=APP/'state.json'
        state=json.loads(state_path.read_text())if state_path.exists()else{}
        if not config.get('enabled'):return
        now=time.time()
        if now-state.get('last_attempt_unix',0)<.95*config['interval_seconds']:
            print(json.dumps({'skipped':'already attempted this hour'}));return
        # Persist intent before queueing; an uncertain acknowledgement must not
        # produce duplicate prompts when launchctl is kicked again.
        state.update(last_attempt_unix=now,phase='queueing');write(state_path,state)
        try:
            result=subprocess.run([config['codex'],'queue','--thread',config['thread_id'],'--message',
                config['prompt'],'--resume'],cwd=config['workspace'],capture_output=True,text=True,timeout=60)
            state.update(phase='queued'if result.returncode==0 else'queue_failed',returncode=result.returncode,
                output=result.stdout[-1200:],warning=result.stderr[-1200:])
        except subprocess.TimeoutExpired:state.update(phase='queue_acknowledgement_unknown')
        write(state_path,state);print(json.dumps(state))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.install:install()
    elif args.run:run()
    else:parser.error('Choose --install or --run')


if __name__=='__main__':main()
