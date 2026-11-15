"""Install the local read-only dashboard in macOS application-owned storage."""
from contextlib import closing
import fcntl
import json
import os
from pathlib import Path
import plistlib
import shutil
import sqlite3
import subprocess
import sys

APP = Path.home()/'Library/Application Support/GEAR/leact-tracker'
SOURCE = Path(__file__).resolve().parent
FILES = ('server.py','index.html','style.css','app.js','leaderboards.js','traces.js')


def atomic_copy(source, destination):
    if not source.is_file():
        return
    if destination.exists() and source.stat().st_mtime_ns == destination.stat().st_mtime_ns and source.stat().st_size == destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+'.sync.tmp')
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def mirror_assets():
    for name in FILES:
        atomic_copy(SOURCE/name, APP/'tracking'/name)


def mirror_viewer(path=None):
    if not (APP/'installed.json').exists():
        return
    path = path or SOURCE.parent/'artifacts/embodied-first-exam/efe-rollouts.html'
    path = Path(path)
    if path.name not in {'efe-rollouts.html','efe-direct-control.html'}:
        return
    with (APP/'mirror-viewer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        atomic_copy(path, APP/'artifacts/embodied-first-exam'/path.name)


def mirror_snapshot(value):
    if not (APP/'installed.json').exists():
        return
    from .server import ingest
    with (APP/'mirror-snapshot.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        mirror_assets()
        ingest(APP/'experiments.sqlite3', value)


def install():
    APP.mkdir(parents=True, exist_ok=True)
    mirror_assets()
    source = SOURCE.parent/'cluster/.build/tracking/experiments.sqlite3'
    destination = APP/'experiments.sqlite3'
    if not destination.exists():
        temporary = APP/'experiments.sqlite3.backup.tmp'
        with closing(sqlite3.connect(source)) as src, closing(sqlite3.connect(temporary)) as dst:
            src.backup(dst, pages=4096)
        temporary.replace(destination)
    (APP/'installed.json').write_text(json.dumps({'source':str(SOURCE), 'storage':str(APP)},indent=2)+'\n')
    mirror_viewer()
    # The base interpreter lives outside Desktop; only stdlib is needed.
    interpreter = str(Path(sys._base_executable).resolve())
    if '/Desktop/' in interpreter:
        raise ValueError('LaunchAgent needs an interpreter outside protected Desktop storage')
    label = 'com.rpunamiya.leact-local-dashboard'
    agent = Path.home()/'Library/LaunchAgents'/f'{label}.plist'
    config = {'Label':label,'ProgramArguments':[interpreter,str(APP/'tracking/server.py'),'serve',
        '--db',str(destination),'--host','127.0.0.1','--port','8126'],
        'WorkingDirectory':str(APP),'RunAtLoad':True,'KeepAlive':True,'ThrottleInterval':10,
        'StandardOutPath':str(APP/'server.log'),'StandardErrorPath':str(APP/'server.log')}
    agent.parent.mkdir(parents=True,exist_ok=True)
    agent.write_bytes(plistlib.dumps(config))
    print(json.dumps({'launch_agent':str(agent),'application_storage':str(APP),'interpreter':interpreter}))


if __name__ == '__main__':
    install()
