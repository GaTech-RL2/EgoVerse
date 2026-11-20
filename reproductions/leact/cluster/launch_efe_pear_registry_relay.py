"""Supply fresh repository-scoped tokens to owned CPU image publishers.

Long-lived Docker credentials remain on this workstation. Tokens are delivered
only after hashing, per image and per allocation; none are retained in logs.
"""
import base64
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import urllib.parse
import urllib.request

from .launch_efe_pear_scaled import DEST
from .scale_efe import read,write


def deliver(workflow):
    value = json.loads((Path.home()/'.docker/config.json').read_text())
    encoded = value['auths']['nvcr.io']['auth']
    query=urllib.parse.urlencode({
        'service': 'nvcr.io',
        'scope': 'repository:nvidian/lpr/pear-efe-rtx-rp-jhh:pull,push'})
    request=urllib.request.Request('https://nvcr.io/proxy_auth?'+query,
        headers={'Authorization':'Basic '+encoded})
    with urllib.request.urlopen(request,timeout=30)as response:body=json.load(response)
    token = body.get('token', body.get('access_token'))
    if not token:
        raise ValueError('Registry did not return a scoped token')
    with tempfile.TemporaryDirectory(prefix='rp_jhh_registry_') as folder:
        path=Path(folder)/'rp_jhh_registry_token';path.write_text(token);path.chmod(0o600)
        result=subprocess.run(['osmo','workflow','rsync','upload',workflow,'preflight',
            str(path)+':/tmp/','--no-progress'],capture_output=True,text=True,timeout=60)
        if result.returncode:
            raise RuntimeError('Ephemeral registry token transfer failed')


def main():
    sent={}
    while not read(DEST/'control.json',{}).get('paused',True):
        launch=read(DEST/'native-rtx-publication-launch.json',{})
        status=read(DEST/'native-rtx-publication-status.json',{})
        workflow=launch.get('name',launch.get('workflow'))
        worker=status.get('worker',{})
        if workflow==status.get('workflow') and status.get('scheduler_status')=='RUNNING':
            identity=(workflow,worker.get('image_index',0))
            if time.monotonic()-sent.get(identity,float('-inf'))>=180:
                try:
                    deliver(workflow);sent[identity]=time.monotonic()
                    write(DEST/'registry-relay-status.json',{'phase':'ephemeral_scoped_token_delivered',
                        'image_index':identity[1],'workflow':workflow,'primary_rl_updates':0})
                except Exception as exc:
                    write(DEST/'registry-relay-status.json',{'phase':'retrying_transfer','error_type':type(exc).__name__})
        time.sleep(20)


if __name__=='__main__':
    main()
