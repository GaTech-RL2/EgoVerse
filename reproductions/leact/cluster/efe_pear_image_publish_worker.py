"""CPU-only private image publication, with an ephemeral token supplied later."""
import json
import os
from pathlib import Path
import subprocess
import time

from efe_pear_registry_push import publish


def main():
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    output=Path(spec['output']);output.mkdir(parents=True,exist_ok=True)
    def record(phase,**extra):
        value={'phase':phase,'primary_rl_updates':0,'gpu':0,**extra}
        (output/'status.json').write_text(json.dumps(value)+'\n');print('PEAR_NATIVE_IMAGE_PUBLISH '+json.dumps(value),flush=True)
    record('installing_private_registry_copy_tool')
    with (output/'install.log').open('w')as log:
        subprocess.run(['apt-get','update','-qq'],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        subprocess.run(['apt-get','install','-y','-qq','skopeo'],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    record('awaiting_ephemeral_scoped_token')
    token=Path('/tmp/rp_jhh_registry_token');deadline=time.monotonic()+1800
    while not token.exists():
        if time.monotonic()>deadline:raise TimeoutError('No short-lived registry token supplied')
        time.sleep(1)
    credential=token.read_bytes();token.unlink()
    receipts=[]
    for index,image in enumerate(spec.get('images',[spec])):
        temporary=Path('/tmp/rp_jhh_registry_token_'+str(index))
        descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'wb')as stream:stream.write(credential)
        receipt=Path(image.get('output',output))/ 'registry-image.json'
        try:publish(image['archive'],image['archive_sha256'],image['destination'],temporary,receipt)
        finally:temporary.unlink(missing_ok=True)
        receipts.append(str(receipt));record('private_image_published',receipt=str(receipt))
    record('private_image_publication_completed',receipts=receipts)


if __name__=='__main__':main()
