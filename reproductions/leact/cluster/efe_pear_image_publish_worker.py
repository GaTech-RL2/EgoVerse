"""CPU-only private image publication, with an ephemeral token supplied later."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from efe_pear_registry_http import publish,RegistryError


def main():
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    output=Path(spec['output']);output.mkdir(parents=True,exist_ok=True)
    def record(phase,**extra):
        value={'phase':phase,'primary_rl_updates':0,'gpu':0,**extra}
        (output/'status.json').write_text(json.dumps(value)+'\n');print('PEAR_NATIVE_IMAGE_PUBLISH '+json.dumps(value),flush=True)
    receipts=[]
    def image_copy(pair):
        index,image=pair
        record('verifying_cached_archive',image_index=index)
        archive=Path(image['archive']);before=archive.stat();digest=hashlib.sha256()
        with archive.open('rb')as stream:
            while block:=stream.read(16<<20):digest.update(block)
        if digest.hexdigest()!=image['archive_sha256']:raise ValueError('Cached simulator OCI checksum changed')
        # Hash large local archives BEFORE requesting the short-lived token,
        # and request a fresh token for each image, including allocation retries.
        temporary=Path('/tmp/rp_jhh_registry_token');deadline=time.monotonic()+1800
        record('awaiting_ephemeral_scoped_token',image_index=index)
        while not temporary.exists():
            if time.monotonic()>deadline:raise TimeoutError('No short-lived registry token supplied')
            time.sleep(1)
        after=archive.stat()
        if(before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Cached archive changed after hashing')
        receipt=Path(image.get('output',output))/ 'registry-image.json'
        record('copying_verified_archive',image_index=index)
        for attempt in range(3):
            try:
                publish(archive,image['archive_sha256'],image['destination'],temporary,receipt,archive_verified=True)
                break
            except RegistryError:
                if attempt==2:raise
                record('resuming_transient_registry_failure',image_index=index,attempt=attempt+1)
                time.sleep(30*(attempt+1))
        record('private_image_published',receipt=str(receipt))
        return str(receipt)
    with ThreadPoolExecutor(max_workers=2)as executor:
        receipts=list(executor.map(image_copy,enumerate(spec.get('images',[spec]))))
    Path('/tmp/rp_jhh_registry_token').unlink(missing_ok=True)
    record('private_image_publication_completed',receipts=receipts)


if __name__=='__main__':main()
