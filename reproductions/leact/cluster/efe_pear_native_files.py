"""Fetch checksum-pinned public EFE files images without a container daemon."""
import argparse
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import re
import tarfile
import urllib.parse
import urllib.request


def unpack_files(raw,output):
    output=Path(output);files={};total=0
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:*')as archive:
        for entry in archive:
            name=PurePosixPath(entry.name.removeprefix('./'))
            if name.is_absolute()or'..'in name.parts:raise ValueError('Unsafe native files image path')
            if entry.isdir():continue
            if not entry.isfile()or any(part.startswith('.wh.')for part in name.parts):
                raise ValueError('Native files images must contain ordinary files only')
            total+=entry.size
            if total>64<<20:raise ValueError('Not a bounded files-only image')
            data=archive.extractfile(entry).read();target=output/str(name)
            if target.exists()and target.read_bytes()!=data:raise ValueError('Conflicting native files image layers')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            target.chmod(0o755 if entry.mode&0o111 else 0o644)
            files[str(name)]=hashlib.sha256(data).hexdigest()
    return files


def fetch(image,output):
    match=re.fullmatch(r'ghcr.io/(mll-lab-nu/efe)(?::[^@]+)?@sha256:([0-9a-f]{64})',image)
    if not match:raise ValueError('Expected a pinned public native EFE files image')
    repository,digest=match.groups();output=Path(output)
    if(output/'files-manifest.json').exists():
        saved=json.loads((output/'files-manifest.json').read_text())
        if saved['image']!=image:raise ValueError('Different image already occupies native files cache')
        for name,expected in saved['files'].items():
            if hashlib.sha256((output/name).read_bytes()).hexdigest()!=expected:raise ValueError('Native files cache changed')
        return saved
    url='https://ghcr.io/token?'+urllib.parse.urlencode({'service':'ghcr.io','scope':'repository:'+repository+':pull'})
    auth=json.load(urllib.request.urlopen(url,timeout=30));token=auth.get('token',auth.get('access_token'))
    def get(kind,digest):
        headers={'Authorization':'Bearer '+token,'Accept':','.join([
            'application/vnd.oci.image.manifest.v1+json','application/vnd.oci.image.index.v1+json',
            'application/vnd.docker.distribution.manifest.v2+json','application/vnd.docker.distribution.manifest.list.v2+json'])}
        request=urllib.request.Request('https://ghcr.io/v2/'+repository+'/'+kind+'/'+digest,headers=headers)
        with urllib.request.urlopen(request,timeout=90)as response:
            raw=response.read(64<<20)
            if response.read(1):raise ValueError('Native files image object exceeds 64 MiB')
        if 'sha256:'+hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Native image content digest mismatch')
        return raw
    manifest=json.loads(get('manifests','sha256:'+digest))
    if'manifests'in manifest:
        platforms=[x for x in manifest['manifests']if x.get('platform',{}).get('os')=='linux'and x.get('platform',{}).get('architecture')=='amd64']
        if len(platforms)!=1:raise ValueError('Native image platform is ambiguous')
        manifest=json.loads(get('manifests',platforms[0]['digest']))
    files={}
    for layer in manifest['layers']:files.update(unpack_files(get('blobs',layer['digest']),output))
    saved={'image':image,'files':files};output.mkdir(parents=True,exist_ok=True)
    (output/'files-manifest.json').write_text(json.dumps(saved,indent=2)+'\n')
    return saved


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('image');parser.add_argument('output',type=Path)
    args=parser.parse_args();saved=fetch(args.image,args.output)
    print(json.dumps({'files':len(saved['files']),'all_digests_verified':True}))
