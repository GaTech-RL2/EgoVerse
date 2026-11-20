"""Pack frozen worker inputs once to keep scheduler template expansion small."""
import base64
import gzip
import hashlib
import io
from pathlib import Path
import tarfile


FORMAT='embedded-targz-v1'


def bootstrap(directory,names,entry):
    directory=Path(directory);names=sorted(set(names))
    if entry not in names or any(Path(name).name!=name for name in names):
        raise ValueError('Worker inputs and entry must be flat frozen filenames')
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w')as archive:
        for name in names:
            raw=(directory/name).read_bytes()
            item=tarfile.TarInfo(name);item.size=len(raw);item.mode=0o644
            archive.addfile(item,io.BytesIO(raw))
    payload=gzip.compress(stream.getvalue(),mtime=0)
    encoded=base64.b64encode(payload).decode('ascii')
    digest=hashlib.sha256(payload).hexdigest()
    return f'''import base64,hashlib,io,os,sys,tarfile
from pathlib import Path
payload=base64.b64decode({encoded!r})
if hashlib.sha256(payload).hexdigest()!={digest!r}:raise ValueError('Frozen worker bundle changed')
lane=Path(__file__).resolve().parent
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz')as archive:
    archive.extractall(lane,filter='data')
os.execv(sys.executable,[sys.executable,'-u',str(lane/{entry!r})])
'''
