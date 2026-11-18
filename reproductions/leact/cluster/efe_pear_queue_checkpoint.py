"""Stage TransferQueue on shared storage while model shards stay node-local."""
from contextlib import contextmanager
from pathlib import Path
import shutil
from uuid import uuid4

from efe_pear_checkpoint_archive import archive


@contextmanager
def shared_queue_checkpoint(tq,shared_root):
    root=Path(shared_root);root.mkdir(mode=0o700,parents=True,exist_ok=True)
    save,load=tq.save_checkpoint,tq.load_checkpoint
    def staged_save(path,**kwargs):
        shared=root/('rp_jhh_tq_save_'+uuid4().hex)
        try:
            save(str(shared),**kwargs)
            archive(shared,Path(path),reserve_bytes=0,record_manifest=False)
        finally:
            shutil.rmtree(shared,ignore_errors=True)
    def staged_load(path,**kwargs):
        shared=root/('rp_jhh_tq_restore_'+uuid4().hex)
        try:
            archive(Path(path),shared,record_manifest=False)
            return load(str(shared),**kwargs)
        finally:
            shutil.rmtree(shared,ignore_errors=True)
    tq.save_checkpoint,tq.load_checkpoint=staged_save,staged_load
    try:yield
    finally:tq.save_checkpoint,tq.load_checkpoint=save,load
