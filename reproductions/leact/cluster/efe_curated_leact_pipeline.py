"""Durable curated LeAct followed by the matched six native training tasks."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback

from efe_training_data import write


def main():
    spec=json.loads(Path('/tmp/efe-training-spec.json').read_text())
    root=Path(spec['archive_root'])/spec['run_id'];root.mkdir(parents=True,exist_ok=True)
    lock=(root/'pipeline-writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def progress(phase,detail):
        record={'phase':phase,'detail':detail,'updated_utc':datetime.now(timezone.utc).isoformat(),
                'workflow':os.environ['EFE_WORKFLOW_ID']}
        write(root/'pipeline-progress.json',record);print(json.dumps(record),flush=True)
    try:
        import efe_cuda_helpers as helpers
        import efe_training_pilot as pilot
        helpers.ROOT=root;helpers.ARCHIVE=root;env=helpers.driver_environment();pilot.ROOT=root
        env.update(OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        progress('curated_leact','Generate student backward candidates from the curated checkpoint, retain positive complete-action likelihood gains, and train with equal warmstart replay.')
        pilot.run_logged('leact-pipeline',[sys.executable,'/tmp/efe_leact_vl.py'],env,24*3600)
        complete=json.loads((root/'complete.json').read_text())
        if not complete.get('success'):raise ValueError('Curated LeAct failed')
        if complete.get('no_admissions'):
            write(root/'pipeline-complete.json',{'success':True,'no_admissions':True,'selection':complete})
            progress('no_admissions','No admitted student candidates; the warmstart checkpoint remains the outcome. No duplicate native rollout is counted.')
            return
        ready=json.loads((root/'leact-ready.json').read_text())
        from efe_leact_pipeline import export_portable
        from efe_direct_warmstart import preserve_training
        portable=export_portable(root,ready)
        portable['optimizer_persistence']='Full FP32 master parameters, Adam and RNG retained on shared storage.'
        write(root/'portable-ready.json',portable)
        preserve_training(root,root,ready,{'durable_from_start':True})
        progress('native_evaluation','LeAct checkpoint preserved. Evaluate the same six tasks, direct-control prompt, image processing and native grader as the curated warmstart.')
        lane=Path('/tmp/efe-training-evaluation-v1/warm96')
        evaluation=json.loads((lane/'evaluation-template.json').read_text())
        evaluation.update(student_checkpoint=portable['checkpoint'],student_manifest=portable['manifest'],
            student_revision=ready['revision'],training_parent=os.environ['EFE_WORKFLOW_ID'])
        write(lane/'spec.json',evaluation)
        pilot.run_logged('native-evaluation',[sys.executable,str(lane/'bootstrap.py')],env,12*3600)
        native=json.loads((lane/'results'/os.environ['EFE_WORKFLOW_ID']/'worker-complete.json').read_text())
        if not native.get('success'):raise ValueError('Native curated LeAct evaluation incomplete')
        write(root/'pipeline-complete.json',{'success':True,'training':ready,'native':native})
        progress('complete','Curated LeAct and matched native rollouts complete.')
    except Exception as error:
        write(root/'pipeline-complete.json',{'success':False,'error':str(error),'traceback':traceback.format_exc()})
        progress('needs_review',str(error));raise
    finally:
        for _ in range(360):
            if (root/'pipeline-collected').exists():break
            time.sleep(5)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
