"""Restore pinned native artifacts and archive the bounded selector experiment."""

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from .worker import Publisher, archive_client, safe_relative, sha256, write_json
from .xiaomi_worker import restore, stop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--maximum-worker-seconds",type=int,default=21240)
    args = parser.parse_args()
    started = time.monotonic()
    workflow = os.environ["ASTRA_RUN_ID"]
    if not workflow.startswith("astra-complex-20261006-robocasa-xiaomi-selector-"):
        raise ValueError("Unexpected selector workflow identity")
    root, output = Path("/opt/astra-xiaomi"), Path("/opt/astra-results")
    output.mkdir(exist_ok=False)
    client = archive_client()
    prefix = f"experiments/astra-complex-20261006/{workflow}/results"
    if client.list_objects_v2(Bucket="rldb",Prefix=prefix+"/",MaxKeys=1).get("KeyCount"):
        raise FileExistsError("Selector result archive exists")
    publisher = Publisher(client,output,prefix)
    event = threading.Event()
    def publish():
        while not event.wait(30):
            try:publisher.publish()
            except Exception as exc:print(json.dumps({"archive_retry":type(exc).__name__}),flush=True)
    thread = threading.Thread(target=publish,daemon=True)
    thread.start()
    process, returncode = None, 1
    protocol = json.loads(args.protocol.read_text())
    write_json(output/"protocol.json",protocol)
    write_json(output/"worker_started.json",{"workflow":workflow,"source_revision":os.environ["ASTRA_SOURCE_REVISION"],
        "worker_limit_seconds":args.maximum_worker_seconds,"stage":args.stage,"started_unix":time.time()})
    try:
        receipt = restore(client,args.stage,args.manifest,root)
        write_json(output/"stage_receipt.json",receipt)
        for item in protocol["historical_artifacts"]:
            relative = str(safe_relative(item["relative"]))
            if not item["key"].startswith(protocol["historical_prefix"]) or item["bytes"] > 64*1024**2:
                raise ValueError("Historical evidence outside the registered scope")
            target = output/"history"/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            client.download_file("rldb",item["key"],str(target))
            if target.stat().st_size != item["bytes"] or sha256(target) != item["sha256"]:
                raise ValueError("Historical evidence checksum differs")
        env = {k:v for k,v in os.environ.items() if not k.startswith("R2_")}
        env.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false",
                   OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="2",PYTHONUNBUFFERED="1")
        command = [str(root/"runtime/bin/python"),"-u","-m",
                   "astra_reversal.complex_manipulation.xiaomi_selector_eval",
                   "--protocol",str(args.protocol),"--output",str(output/"evaluation")]
        write_json(output/"evaluation_command.json",{"argv":command})
        with (output/"evaluation.log").open("w") as log:
            process = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        returncode = process.wait(timeout=max(1,args.maximum_worker_seconds-(time.monotonic()-started)))
        if returncode:raise RuntimeError("Selector evaluator exited with an error")
    except Exception as exc:
        write_json(output/"worker_error.json",{"type":type(exc).__name__,"message":str(exc)})
        raise
    finally:
        stop(process)
        event.set()
        thread.join(timeout=180)
        if thread.is_alive():raise RuntimeError("Publisher did not finish")
        write_json(output/"worker_finished.json",{"returncode":returncode,"elapsed_seconds":time.monotonic()-started})
        publisher.publish()


if __name__ == "__main__":
    main()
