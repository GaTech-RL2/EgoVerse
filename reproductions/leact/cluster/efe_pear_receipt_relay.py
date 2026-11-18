"""Keep one CPU-only mount available until the qualified model head starts."""
import json
from pathlib import Path
import time


def main():
    spec=json.loads(Path(__file__).with_name('spec.json').read_text())
    started=time.time();root=Path(spec['root']);deadline=time.monotonic()+3*3600
    print('PEAR_RECEIPT_RELAY '+json.dumps({'phase':'read_only_mount_ready','gpu':0,'primary_rl_updates':0}),flush=True)
    while time.monotonic()<deadline:
        for path in(root/'runs').glob('rp_jhh_pear_moe_warmstart496_full_*/head-heartbeat.json'):
            if path.stat().st_mtime>started and time.time()-path.stat().st_mtime<120:
                print('PEAR_RECEIPT_RELAY '+json.dumps({'phase':'primary_head_owns_mount','gpu':0}),flush=True)
                return
        time.sleep(15)


if __name__=='__main__':main()
