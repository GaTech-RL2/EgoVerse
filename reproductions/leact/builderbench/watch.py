"""Publish the locally running BuilderBench probe to the experiment tracker."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

from ..data import write_json
from ..tracking.export import snapshot
from ..tracking.server import ingest
from .speed import publish as publish_speed


def main():
    base = Path(__file__).resolve().parents[1]
    previous = None
    for _ in range(720):
        path = base / 'artifacts/builderbench/expert-status.json'
        if not path.exists():
            time.sleep(10)
            continue
        raw = path.read_text()
        probe = json.loads(raw)
        if raw != previous:
            try:
                plan_path = base / 'artifacts/experiment-plan.json'
                plan = json.loads(plan_path.read_text())
                plan['updated_utc'] = datetime.now(timezone.utc).isoformat()
                label = 'collection' if probe.get('study_kind') == 'teacher_collection' else 'probe'
                description = (f'GPT-6 {label} {probe["status"]}: {probe["completed_rollouts"]}/'
                               f'{probe["planned_rollouts"]} rollouts complete, '
                               f'{probe["successful_rollouts"]} fully solved')
                for row in plan['sequence']:
                    if row['id'] == 'builderbench':
                        row['status'] = description
                plan['builderbench']['implementation_status'] = description + '; no GPU training started'
                write_json(plan_path, plan)
                publish_speed(base)
                value = snapshot(base)
                # Older live GPU monitors read this embedded status too.
                status_path = base / 'artifacts/run-status.json'
                status = json.loads(status_path.read_text())
                for key in ['experiment_plan', 'builderbench_probe', 'builderbench_expert', 'builderbench_examples', 'builderbench_speed', 'training_lineage', 'self_leact']:
                    status[key] = value['status'].get(key)
                write_json(status_path, status)
                receipt = ingest(base / 'cluster/.build/tracking/experiments.sqlite3', value)
                print(json.dumps(dict(updated_utc=plan['updated_utc'], probe_status=probe['status'],
                                      completed=probe['completed_rollouts'], snapshot_id=receipt['id'])), flush=True)
                previous = raw
            except Exception as error:
                print(json.dumps(dict(tracking_error=repr(error))), flush=True)
        if probe['status'] in {'completed', 'stopped', 'superseded_prompt_conflict'} and previous == raw:
            return
        time.sleep(20)


if __name__ == '__main__':
    main()
