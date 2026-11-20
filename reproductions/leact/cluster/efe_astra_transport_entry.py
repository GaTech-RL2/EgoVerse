"""Frozen teacher-only replacement entry for an unchanged native remote worker."""
import json
from pathlib import Path
import runpy

from efe_astra_transport_diagnostics import VERSION, install


def main():
    lane = Path(__file__).resolve().parent
    spec = json.loads((lane/'spec.json').read_text())
    if (spec.get('owner_model_workflow') != 'rp_jhh_astra_wide_warmstart_20261119_v1'
            or spec.get('teacher_profile_id') != 'astra62-xhigh-v1'
            or spec.get('teacher_native_entry') != 'efe_pear_primary_native.py'
            or spec.get('teacher_transport_diagnostics', {}).get('version') != VERSION):
        raise ValueError('Transport diagnostics are restricted to the explicitly staged Astra teacher EFE worker')
    install()
    runpy.run_path(str(lane/'efe_astra_original_remote_worker.py'), run_name='__main__')


if __name__ == '__main__':
    main()
