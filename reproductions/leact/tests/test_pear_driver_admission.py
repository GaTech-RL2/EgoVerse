import hashlib
import json
from pathlib import Path

import pytest

from reproductions.leact.cluster import efe_pear_full_admission as admission


@pytest.mark.parametrize('invalid',[{},
    {'passed':False,'source_check_passed':True,'rollout_nodes':[6,4]},
    {'passed':True,'source_check_passed':True,'rollout_nodes':[6,4],'script_sha256':'old-command'},
    {'passed':True,'source_check_passed':True,'rollout_nodes':[6]}])
def test_large_allocation_rejects_missing_failed_or_stale_driver_checks(tmp_path,invalid):
    (tmp_path/'primary-driver-check-status.json').write_text(json.dumps({'scheduler_status':'COMPLETED','worker':invalid}))
    with pytest.raises(admission.PendingAdmission,match='Exact primary training command'):
        admission.model_qualification(tmp_path)


def test_exact_driver_proof_allows_the_remaining_independent_gates_to_run(tmp_path):
    script=Path(admission.__file__).with_name('efe_pear_scaled_train.sh')
    (tmp_path/'primary-driver-check-status.json').write_text(json.dumps({'scheduler_status':'COMPLETED','worker':{
        'passed':True,'source_check_passed':True,'rollout_nodes':[6,4],
        'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),
        'custom_trainer_config_passed':True,
        'driver_entry_sha256':hashlib.sha256(script.with_name('efe_pear_multitask_train.py').read_bytes()).hexdigest(),
        'multitask_config_sha256':hashlib.sha256(script.with_name('efe_pear_multitask_config.py').read_bytes()).hexdigest()}}))
    with pytest.raises(admission.PendingAdmission,match='Two-node rollout-queue'):
        admission.model_qualification(tmp_path)
