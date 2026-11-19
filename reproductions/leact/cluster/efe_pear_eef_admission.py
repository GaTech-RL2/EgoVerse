"""Require measured, source-pinned EEF motion before admitting an RL branch."""
import copy
import hashlib
import json
from pathlib import Path

from .efe_pear_full_admission import PendingAdmission
from .efe_pear_eef import VERSION,POSE_SUITES,DELTA_SUITES


SUITES=POSE_SUITES|DELTA_SUITES|{'native_gr00t_yam'}


def interface_metadata(source=None):
    result={'version':VERSION,'scope':'pear_rl_only','teacher_collection_changed':False,
        'native_qualification_required_before_driver':True,
        'initialization':'moe_warmstart496','optimizer_state':'fresh_new_interface_branch',
        'prior_joint_interface_updates_reported_separately':True}
    if source:
        names=['efe_pear_eef.py','efe_pear_eef_service.py','efe_pear_vlabench_eef.py',
            'efe_pear_yam_eef.py','efe_pear_yam_session_policy.py','efe_pear_rtx_env.py',
            'efe_pear_native_episode.py','efe_action_loop.py','efe_pear_context.py']
        result['native_adapter_sha256']={n:hashlib.sha256((Path(source)/n).read_bytes()).hexdigest()for n in names}
    return result


def qualify(root,source):
    path=Path(root)/'eef-interface-v1/qualification.json'
    if not path.is_file():raise PendingAdmission('PEAR EEF native motion checks are still running')
    proof=json.loads(path.read_text())
    if (proof.get('control_interface')!=VERSION or proof.get('scope')!='pear_rl_only' or
            proof.get('teacher_collection_changed')is not False or proof.get('passed')is not True or
            set(proof.get('simulators',{}))!=SUITES):
        raise PendingAdmission('The complete PEAR-only EEF mixture has not passed qualification')
    for suite,item in proof['simulators'].items():
        if item.get('passed')is not True:raise PendingAdmission('EEF native motion has not passed: '+suite)
        receipt=Path(item['receipt'])
        if hashlib.sha256(receipt.read_bytes()).hexdigest()!=item['receipt_sha256']:
            raise ValueError('An EEF native qualification receipt changed')
    for name,expected in proof.get('adapter_sha256',{}).items():
        if Path(name).name!=name or hashlib.sha256((Path(source)/name).read_bytes()).hexdigest()!=expected:
            raise PendingAdmission('An EEF adapter changed after native qualification: '+name)
    if not proof.get('adapter_sha256'):raise PendingAdmission('EEF qualification has no source pins')
    return interface_metadata()


def admission(source,interface):
    result=copy.deepcopy(source)
    suites={item['row']['suite']for item in result['native_membership']['training_tasks'].values()}
    if suites!=SUITES-{'native_gr00t_yam'}:
        raise ValueError('EEF fork changed the complete admitted manipulation mixture')
    result['curriculum']['control_interface']=interface
    for item in result['native_membership']['training_tasks'].values():
        item['control_interface']=VERSION
        # The original receipt establishes scene/reset/grade identity. Its old
        # action contract is evidence, not the executable EEF declaration.
        item['prior_native_action_contract']=item.pop('action_contract')
        item['action_contract_source']='live_source_pinned_eef_adapter_after_reset'
    return result


def protected_teacher_allocations(root):
    path=Path(root)/'astra-wide-warmstart-v1/launches.json'
    if not path.is_file():return set()
    return {item['workflow']for item in json.loads(path.read_text()).values()
        if not item.get('superseded_by')and item.get('workflow')}
