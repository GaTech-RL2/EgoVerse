"""Reuse completed native controls without turning interruptions into passes."""
import copy


def passed_checks(spec,checks,receipt_sha256=None):
    rows={row['task']:row for row in spec['tasks']};passed={}
    if set(checks)-set(rows):raise ValueError('A resumed receipt contains another shard\'s tasks')
    for task,value in checks.items():
        row=rows[task];negative=value.get('native_final',{})
        valid=(value.get('row')==row and value.get('scene_sha256')==spec['expected_scenes'][task]
            and value.get('native_rtx_api_passed')is True and value.get('native_file_transport_passed')is True
            and bool(value.get('action_contract'))and negative.get('finalized')is True
            and negative.get('success')is False and negative.get('graded_by')=='live'and negative.get('replays')==0)
        if row.get('oracle')=='full':
            positive=value.get('positive_native_final',{})
            valid=valid and value.get('positive_solver_control')is True and positive.get('finalized')is True\
                and positive.get('success')is True and positive.get('graded_by')=='live'and positive.get('replays')==0
        if valid:
            passed[task]=copy.deepcopy(value)
            if receipt_sha256:passed[task]['resumed_from_receipt_sha256']=receipt_sha256
    return passed
