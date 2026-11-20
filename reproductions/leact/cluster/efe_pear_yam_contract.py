"""Read the native termination API without relying on private buffer layout."""


def termination_snapshot(manager):
    return {name:bool(manager.get_term(name)[0].item())for name in manager.active_terms}


def observe_termination_computes(manager,callback):
    """Observe official outcomes before native auto-reset, without changing them."""
    original=manager.compute
    def observed(*args,**kwargs):
        result=original(*args,**kwargs)
        callback(termination_snapshot(manager))
        return result
    manager.compute=observed
    return original


def terminal_result(contract,instruction,scene,terms,action,executed,trial_path):
    """Grade captured native terms without touching an auto-reset observation."""
    import hashlib,json
    if not terms or'success'not in terms or any(type(v)is not bool for v in terms.values()):
        raise ValueError('A terminal native episode needs captured official boolean terms')
    actions=[]if action is None or action.get('stop')else[action]
    return {'images':None,'state':None,'public_blocks':None,'action_contract':contract,
        'instruction':instruction,'actions':actions,'rewards':[float(terms['success'])],
        'done':True,'executed':len(actions),'outcome':'done','native_trial_path':str(trial_path),
        'scene_sha256':scene,'native_termination_terms':terms,
        'interface_context_version':'pear-public-interface-language-v1','interface_context_sha256':None,
        'requested_control_steps':0 if action is None else action.get('repeat',0),
        'executed_control_steps':executed,
        'action_contract_sha256':hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()}
