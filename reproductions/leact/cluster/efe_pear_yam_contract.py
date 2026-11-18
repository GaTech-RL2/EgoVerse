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
