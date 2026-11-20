from types import SimpleNamespace

import numpy as np

from reproductions.leact.cluster.efe_pear_yam_contract import termination_snapshot,observe_termination_computes


def test_native_tensor_buffer_is_read_through_named_public_terms():
    values={'success':np.array([False]),'time_out':np.array([True])}
    manager=SimpleNamespace(active_terms=list(values),get_term=values.__getitem__,
        _term_dones=np.array([[False,True]]))
    assert termination_snapshot(manager)=={'success':False,'time_out':True}


def test_outcome_observer_preserves_native_compute_identity_and_survives_reset():
    native=np.array([True]);values={'success':np.array([False]),'time_out':np.array([False])};calls=[]
    def compute(step):
        calls.append(step);values['success'][0]=True
        return native
    manager=SimpleNamespace(active_terms=list(values),get_term=values.__getitem__,compute=compute)
    snapshots=[];original=observe_termination_computes(manager,snapshots.append)
    assert original is compute and manager.compute(13)is native and calls==[13]
    values['success'][0]=False
    assert snapshots==[{'success':True,'time_out':False}]
