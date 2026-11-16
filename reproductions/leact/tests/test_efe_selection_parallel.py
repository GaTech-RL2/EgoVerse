import multiprocessing
from pathlib import Path
import time

import pytest

from reproductions.leact.cluster.efe_qwen36_leact import candidate_guard,selection_indices


def contender(root,entered):
    with candidate_guard(root,'a'*64):entered.set()


def test_selection_membership_rejects_duplicates_and_out_of_range():
    assert selection_indices(8,{'indices':[1,3,7]})=={1,3,7}
    for indices in [[1,1],[-1],[8],[True]]:
        with pytest.raises(ValueError):selection_indices(8,{'indices':indices})


def test_candidate_lock_protects_cross_process_publication(tmp_path):
    ctx=multiprocessing.get_context('fork');entered=ctx.Event()
    with candidate_guard(tmp_path,'a'*64):
        child=ctx.Process(target=contender,args=(str(tmp_path),entered));child.start()
        assert not entered.wait(.15)
    assert entered.wait(3)
    child.join(3)
    assert child.exitcode==0


def test_candidate_lock_releases_after_failure_and_rejects_paths(tmp_path):
    with pytest.raises(RuntimeError):
        with candidate_guard(tmp_path,'a'*64):raise RuntimeError('interrupted producer')
    with candidate_guard(tmp_path,'a'*64):pass
    with pytest.raises(ValueError):
        with candidate_guard(tmp_path,'../bad'):pass
