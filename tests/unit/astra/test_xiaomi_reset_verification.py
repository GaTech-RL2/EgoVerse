"""Rendering variance must never admit a mismatched policy start or hide resets."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from astra_reversal.complex_manipulation import xiaomi_selector_eval as selector


def setup(monkeypatch, tmp_path, fingerprints):
    expected = dict(state_sha256='state',observation_sha256='image',instruction='task')
    pending = iter(fingerprints)
    monkeypatch.setattr(selector,'restore_anchor',lambda *args: {'image':np.zeros((2,2,3),np.uint8)})
    monkeypatch.setattr(selector,'fingerprint',lambda *args: {**expected,**next(pending)})
    started = []
    record = SimpleNamespace(root=tmp_path,directory=tmp_path/'seed')
    def reset(*args):
        started.append(args)
        record.directory.mkdir()
    record.reset = reset
    counts = {'policy_restore_resets':0}
    wrapped = selector.ReplayEnvironment(object(),record,{},expected,counts,maximum_restore_attempts=3)
    return wrapped,counts,started


def test_observation_only_retry_still_requires_exact_start_and_counts_extra_restore(monkeypatch,tmp_path):
    env,counts,started = setup(monkeypatch,tmp_path,[{'observation_sha256':'different'},{}])
    env.reset(seed=7)
    assert len(started)==1
    assert counts == {'policy_restore_resets':1,'additional_verification_restore_resets':1}
    audit = json.loads((tmp_path/'reset_verification.json').read_text())
    assert [r['exact'] for r in audit['checks']] == [False,True]
    assert (tmp_path/'reset_verification_0_observation.npz').exists()
    assert json.loads((tmp_path/'seed/paired_reset.json').read_text())['verified'] is True


@pytest.mark.parametrize('physical_change',[False,True])
def test_persistent_or_physical_mismatch_never_starts_a_policy_episode(monkeypatch,tmp_path,physical_change):
    change = {'state_sha256':'wrong'} if physical_change else {'observation_sha256':'different'}
    env,counts,started = setup(monkeypatch,tmp_path,[change]*3)
    with pytest.raises(ValueError,match='exact fingerprint'):
        env.reset(seed=7)
    assert not started and not (tmp_path/'seed').exists()
    assert counts.get('additional_verification_restore_resets',0) == (0 if physical_change else 2)
    assert (tmp_path/'reset_mismatch_observation.npz').exists()
    audit = json.loads((tmp_path/'reset_verification.json').read_text())
    assert len(audit['checks']) == (1 if physical_change else 3)
