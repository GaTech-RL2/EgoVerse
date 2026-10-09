"""Camera rounding cannot hide physical, state, instruction or larger image changes."""

import copy

import numpy as np
import pytest

from astra_reversal.records import digest
from astra_reversal.complex_manipulation.xiaomi_reset_match import CAMERAS, assess_reset


ALLOWANCE = dict(maximum_channel_difference=1,maximum_changed_pixels_per_camera=64)


def observations():
    return {**{k:np.full((256,256,3),100,np.uint8) for k in CAMERAS},
            'state':np.zeros(14,np.float64),'annotation.human.task_description':'Pack two lunches.'}


def fingerprint(obs,**changes):
    return dict(model_sha256='same',state_sha256='same',integration_state_sha256='same',
                environment_rng_sha256='same',observation_sha256=digest(obs),
                instruction=obs['annotation.human.task_description'],**changes)


@pytest.mark.parametrize('pixels,magnitude,accepted',[(0,0,True),(3,1,True),(64,1,True),(65,1,False),(1,2,False)])
def test_sparse_rounding_boundary_and_larger_image_changes(pixels,magnitude,accepted):
    reference = observations()
    actual = copy.deepcopy(reference)
    actual[CAMERAS[1]].reshape(-1,3)[:pixels] += magnitude
    result = assess_reset(fingerprint(reference),fingerprint(actual),reference,actual,ALLOWANCE)
    assert result['accepted'] is accepted
    assert result['exact'] is (pixels==0)
    assert result['cameras'][CAMERAS[1]]['changed_pixels'] == pixels
    if pixels:
        assert assess_reset(fingerprint(reference),fingerprint(actual),reference,actual)['accepted'] is False


@pytest.mark.parametrize('change',['physics','state','instruction'])
def test_camera_allowance_never_allows_nonvisual_differences(change):
    reference = observations()
    actual = copy.deepcopy(reference)
    if change=='state':actual['state'][0] = 1e-12
    if change=='instruction':actual['annotation.human.task_description'] = 'A different task.'
    actual_fingerprint = fingerprint(actual)
    if change=='physics':actual_fingerprint['integration_state_sha256'] = 'changed'
    result = assess_reset(fingerprint(reference),actual_fingerprint,reference,actual,ALLOWANCE)
    assert result['accepted'] is False


def test_reference_arrays_and_fingerprints_must_agree():
    reference = observations()
    expected = fingerprint(reference)
    reference[CAMERAS[0]][0,0,0] += 1
    with pytest.raises(ValueError,match='stored fingerprints'):
        assess_reset(expected,fingerprint(reference),reference,reference,ALLOWANCE)


def test_tolerance_cannot_expand_or_run_without_measured_arrays():
    obs = observations()
    with pytest.raises(ValueError,match='rounding bound'):
        assess_reset(fingerprint(obs),fingerprint(obs),obs,obs,{**ALLOWANCE,'maximum_channel_difference':2})
    with pytest.raises(ValueError,match='original observation arrays'):
        assess_reset(fingerprint(obs),fingerprint(obs),allowance=ALLOWANCE)
