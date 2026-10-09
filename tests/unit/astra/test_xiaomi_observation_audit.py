"""Do not hide non-image differences behind a global observation hash."""

import numpy as np
import pytest

from astra_reversal.complex_manipulation.xiaomi_observation_audit import compare_observations


def test_pixel_quantization_and_robot_state_are_measured_separately():
    original = {'image':np.zeros((3,4,3),np.uint8),'state':np.array([1.,2.]),'text':'pack lunch'}
    actual = {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in original.items()}
    actual['image'][1,2,1] = 1
    result = compare_observations(actual,original)
    assert result['state']['exact'] and result['text']['exact']
    assert result['image']['max_abs'] == 1 and result['image']['changed_pixels'] == 1
    assert result['image']['changed_elements'] == 1 and result['image']['elements'] == 36
    assert result['image']['mean_abs'] == pytest.approx(1/36)
    actual['state'][0] += .01
    assert compare_observations(actual,original)['state']['exact'] is False


def test_missing_observation_field_is_not_silently_ignored():
    with pytest.raises(ValueError):
        compare_observations({'image':np.zeros(3)},{'image':np.zeros(3),'state':np.ones(2)})
