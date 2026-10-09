"""Separate exact physical resets from explicitly bounded camera rounding."""

import numpy as np

from astra_reversal.records import digest


CAMERAS = tuple('video.robot0_'+name for name in ('agentview_left','agentview_right','eye_in_hand'))


def load_observation(path):
    with np.load(path,allow_pickle=False) as archive:
        return {k:archive[k].item() if k == 'annotation.human.task_description' else archive[k]
                for k in archive.files if k != 'simulator_state'}


def assess_reset(expected, actual, reference=None, observed=None, allowance=None):
    if set(expected) != set(actual):
        raise ValueError('Reset fingerprint fields changed')
    physical = all(actual[k] == expected[k] for k in expected if k != 'observation_sha256')
    exact = actual == expected
    if allowance is not None:
        if (set(allowance) != {'maximum_channel_difference','maximum_changed_pixels_per_camera'}
                or type(allowance['maximum_channel_difference']) is not int
                or allowance['maximum_channel_difference'] != 1
                or type(allowance['maximum_changed_pixels_per_camera']) is not int
                or not 1 <= allowance['maximum_changed_pixels_per_camera'] <= 64):
            raise ValueError('Camera tolerance exceeds the registered rounding bound')
    result = dict(exact=exact,physical_fingerprints_exact=physical,
                  noncamera_observations_exact=None,cameras={},allowance=allowance)
    if reference is None or observed is None:
        if allowance is not None:
            raise ValueError('A tolerant comparison requires both original observation arrays')
        return dict(result,accepted=exact)
    if digest(reference) != expected['observation_sha256'] or digest(observed) != actual['observation_sha256']:
        raise ValueError('Observation arrays do not match the stored fingerprints')
    if set(reference) != set(observed) or not set(CAMERAS).issubset(reference):
        raise ValueError('Reset observation keys changed')
    noncamera = all(np.asarray(reference[k]).dtype == np.asarray(observed[k]).dtype
                   and np.array_equal(reference[k],observed[k]) for k in reference if k not in CAMERAS)
    result['noncamera_observations_exact'] = noncamera
    allowed = True
    for key in CAMERAS:
        a,b = np.asarray(observed[key]),np.asarray(reference[key])
        if a.shape != b.shape or a.dtype != np.uint8 or b.dtype != np.uint8 or a.ndim != 3 or a.shape[-1] != 3:
            raise ValueError('Native uint8 RGB camera contract changed')
        delta = np.abs(a.astype(np.int16)-b.astype(np.int16))
        changed = int(np.count_nonzero(np.any(delta,axis=-1)))
        maximum = int(delta.max(initial=0))
        camera_ok = changed == 0 or (allowance is not None
            and maximum <= allowance['maximum_channel_difference']
            and changed <= allowance['maximum_changed_pixels_per_camera'])
        result['cameras'][key] = dict(exact=changed==0,maximum_channel_difference=maximum,
            changed_pixels=changed,pixels=int(a.shape[0]*a.shape[1]),accepted=camera_ok)
        allowed = allowed and camera_ok
    return dict(result,accepted=bool(physical and noncamera and allowed))
