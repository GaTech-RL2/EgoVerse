import math

import numpy as np
import pytest

from reproductions.leact.cluster.efe_pear_reasoning_scenes import SceneEpisode,checked_point,REASONING_FAMILIES


@pytest.mark.parametrize('point',[[True,.1],[float('nan'),0],[0,float('inf')],[-.1,0],[0,1.1],[0],None])
def test_malformed_or_nonfinite_image_coordinates_are_not_actions(point):
    with pytest.raises(ValueError):checked_point(point)


def test_normalized_image_coordinate_boundaries_are_permitted():
    np.testing.assert_array_equal(checked_point([0,1]),[0.,1.])


@pytest.mark.parametrize('family',REASONING_FAMILIES)
def test_independent_partitions_do_not_reuse_a_physical_scene(family):
    pytest.importorskip('mujoco',reason='Optional native CPU physics integration')
    episodes=[]
    try:
        for seed,partition in [(0,'train'),(1,'train'),(0,'development')]:
            episodes.append(SceneEpisode(family,seed,partition))
        assert len({e.scene_sha256 for e in episodes})==3
        for e in episodes:assert np.isfinite(e.initial_state).all()
        if family=='affordance_part_selection':
            for e in episodes:
                assert e.model.geom_bodyid[e.ids[0]]==e.model.joint('door_hinge').bodyid[0]
    finally:
        for e in episodes:e.close()


def test_a_qualification_failure_cannot_be_labeled_cluster_qualified(tmp_path):
    pytest.importorskip('mujoco',reason='Optional native CPU physics integration')
    from reproductions.leact.cluster.efe_pear_reasoning_qualify import qualify
    with pytest.raises(ValueError,match='OSMO workflow'):
        qualify(tmp_path,train_scenes=1,development_scenes=1,cluster=True)
