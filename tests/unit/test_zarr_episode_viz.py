"""The two zarr episode viz scripts share their drawing helpers."""

import importlib

import numpy as np
import pytest

from egomimic.scripts import zarr_episode_viz as zv


@pytest.mark.parametrize(
    "module",
    [
        "egomimic.scripts.abc_process.viz_eva_episode",
        "egomimic.scripts.mecka_process.viz_mecka_episode",
    ],
)
def test_viz_scripts_import(module):
    importlib.import_module(module)


def test_episode_path():
    assert zv.episode_path("/x/ep.zarr", "/data") == "/x/ep.zarr"
    assert zv.episode_path("abc", "/data") == "/data/abc.zarr"


def test_project_and_draw_triad():
    K = np.array([[100.0, 0, 32, 0], [0, 100.0, 32, 0], [0, 0, 1, 0]])
    px = zv.project(K, np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]))
    np.testing.assert_allclose(px[0], [32, 32])
    assert np.isnan(px[1]).all()
    img = np.zeros((64, 64, 3), np.uint8)
    M = np.eye(4)
    M[2, 3] = 1.0
    zv.draw_triad(img, K, M, 0.1, "t")
    assert img.any()
