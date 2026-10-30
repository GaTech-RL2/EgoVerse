import numpy as np
import pytest
import torch

mano = pytest.importorskip("mano")
aria_utils = pytest.importorskip("egomimic.scripts.aria_process.aria_utils")


def _model(is_rhand, n):
    try:
        return mano.load(
            model_path=aria_utils._default_mano_model_dir(),
            is_rhand=is_rhand,
            num_pca_comps=45,
            batch_size=n,
            flat_hand_mean=False,
        )
    except (FileNotFoundError, OSError, AssertionError) as e:
        pytest.skip(f"MANO weights unavailable: {e}")


@pytest.mark.parametrize("is_rhand", [False, True])
def test_mano_joints_fn_matches_full_forward(is_rhand):
    n = 64
    model = _model(is_rhand, n)
    g = torch.Generator().manual_seed(0)
    betas = torch.randn(n, 10, generator=g)
    orient = torch.randn(n, 3, generator=g)
    pose = 0.5 * torch.randn(n, 45, generator=g)
    transl = torch.randn(n, 3, generator=g)
    ref = model(
        betas=betas,
        global_orient=orient,
        hand_pose=pose,
        transl=transl,
        return_verts=True,
        return_tips=True,
    ).joints
    got = aria_utils.mano_joints_fn(model)(betas, orient, pose, transl)
    assert got.shape == ref.shape == (n, 21, 3)
    torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-5)


def test_batch_size_does_not_change_fit():
    _model(True, 1)
    rng = np.random.default_rng(0)
    kp = rng.normal(scale=0.05, size=(40, 21, 3)) + np.array([0.3, 0.1, 0.5])
    kp[5] = 1e9
    kp[7, 2] = 1e9
    kp = kp.reshape(40, 63)
    kw = dict(is_rhand=True, device="cpu", n_iters=20, chunk_size=16)
    small = aria_utils.convert_aria_keypoints_to_mano(kp, batch_size=16, **kw)
    big = aria_utils.convert_aria_keypoints_to_mano(kp, batch_size=48, **kw)
    assert np.isnan(small[5]).all() and np.isnan(big[5]).all()
    np.testing.assert_allclose(big, small, atol=1e-6, equal_nan=True)
