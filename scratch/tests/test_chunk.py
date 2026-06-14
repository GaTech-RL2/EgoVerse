"""Unit tests for the chunk-token baseline: chunk-target boundaries + I/O shapes."""
import torch
from egomimic.algo.hnet_chunk import (
    _build_chunk_targets_packed, ChunkTokenPolicy,
)


def test_chunk_targets_packed():
    # 2 episodes: frames [0,1,2] and [3,4]; actions = frame index in both dims.
    T, A, K = 5, 2, 2
    actions = torch.arange(T, dtype=torch.float32).view(T, 1).repeat(1, A)
    cu = torch.tensor([0, 3, 5])
    target, mask = _build_chunk_targets_packed(actions, cu, K)
    assert target.shape == (T, K, A) and mask.shape == (T, K)
    # frame 0 -> [a0,a1] valid; frame 2 -> [a2, (clamped a2)] mask [1,0];
    # frame 3 -> [a3,a4]; frame 4 -> [a4, clamp] mask [1,0]
    assert target[0, 0, 0] == 0 and target[0, 1, 0] == 1 and mask[0].tolist() == [1, 1]
    assert target[2, 0, 0] == 2 and mask[2].tolist() == [1, 0]
    assert target[3, 0, 0] == 3 and target[3, 1, 0] == 4 and mask[3].tolist() == [1, 1]
    assert mask[4].tolist() == [1, 0]
    # masked tail entries are zeroed
    assert target[2, 1, 0] == 0 and target[4, 1, 0] == 0
    print("chunk_targets_packed OK")


def test_policy_shapes():
    pol = ChunkTokenPolicy(action_dim=2, chunk_k=4, d_model=32, image_size=96,
                           arch_layout="T2", num_heads=4, d_intermediate=64)
    N = 5
    obs = {"front_img_1": torch.randn(N, 3, 96, 96),
           "state_agent_obj": torch.randn(N, 5)}
    pred, aux = pol(None, obs)
    assert pred.shape == (N, 4, 2), pred.shape
    print("img tokens =", pol._n_img_tok, " forward OK", pred.shape)
    # generate: single frame -> (1, T, A)
    g_obs = {"front_img_1": torch.randn(1, 3, 96, 96),
             "state_agent_obj": torch.randn(1, 5)}
    chunk = pol.generate(g_obs, batch_size=1, device="cpu", T=3)
    assert chunk.shape == (1, 3, 2), chunk.shape
    print("generate OK", chunk.shape)


if __name__ == "__main__":
    test_chunk_targets_packed()
    test_policy_shapes()
    print("ALL CHUNK TESTS PASSED")
