"""Smoke test for the config-switchable backbone in ChunkTokenPolicy.

Verifies both backbone=="flat" (unchanged) and backbone=="hnet_chunked"
(dynamic chunker spliced into the middle of the transformer) produce correct
chunk shapes for both _predict_chunk (forward) and generate.
"""
import torch

from egomimic.algo.hnet_chunk import ChunkTokenPolicy


def _obs(N=5, dev="cuda"):
    return {
        "front_img_1": torch.randn(N, 3, 96, 96, device=dev),
        "state_agent_obj": torch.randn(N, 5, device=dev),
    }


def run(backbone):
    dev = "cuda"
    pol = ChunkTokenPolicy(
        action_dim=2,
        chunk_k=8,
        d_model=128,
        image_size=96,
        arch_layout="T4",
        num_heads=4,
        d_intermediate=512,
        backbone=backbone,
    ).to(dev)
    pol.train()

    obs = _obs(5, dev)
    pred, _ = pol(None, obs)
    print(f"[{backbone}] forward chunk shape: {tuple(pred.shape)}")
    assert tuple(pred.shape) == (5, 8, 2), f"bad chunk shape {tuple(pred.shape)}"

    if backbone == "hnet_chunked":
        rl = pol._last_ratio_loss
        print(f"[{backbone}] ratio_loss: {rl.item():.4f}  (requires_grad={rl.requires_grad})")
        assert rl.requires_grad, "ratio loss must be differentiable"
        # backprop sanity through the full pipeline + ratio loss
        loss = (pred.float() ** 2).mean() + 0.03 * rl
        loss.backward()
        print(f"[{backbone}] backward OK")

    g = pol.generate(_obs(1, dev), batch_size=1, device=dev)
    print(f"[{backbone}] generate shape: {tuple(g.shape)}")
    assert tuple(g.shape) == (1, 8, 2), f"bad generate shape {tuple(g.shape)}"
    print(f"[{backbone}] PASS\n")


if __name__ == "__main__":
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    run("flat")
    run("hnet_chunked")
    print("ALL SMOKE TESTS PASSED")
