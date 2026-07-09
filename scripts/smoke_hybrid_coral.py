"""Smoke test for the HYBRID dual-stream + CORAL H-Net (config v4).

Verifies, on a compute node (CUDA), WITHOUT touching the slow data/norm pipeline:

  1. The model config ``hnet_dualstream_hybrid_coral`` composes + instantiates
     (DualStreamOuterStage + HybridDualStreamCoralStage + chunker-free agnostic
     H-Net + per-emb chunking specific H-Nets + partitioned GMM heads), wired
     through PackedAlgoBase with coral_weight>0.
  2. Cross-attention out_proj is ZERO-INIT (A_top == A_full at init).
  3. GRADIENT CONTRACT (one-way + detach), driving outer_stage directly:
       * A_top.backward -> grads in agnostic trunk + cross_attn, NONE in the
         specific H-Nets (S_M is detached; agnostic reads it one-way).
       * S_top.backward -> grads in the specific H-Net, NONE in the agnostic
         trunk / cross_attn (specific never reads agnostic).
  4. FULL ALGO PATH on a synthetic 2-embodiment packed batch:
       process_batch_for_training -> forward_training -> compute_losses ->
       backward. Asserts ``coral_loss`` is present, finite, > 0 (random per-emb
       features have differing covariance) and that it is folded into the
       optimized ``action_loss``; all grads finite.

Run (compute node):
    srun --jobid=<JOB> --chdir=$PWD \
      /coc/flash7/agao81/EgoVerse/emimic2/bin/python scripts/smoke_hybrid_coral.py
"""

from __future__ import annotations

import socket
import sys

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from egomimic.models.hnet.context import HNetContext
from egomimic.rldb.embodiment.embodiment import get_embodiment_id

CFG_DIR = "/coc/flash7/agao81/Egoversedev/egomimic/hydra_configs/model"
CFG_NAME = "hnet_dualstream_hybrid_coral"
DOMAINS = ["pushshapes_sim", "pushshapes_sim_small_circle"]
D_MODEL = 704
ACTION_DIM = 2
STATE_DIM = 2
IMG = (3, 96, 96)


class StubNormStats:
    """Minimal MultiDataset stand-in: identity keyname + identity normalize,
    fixed key topology for the two pushshapes embodiments."""

    _BY_TYPE = {
        "action_keys": ["actions"],
        "proprio_keys": ["state_agent_obj"],
        "camera_keys": ["front_img_1"],
        "lang_keys": [],
    }

    def keys_of_type(self, type_name, emb_id):
        return list(self._BY_TYPE.get(type_name, []))

    def is_key_with_embodiment(self, key, emb_id):
        return True

    def zarr_key_to_keyname(self, key, emb_id):
        data_keys = {"actions", "state_agent_obj", "front_img_1"}
        return key if key in data_keys else None

    def normalize(self, d, emb_id):
        return d  # synthetic data already ~normalized

    def unnormalize(self, d, emb_id):
        return d


def _make_packed_raw(ep_lens, device):
    """One embodiment's raw (pre-process) packed batch dict with zarr keys."""
    T = int(sum(ep_lens))
    cu = [0]
    for n in ep_lens:
        cu.append(cu[-1] + int(n))
    return {
        "actions": torch.randn(T, ACTION_DIM, device=device),
        "state_agent_obj": torch.randn(T, STATE_DIM, device=device),
        "front_img_1": torch.rand(T, *IMG, device=device),
        "cu_seqlens": torch.tensor(cu, dtype=torch.long, device=device),
        "max_seq_len": int(max(ep_lens)),
        "seq_lens": torch.tensor(ep_lens, dtype=torch.long, device=device),
        "batch_size": len(ep_lens),
    }


def _outer_stage_inputs(ep_lens, device):
    """Direct (batch, ctx) for an outer_stage.forward call (no obs_stride)."""
    T = int(sum(ep_lens))
    cu = [0]
    for n in ep_lens:
        cu.append(cu[-1] + int(n))
    batch = {
        "actions": torch.randn(T, ACTION_DIM, device=device),
        "__obs": {
            "state_agent_obj": torch.randn(T, STATE_DIM, device=device),
            "front_img_1": torch.rand(T, *IMG, device=device),
        },
    }
    cu_t = torch.tensor(cu, dtype=torch.long, device=device)
    return batch, cu_t, int(max(ep_lens))


def _has_grad(module):
    return any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
    )


def main():
    host = socket.gethostname()
    if not torch.cuda.is_available():
        print(f"[abort] no CUDA on {host}; run through srun on a GPU node.")
        sys.exit(1)
    device = torch.device("cuda")
    print(f"[host] {host}  device={device}")

    # --- instantiate the full algo (PackedAlgoBase) from the model config ---
    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        cfg = compose(config_name=CFG_NAME)
    norm_stats = StubNormStats()
    algo = instantiate(cfg.robomimic_model, norm_stats=norm_stats, device=device)
    algo.nets.train()
    n_params = sum(p.numel() for p in algo.nets.parameters())
    print(
        f"[instantiate] OK  params={n_params / 1e6:.1f}M  coral_weight={algo.coral_weight}"
        f"  coral_include_mean={algo.coral_include_mean}  obs_stride={algo.obs_stride}"
    )

    outer = algo.outer_stage
    stage = outer.inner_stage.stages[0]  # HybridDualStreamCoralStage
    print(f"[stage] {type(stage).__name__}")
    assert type(stage).__name__ == "HybridDualStreamCoralStage"
    assert set(stage.hnet_specific.keys()) == set(DOMAINS), stage.hnet_specific.keys()

    # --- (2) zero-init cross_attn out_proj ---
    w = stage.cross_attn.out_proj.weight
    assert torch.count_nonzero(w) == 0, "cross_attn.out_proj must be zero-init"
    print("[zero-init] cross_attn.out_proj is all-zero -> A_top == A_full at init ✓")

    emb0 = DOMAINS[0]

    # --- (3) gradient contract via outer_stage.forward ---
    def fresh_forward():
        algo.nets.zero_grad(set_to_none=True)
        batch, cu, msl = _outer_stage_inputs([12, 16], device)
        ctx = HNetContext(
            cond_dict={},
            aux=[],
            inference_params=None,
            extras={},
            cu_seqlens=cu,
            max_seqlen=msl,
            embodiment_id=emb0,
        )
        outer(batch, ctx)
        return ctx

    ctx = fresh_forward()
    A_top = ctx.extras["agnostic_repr"]
    S_top = ctx.extras["specific_tokens"]
    print(f"[shapes] A_top={tuple(A_top.shape)}  S_top={tuple(S_top.shape)}")
    assert A_top.shape == S_top.shape and A_top.shape[-1] == D_MODEL

    # A_top.backward: agnostic trunk + cross_attn get grad; specific gets NONE.
    A_top.sum().backward()
    spec0 = stage.hnet_specific[emb0]
    assert _has_grad(stage.hnet_agnostic), "agnostic trunk must get grad from A_top"
    assert _has_grad(stage.cross_attn), "cross_attn must get grad from A_top"
    assert not _has_grad(
        spec0
    ), "specific H-Net got grad from A_top -> detach/one-way BROKEN"
    print("[grad-contract] A_top.backward -> agnostic+cross_attn grad, specific NONE ✓")

    ctx = fresh_forward()
    S_top = ctx.extras["specific_tokens"]
    S_top.sum().backward()
    assert _has_grad(stage.hnet_specific[emb0]), "specific must get grad from S_top"
    assert not _has_grad(
        stage.hnet_agnostic
    ), "agnostic trunk got grad from S_top -> direction BROKEN"
    assert not _has_grad(stage.cross_attn), "cross_attn got grad from S_top -> BROKEN"
    print("[grad-contract] S_top.backward -> specific grad, agnostic+cross_attn NONE ✓")

    # --- (4) full algo path with CORAL on a synthetic 2-embodiment batch ---
    algo.nets.zero_grad(set_to_none=True)
    raw = {
        DOMAINS[0]: _make_packed_raw([48, 64], device),
        DOMAINS[1]: _make_packed_raw([52, 40], device),
    }
    processed = algo.process_batch_for_training(raw)
    predictions = algo.forward_training(processed)

    ids = [get_embodiment_id(d) for d in DOMAINS]
    for i in ids:
        assert f"{i}_agnostic_repr" in predictions, f"missing agnostic_repr for emb {i}"
    print(f"[forward_training] captured agnostic_repr for emb ids {ids} ✓")

    losses = algo.compute_losses(predictions, processed)
    assert "coral_loss" in losses, f"coral_loss missing; keys={list(losses)}"
    coral = losses["coral_loss"]
    total = losses["action_loss"]
    assert torch.isfinite(coral) and float(coral) > 0, f"coral={float(coral)}"
    assert torch.isfinite(total), f"action_loss={float(total)}"
    print(
        f"[compute_losses] coral_loss={float(coral):.4f} (>0, finite) ✓  "
        f"action_loss(total, incl. coral*{algo.coral_weight})={float(total):.4f}"
    )

    total.backward()
    n_grad = sum(1 for p in algo.nets.parameters() if p.grad is not None)
    bad = [
        n
        for n, p in algo.nets.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert not bad, f"non-finite grads: {bad[:5]}"
    print(f"[backward] {n_grad} params with finite grad ✓")

    print("\n=== HYBRID DUAL-STREAM + CORAL SMOKE PASSED ===")


if __name__ == "__main__":
    main()
