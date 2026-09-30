"""Does compile_for_training change the numbers?

Same seed, same synthetic batch, eager vs compiled: the loss and every gradient
must agree to bf16-autocast tolerance. A mismatch bigger than that means the
compiled graph is not the model we think it is.
"""

from __future__ import annotations

import argparse
import os
import sys

import hydra
import torch
from hydra import compose, initialize_config_module

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn  # noqa: E402

from bench.bench_hpt import EMB_HUMAN, FakeNormStats, synth_batch  # noqa: E402
from egomimic.models.fm_policy import FMPolicy  # noqa: E402


def make_deterministic(algo, noise, time):
    """Strip every source of randomness from the train step.

    Inductor does not reproduce eager's RNG stream, so a stochastic model gives
    two different losses for reasons that have nothing to do with whether the
    compiled graph is right. Dropout, drop-path, the colour jitter and the
    flow-matching noise/time all get pinned, and what is left must match.
    """
    for module in algo.nets.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        if isinstance(module, nn.MultiheadAttention):
            module.dropout = 0.0
        if type(module).__name__ == "DropPath":
            module.drop_prob = 0.0
    algo.train_image_augs = algo.eval_image_augs

    def predict(self, actions, global_cond):
        t = time.to(actions.device).view(-1, 1, 1)
        x_t = t * noise + (1 - t) * actions
        return self.model(x_t, time.to(actions.device), global_cond), noise - actions

    FMPolicy.predict = predict


def build(keys, compile_it, mode):
    torch.manual_seed(0)
    with initialize_config_module(
        config_module="egomimic.hydra_configs", version_base=None
    ):
        cfg = compose(
            config_name="train_zarr_cartesian", overrides=["model=hpt_bc_mecka_6d_300M"]
        )
    algo = hydra.utils.instantiate(
        cfg.model.robomimic_model, norm_stats=FakeNormStats(keys, EMB_HUMAN)
    )
    algo.device = torch.device("cuda")
    algo.nets = algo.nets.to("cuda")
    if compile_it:
        algo.compile_for_training(mode=mode)
    return algo


def run(algo, batch, seed, bf16=True):
    algo.nets.train()
    algo.nets.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        preds = algo.forward_training({k: dict(v) for k, v in batch.items()})
        loss = algo.compute_losses(preds, batch)["action_loss"]
    loss.backward()
    grads = {
        n: p.grad.detach().clone()
        for n, p in algo.nets.named_parameters()
        if p.grad is not None
    }
    return float(loss), grads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--rtol", type=float, default=2e-2)
    ap.add_argument("--atol", type=float, default=2e-3)
    ap.add_argument("--min-cos", type=float, default=0.999)
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    keys = {
        "camera_keys": ["observations.images.front_img_1"],
        "proprio_keys": ["observations.state.ee_pose"],
        "action_keys": ["actions_cartesian"],
        "lang_keys": [],
    }
    torch.manual_seed(1234)
    batch = synth_batch(
        keys,
        EMB_HUMAN,
        args.batch_size,
        torch.device("cuda"),
        18,
        20,
        100,
        (224, 224),
        1,
        "fold the towel",
    )

    noise = torch.randn(args.batch_size, 100, 18, device="cuda")
    time = torch.rand(args.batch_size, device="cuda") * 0.9 + 0.05

    eager = build(keys, False, None)
    make_deterministic(eager, noise, time)
    compiled = build(keys, True, args.mode)
    make_deterministic(compiled, noise, time)

    # The comparison is meaningless unless both copies start from the same
    # weights, and instantiate() draws from the global RNG.
    a = dict(eager.nets.named_parameters())
    b = dict(compiled.nets.named_parameters())
    assert set(a) == set(b)
    bad = [n for n in a if not torch.equal(a[n], b[n])]
    assert not bad, f"{len(bad)} weights differ before the step, e.g. {bad[:3]}"

    eager_loss, eager_grads = run(eager, batch, seed=7, bf16=not args.fp32)
    comp_loss, comp_grads = run(compiled, batch, seed=7, bf16=not args.fp32)

    print(f"loss  eager={eager_loss:.6f}  compiled={comp_loss:.6f}")
    assert abs(eager_loss - comp_loss) <= args.atol + args.rtol * abs(
        eager_loss
    ), "loss differs beyond tolerance"

    assert set(eager_grads) == set(comp_grads), "gradient key sets differ"

    # Per tensor: max |delta| against that tensor's own scale, and the cosine of
    # the two gradient vectors. The elementwise ratio is noisy under bf16
    # autocast (inductor fuses and reorders reductions); the direction is what
    # the optimizer actually follows, so it is the honest check.
    rows, dead = [], []
    for name, g in eager_grads.items():
        h = comp_grads[name]
        if g.count_nonzero() == 0 and h.count_nonzero() == 0:
            dead.append(name)  # unused parameter in both; no direction to compare
            continue
        scale = g.abs().max().clamp(min=1e-8)
        rel = ((g - h).abs().max() / scale).item()
        cos = torch.nn.functional.cosine_similarity(
            g.flatten().float(), h.flatten().float(), dim=0
        ).item()
        rows.append((rel, cos, name))
    if dead:
        print(f"{len(dead)} parameters have a zero gradient in both, e.g. {dead[:2]}")
    rows.sort(reverse=True)
    print(f"{len(rows)} gradient tensors")
    print("worst by relative max-abs delta:")
    for rel, cos, name in rows[:5]:
        print(f"  rel={rel:.2e}  cos={cos:.6f}  {name}")
    min_cos = min(c for _, c, _ in rows)
    worst_name = min(rows, key=lambda r: r[1])[2]
    print(
        f"min cosine similarity: {min_cos:.6f} ({worst_name}) "
        f"|eager|={eager_grads[worst_name].norm():.3e} "
        f"|compiled|={comp_grads[worst_name].norm():.3e}"
    )
    assert min_cos >= args.min_cos, "a gradient points somewhere else"
    print("PARITY OK")


if __name__ == "__main__":
    main()
