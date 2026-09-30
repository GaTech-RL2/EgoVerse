"""Isolated Pi (openpi pi0.5) train-step benchmark: real model config, synthetic
batches, random init (step time does not depend on the base weights).

  python bench/bench_pi.py --variants base,compile_expert
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import hydra
import torch
from hydra import compose, initialize_config_module

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.bench_hpt import FakeNormStats  # noqa: E402


def _patch_transformers_5_image_features() -> bool:
    """The venv ships transformers 5.x while openpi pins 4.53.2, where
    `get_image_features` returned a bare tensor. Unwrap it so the benchmark can
    run; the numbers are indicative of the pinned stack, not a substitute."""
    from openpi.models_pytorch import gemma_pytorch as gp

    real = gp.PaliGemmaWithExpertModel.embed_image

    def embed_image(self, image):
        out = real(self, image)
        return getattr(out, "last_hidden_state", out)

    gp.PaliGemmaWithExpertModel.embed_image = embed_image
    return True


EMB_HUMAN = 3


def v_sdpa(cfg):
    """openpi pins the gemma tower's attention to `eager` inside sample_actions
    and never puts it back, so a train step that follows a val pass runs
    materialised-score attention. Force sdpa before every forward."""
    os.environ["BENCH_PI_SDPA"] = "1"


def v_compile_expert(cfg):
    os.environ["BENCH_PI_COMPILE"] = "expert"


def v_compile_forward(cfg):
    os.environ["BENCH_PI_COMPILE"] = "forward"


def v_compile_hook(cfg):
    """What the shipped knob does: PI.compile_for_training via Module.compile."""
    os.environ["BENCH_PI_COMPILE"] = "hook"


def v_grad_ckpt(cfg):
    os.environ["BENCH_PI_GRAD_CKPT"] = "1"


def v_fused_adam(cfg):
    os.environ["BENCH_PI_FUSED"] = "1"


VARIANTS = {
    "base": lambda cfg: None,
    "sdpa": v_sdpa,
    "compile_expert": v_compile_expert,
    "compile_forward": v_compile_forward,
    "compile_hook": v_compile_hook,
    "grad_ckpt": v_grad_ckpt,
    "fused_adam": v_fused_adam,
}


def run(args, variant):
    for key in (
        "BENCH_PI_COMPILE",
        "BENCH_PI_GRAD_CKPT",
        "BENCH_PI_FUSED",
        "BENCH_PI_SDPA",
    ):
        os.environ.pop(key, None)
    for name in variant.split("+"):
        VARIANTS[name](None)

    if args.compat_transformers5:
        _patch_transformers_5_image_features()
    torch.manual_seed(0)
    device = torch.device("cuda")
    keys = {
        "camera_keys": [f"observations.images.{c}" for c in args.cameras.split(",")],
        "proprio_keys": [f"observations.state.{args.proprio}"],
        "action_keys": [args.action_key],
        "lang_keys": [],
    }

    with initialize_config_module(
        config_module="egomimic.hydra_configs", version_base=None
    ):
        cfg = compose(
            config_name="train_zarr_cartesian_pi",
            overrides=[
                f"model={args.model}",
                "model.robomimic_model.config.pytorch_weight_path=null",
                *(args.override or []),
            ],
        )
    algo = hydra.utils.instantiate(
        cfg.model.robomimic_model, norm_stats=FakeNormStats(keys, EMB_HUMAN)
    )
    algo.device = device
    algo.nets = algo.nets.to(device)

    if os.environ.get("BENCH_PI_GRAD_CKPT"):
        algo.nets["policy"].gradient_checkpointing_enable()
    if os.environ.get("BENCH_PI_COMPILE") == "expert":
        pwe = algo.nets["policy"].paligemma_with_expert
        pwe.forward = torch.compile(pwe.forward, dynamic=False)
    if os.environ.get("BENCH_PI_COMPILE") == "hook":
        algo.compile_for_training()
    if os.environ.get("BENCH_PI_COMPILE") == "forward":
        policy = algo.nets["policy"]
        policy.forward = torch.compile(policy.forward, dynamic=False)

    params = [p for p in algo.nets.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params, lr=1e-5, fused=True if os.environ.get("BENCH_PI_FUSED") else None
    )

    b, hz = args.batch_size, args.horizon
    hw = tuple(int(x) for x in args.img.split("x"))
    raw = {
        keys["camera_keys"][0]: torch.rand(b, 3, *hw, device=device),
        keys["proprio_keys"][0]: torch.randn(b, args.proprio_dim, device=device),
        args.action_key: torch.randn(b, hz, args.action_dim, device=device),
        "annotations": [["fold the towel"] for _ in range(b)],
    }
    batch = algo.process_batch_for_training({"human_bimanual": raw})

    algo.nets.train()
    torch.cuda.reset_peak_memory_stats()

    def force_sdpa():
        if not os.environ.get("BENCH_PI_SDPA"):
            return
        pwe = algo.nets["policy"].paligemma_with_expert
        pwe.paligemma.language_model.config._attn_implementation = "sdpa"
        pwe.gemma_expert.model.config._attn_implementation = "sdpa"

    def step():
        force_sdpa()
        opt.zero_grad(set_to_none=True)
        preds = algo.forward_training(batch)
        loss = algo.compute_losses(preds, batch)["action_loss"]
        loss.backward()
        opt.step()

    t0 = time.perf_counter()
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    warm = time.perf_counter() - t0

    times = []
    for _ in range(args.steps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1e3)

    return {
        "variant": variant,
        "median_ms": round(statistics.median(times), 2),
        "warmup_s": round(warm, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "params_M": round(sum(p.numel() for p in params) / 1e6, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="pi0.5_bc_mecka_6d")
    p.add_argument("--cameras", default="front_img_1")
    p.add_argument("--proprio", default="ee_pose")
    p.add_argument("--action-key", default="actions_cartesian")
    p.add_argument("--action-dim", type=int, default=18)
    p.add_argument("--proprio-dim", type=int, default=20)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--img", default="224x224")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--override", action="append")
    p.add_argument("--variants", default="base")
    p.add_argument("--compat-transformers5", action="store_true")
    args = p.parse_args()

    rows = [run(args, v) for v in args.variants.split(",")]
    for r in rows:
        print(json.dumps(r), flush=True)
    base = rows[0]["median_ms"]
    print(f"\n{'variant':<28}{'ms':>9}{'speedup':>9}{'mem GB':>9}{'warmup s':>10}")
    for r in rows:
        print(
            f"{r['variant']:<28}{r['median_ms']:>9.2f}{base / r['median_ms']:>8.2f}x"
            f"{r['peak_mem_gb']:>9.2f}{r['warmup_s']:>10.1f}"
        )


if __name__ == "__main__":
    main()
