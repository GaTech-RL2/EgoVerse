"""Isolated HPT train-step benchmark: real model config, synthetic batches.

No dataloader, no Lightning -- just forward_training + compute_losses +
backward + optimizer step, so a change to the model path shows up undiluted.

  python bench/bench_hpt.py --variants base,tf32,sdpa,compile_trunk

Variants are monkeypatches applied before the model is built, so an A/B needs
no source edit; the winners get ported into egomimic/ afterwards.
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

EMB_HUMAN = 3
EMB_EVA = 6


class FakeNormStats:
    """The slice of MultiDataset that HPT.__init__ reads: key discovery."""

    norm_mode = "quantile"

    def __init__(self, keys: dict, embodiment_id: int):
        self._keys = keys
        self._emb = embodiment_id

    def keys_of_type(self, kind, embodiment_id):
        return list(self._keys.get(kind, []))

    def is_key_with_embodiment(self, key, embodiment_id):
        return True

    def zarr_key_to_keyname(self, key, embodiment_id):
        return key

    def unnormalize(self, preds, embodiment_id):
        return preds


# ---------------------------------------------------------------- variants


def v_tf32(cfg):
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def v_sdpa(cfg):
    """need_weights=False on the flow head's attentions.

    nn.MultiheadAttention defaults to need_weights=True, which forces the slow
    path: it materialises B x H x L x L scores and averages them over heads,
    instead of dispatching to scaled_dot_product_attention.
    """
    from egomimic.models import denoising_nets as dn

    def forward_cross(self, x, cond):
        res = x
        x = self.ln1(x)
        x, _ = self.mha(x, x, x, need_weights=False)
        x = x + res
        res = x
        x = self.ln2(x)
        x, _ = self.cmha(x, cond, cond, need_weights=False)
        x = x + res
        res = x
        x = self.ln3(x)
        x = self.mlp(x)
        return x + res

    dn.CrossBlock.forward_cross = forward_cross


def v_poscache(cfg):
    """Cache the sinusoid position tables instead of rebuilding them on the CPU
    (and H2D-copying them) inside every forward."""
    import egomimic.algo.hpt as hpt_mod
    import egomimic.utils.tensor_utils as tu

    real = tu.get_sinusoid_encoding_table
    cache: dict = {}

    def cached(position_start, position_end, d_hid):
        key = (position_start, position_end, d_hid)
        if key not in cache:
            cache[key] = real(position_start, position_end, d_hid)
        return cache[key]

    tu.get_sinusoid_encoding_table = cached
    hpt_mod.get_sinusoid_encoding_table = cached


def v_noclone(cfg):
    """_clone_batch deep-copies every image tensor each step; only the optional
    OT loss reads the copy."""
    import egomimic.algo.hpt as hpt_mod

    hpt_mod.HPT._clone_batch = lambda self, batch: batch


def v_nozerosync(cfg):
    """`if not torch.all(img == 0)` in _robomimic_to_hpt_data is a blocking
    device->host sync per camera per step. Always augment instead."""
    import egomimic.algo.hpt as hpt_mod

    def patched(self, batch, cam_keys, proprio_keys, lang_keys, ac_key, aux_ac_keys=[]):
        data = {}
        for key in proprio_keys:
            if key in batch:
                short = key.rsplit(".", 1)[-1]
                modality = f"state_{short}"
                value = batch[key]
                data[modality] = value.unsqueeze(1) if value.ndim == 2 else value
        for key in cam_keys:
            if key in batch:
                short = key.rsplit(".", 1)[-1]
                _data = self._apply_image_augs(batch[key], short)
                data[short] = _data.unsqueeze(1).unsqueeze(1)
        for key in lang_keys:
            if key in batch:
                data[key] = batch[key]
        if "sampled_prompt" in batch:
            data[self.annotation_modality] = batch["sampled_prompt"]
        data["is_6dof"] = self.is_6dof
        data["pad_mask"] = batch["pad_mask"]
        data["embodiment"] = batch["embodiment"]
        for aux in aux_ac_keys:
            data[aux] = batch[aux]
        data["action"] = batch[self.shared_ac_key or ac_key]
        return data

    hpt_mod.HPT._robomimic_to_hpt_data = patched


# Every attribute a variant may replace. Without restoring these between
# variants the second row of a sweep measures the first row plus itself, and
# the per-variant column becomes a running total.
def _patch_targets():
    import egomimic.algo.hpt as hpt_mod
    import egomimic.utils.tensor_utils as tu
    from egomimic.models import denoising_nets, image_augs

    return [
        (image_augs.PerSampleAugs, "forward"),
        (hpt_mod.HPT, "_apply_image_augs"),
        (hpt_mod.HPT, "_robomimic_to_hpt_data"),
        (hpt_mod.HPT, "_clone_batch"),
        (denoising_nets.CrossBlock, "forward_cross"),
        (tu, "get_sinusoid_encoding_table"),
        (hpt_mod, "get_sinusoid_encoding_table"),
    ]


_PRISTINE = None
_PRISTINE_TF32 = None


def _restore_patched_globals():
    global _PRISTINE, _PRISTINE_TF32
    targets = _patch_targets()
    if _PRISTINE is None:
        _PRISTINE = [getattr(obj, name) for obj, name in targets]
        _PRISTINE_TF32 = (
            torch.get_float32_matmul_precision(),
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.allow_tf32,
        )
        return
    for (obj, name), original in zip(targets, _PRISTINE):
        setattr(obj, name, original)
    precision, benchmark, cudnn_tf32 = _PRISTINE_TF32
    torch.set_float32_matmul_precision(precision)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.allow_tf32 = cudnn_tf32


def v_loopaugs(cfg):
    """The pre-2026-09-19 image-aug path: torchvision's ColorJitter called once
    per image. Kept so the table can show what the vectorized one replaced."""
    import torch

    from egomimic.models import image_augs

    def forward(self, images):
        return torch.stack([self.augs(image) for image in images])

    image_augs.PerSampleAugs.forward = forward


def v_noaugs(cfg):
    """Upper bound for the image-aug path: PerSampleAugs loops ColorJitter over
    the batch one image at a time."""
    import egomimic.algo.hpt as hpt_mod

    hpt_mod.HPT._apply_image_augs = lambda self, images, short: images


def v_channels_last(cfg):
    os.environ["BENCH_CHANNELS_LAST"] = "1"


def v_compile_trunk(cfg):
    os.environ["BENCH_COMPILE"] = "trunk"


def v_compile_head(cfg):
    os.environ["BENCH_COMPILE"] = "head"


def v_compile_both(cfg):
    os.environ["BENCH_COMPILE"] = "both"


def v_compile_block(cfg):
    os.environ["BENCH_COMPILE"] = "block"


def v_compile_all(cfg):
    os.environ["BENCH_COMPILE"] = "all"


def v_compile_hook(cfg):
    """What the shipped knob does: HPT.compile_for_training via nn.Module.compile."""
    os.environ["BENCH_COMPILE"] = "hook"


def v_maxautotune(cfg):
    os.environ["BENCH_COMPILE_MODE"] = "max-autotune-no-cudagraphs"


def v_cudagraphs(cfg):
    os.environ["BENCH_COMPILE_MODE"] = "reduce-overhead"


VARIANTS = {
    "base": lambda cfg: None,
    "tf32": v_tf32,
    "sdpa": v_sdpa,
    "poscache": v_poscache,
    "noclone": v_noclone,
    "nozerosync": v_nozerosync,
    "channels_last": v_channels_last,
    "noaugs": v_noaugs,
    "loopaugs": v_loopaugs,
    "compile_trunk": v_compile_trunk,
    "compile_head": v_compile_head,
    "compile_both": v_compile_both,
    "compile_block": v_compile_block,
    "compile_all": v_compile_all,
    "compile_hook": v_compile_hook,
    "maxautotune": v_maxautotune,
    "cudagraphs": v_cudagraphs,
}


# ---------------------------------------------------------------- build


def build_cfg(model_name: str, overrides: list[str]):
    with initialize_config_module(
        config_module="egomimic.hydra_configs", version_base=None
    ):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"model={model_name}", *overrides],
        )


def build_algo(cfg, embodiment: str, emb_id: int, keys: dict, device):
    node = cfg.model.robomimic_model
    algo = hydra.utils.instantiate(node, norm_stats=FakeNormStats(keys, emb_id))
    algo.device = device
    algo.nets = algo.nets.to(device)
    return algo


def synth_batch(
    keys,
    emb_id,
    batch_size,
    device,
    action_dim,
    proprio_dim,
    horizon,
    img_hw,
    history_len,
    prompt,
):
    b = {}
    for k in keys["camera_keys"]:
        b[k] = torch.rand(batch_size, 3, *img_hw, device=device)
    for k in keys["proprio_keys"]:
        shape = (
            (batch_size, proprio_dim)
            if history_len == 1
            else (batch_size, history_len, proprio_dim)
        )
        b[k] = torch.randn(*shape, device=device)
    for k in keys["action_keys"]:
        b[k] = torch.randn(batch_size, horizon, action_dim, device=device)
    b["pad_mask"] = torch.ones(batch_size, horizon, 1, device=device)
    b["embodiment"] = torch.tensor([emb_id], device=device, dtype=torch.int64)
    if prompt is not None:
        b["sampled_prompt"] = [prompt] * batch_size
    return {emb_id: b}


def maybe_compile(algo):
    what = os.environ.get("BENCH_COMPILE")
    if not what:
        return
    mode = os.environ.get("BENCH_COMPILE_MODE") or None
    kw = {"dynamic": False}
    if mode:
        kw["mode"] = mode
    policy = algo.nets["policy"]
    if what == "hook":
        algo.compile_for_training(mode=mode, dynamic=False)
        return
    if what == "all":
        policy.trunk["trunk"] = torch.compile(policy.trunk["trunk"], **kw)
        for name, head in policy.heads.items():
            head.model = torch.compile(head.model, **kw)
        for name, enc in policy.encoders.items():
            policy.encoders[name] = torch.compile(enc, **kw)
        for name, st in policy.stems.items():
            if name.endswith("_annotation"):
                continue  # frozen + text-feature cached; input is list[str]
            policy.stems[name] = torch.compile(st, **kw)
        return
    if what in ("trunk", "both"):
        policy.trunk["trunk"] = torch.compile(policy.trunk["trunk"], **kw)
    if what in ("head", "both"):
        for name, head in policy.heads.items():
            head.model = torch.compile(head.model, **kw)
    if what == "block":
        # per-block compile: one small graph reused 23x, so compile time stays
        # flat in depth and the whole trunk still runs compiled kernels
        blocks = policy.trunk["trunk"].blocks
        for i, blk in enumerate(blocks):
            blocks[i] = torch.compile(blk, **kw)
        for name, head in policy.heads.items():
            for i, layer in enumerate(head.model.layers):
                head.model.layers[i] = torch.compile(layer, **kw)


def maybe_channels_last(algo):
    if os.environ.get("BENCH_CHANNELS_LAST"):
        algo.nets["policy"].encoders.to(memory_format=torch.channels_last)


# ---------------------------------------------------------------- run


def run(args, variant: str):
    for key in ("BENCH_COMPILE", "BENCH_COMPILE_MODE", "BENCH_CHANNELS_LAST"):
        os.environ.pop(key, None)
    _restore_patched_globals()
    torch.manual_seed(0)
    device = torch.device("cuda")

    emb_id = EMB_EVA if args.embodiment == "eva_bimanual" else EMB_HUMAN
    keys = {
        "camera_keys": [f"observations.images.{c}" for c in args.cameras.split(",")],
        "proprio_keys": [f"observations.state.{args.proprio}"],
        "action_keys": [args.action_key],
        "lang_keys": [],
    }

    for name in variant.split("+"):
        VARIANTS[name](None)

    cfg = build_cfg(args.model, args.override or [])
    algo = build_algo(cfg, args.embodiment, emb_id, keys, device)
    maybe_channels_last(algo)
    maybe_compile(algo)

    params = [p for p in algo.nets.parameters() if p.requires_grad]
    # An explicit fused=False would also switch OFF the foreach auto-default,
    # i.e. benchmark the single-tensor path against the real config's foreach.
    adam_kw = {"foreach": None, "fused": None}
    if args.adam == "fused":
        adam_kw["fused"] = True
    elif args.adam == "foreach":
        adam_kw["foreach"] = True
    elif args.adam == "single":
        adam_kw["foreach"] = False
    opt = torch.optim.AdamW(params, lr=1e-5, **adam_kw)

    stem = algo.nets["policy"].stems
    hist = 1
    for n, s in stem.items():
        if n.endswith(f"_state_{args.proprio}"):
            hist = int(getattr(s, "history_len", 1))
    prompt = "fold the towel" if algo.annotation_key else None
    batch = synth_batch(
        keys,
        emb_id,
        args.batch_size,
        device,
        args.action_dim,
        args.proprio_dim,
        args.horizon,
        tuple(int(x) for x in args.img.split("x")),
        hist,
        prompt,
    )

    algo.nets.train()
    torch.cuda.reset_peak_memory_stats()
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.fp32)

    def grad_norm_hooks():
        """What ModelWrapper does between backward and the optimizer step.

        on_after_backward and on_before_optimizer_step each run a full
        clip_grad_norm_(inf) over every parameter and then read it on the host.
        """
        if args.grad_norm == "none":
            return
        params = list(algo.nets.parameters())
        n1 = torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf"))
        float(n1)
        if args.grad_norm == "current":
            n2 = torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf"))
            float(n2)

    def step():
        opt.zero_grad(set_to_none=True)
        with amp:
            preds = algo.forward_training({k: dict(v) for k, v in batch.items()})
            losses = algo.compute_losses(preds, batch)
        losses["action_loss"].backward()
        grad_norm_hooks()
        opt.step()
        return losses["action_loss"]

    t_compile = time.perf_counter()
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    t_compile = time.perf_counter() - t_compile

    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
        ) as prof:
            for _ in range(3):
                step()
            torch.cuda.synchronize()
        print(
            prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=args.profile_rows
            )
        )
        print(
            prof.key_averages().table(
                sort_by="self_cpu_time_total", row_limit=args.profile_rows
            )
        )

    times = []
    for _ in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    return {
        "variant": variant,
        "median_ms": round(statistics.median(times), 2),
        "p10_ms": round(sorted(times)[len(times) // 10], 2),
        "warmup_s": round(t_compile, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "params_M": round(sum(p.numel() for p in params) / 1e6, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="hpt_bc_mecka_6d_300M")
    p.add_argument("--embodiment", default="human_bimanual")
    p.add_argument("--cameras", default="front_img_1")
    p.add_argument("--proprio", default="ee_pose")
    p.add_argument("--action-key", default="actions_cartesian")
    p.add_argument("--action-dim", type=int, default=18)
    p.add_argument("--proprio-dim", type=int, default=20)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--img", default="224x224")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--fp32", action="store_true")
    p.add_argument(
        "--adam", default="default", choices=["default", "foreach", "fused", "single"]
    )
    p.add_argument("--grad-norm", default="none", choices=["none", "once", "current"])
    p.add_argument("--override", action="append")
    p.add_argument("--variants", default="base")
    p.add_argument("--json", default=None)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-rows", type=int, default=25)
    args = p.parse_args()

    rows = []
    for variant in args.variants.split(","):
        rows.append(run(args, variant))
        print(json.dumps(rows[-1]), flush=True)

    base = rows[0]["median_ms"]
    print(f"\n{'variant':<34}{'ms':>9}{'speedup':>9}{'mem GB':>9}{'warmup s':>10}")
    for r in rows:
        print(
            f"{r['variant']:<34}{r['median_ms']:>9.2f}{base / r['median_ms']:>8.2f}x"
            f"{r['peak_mem_gb']:>9.2f}{r['warmup_s']:>10.1f}"
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
