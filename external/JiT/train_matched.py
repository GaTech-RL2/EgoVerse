"""Distributed matched training for JiT-B/16 and latent denoising variants."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.utils import save_image

from matched_models import build_model, trainable_parameter_count
from util.crop import center_crop_arr


STOP_REQUESTED = False


def _request_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"CHECKPOINT_STOP_REQUESTED signal={signum}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        choices=("jit_b16", "endpoint_latent", "unified_latent"),
        required=True,
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8, help="Per GPU microbatch")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--base-lr", type=float, default=5e-5)
    parser.add_argument("--warmup-epochs", type=float, default=5.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--save-every-steps", type=int, default=1000)
    parser.add_argument("--val-every-steps", type=int, default=5000)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--sample-batch", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=16)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overfit-one-batch", action="store_true")
    parser.add_argument("--overfit-force-steps", type=int, default=2)
    parser.add_argument("--smoke-require-validation", action="store_true")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--wandb-mode", default="offline", choices=("online", "offline", "disabled"))
    parser.add_argument("--expected-min-params", type=int, default=0)
    parser.add_argument("--expected-max-params", type=int, default=0)
    parser.add_argument("--stop-file", default="")
    return parser.parse_args()


def init_distributed() -> Tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("Matched image training requires CUDA")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", init_method="env://")
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def is_main(rank: int) -> bool:
    return rank == 0


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    result = value.detach().float().clone()
    if world_size > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= world_size
    return result


def transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: center_crop_arr(image, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
        ]
    )


def validation_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: center_crop_arr(image, image_size)),
            transforms.PILToTensor(),
        ]
    )


def normalize(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    return images.to(device, non_blocking=True).float().div_(255.0).mul_(2.0).sub_(1.0)


@torch.no_grad()
def update_ema(ema: nn.Module, source: nn.Module, decay: float) -> None:
    source_parameters = dict(source.named_parameters())
    for name, parameter in ema.named_parameters():
        parameter.mul_(decay).add_(source_parameters[name], alpha=1.0 - decay)
    source_buffers = dict(source.named_buffers())
    for name, buffer in ema.named_buffers():
        if name in source_buffers and buffer.shape == source_buffers[name].shape:
            buffer.copy_(source_buffers[name])


def append_jsonl(path: Path, payload: Dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def atomic_checkpoint(path: Path, payload: Dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    ema: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    next_batch: int,
    optimizer_step: int,
    rank: int,
) -> None:
    if not is_main(rank):
        return
    payload = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "epoch": int(epoch),
        "next_batch": int(next_batch),
        "optimizer_step": int(optimizer_step),
    }
    atomic_checkpoint(output_dir / "checkpoint-last.pth", payload)
    print(f"CHECKPOINT_SAVED step={optimizer_step} epoch={epoch} next_batch={next_batch}", flush=True)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    ema: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[int, int, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    ema.load_state_dict(checkpoint["ema"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    return (
        int(checkpoint["epoch"]),
        int(checkpoint.get("next_batch", 0)),
        int(checkpoint["optimizer_step"]),
    )


def current_lr(
    base_lr: float,
    effective_batch: int,
    optimizer_step: int,
    updates_per_epoch: int,
    warmup_epochs: float,
) -> float:
    absolute = base_lr * effective_batch / 256.0
    warmup_updates = max(int(warmup_epochs * updates_per_epoch), 1)
    return absolute * min((optimizer_step + 1) / warmup_updates, 1.0)


def set_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


@torch.no_grad()
def validate(
    ema: nn.Module,
    loader: DataLoader,
    sampler: DistributedSampler,
    args: argparse.Namespace,
    optimizer_step: int,
    rank: int,
    world_size: int,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, float]:
    ema.eval()
    sampler.set_epoch(optimizer_step)
    metric_values: Dict[str, list[torch.Tensor]] = {}
    first_images = first_labels = None
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(args.seed + 100_000 + rank)
        for index, (images, labels) in enumerate(loader):
            if index >= args.val_batches:
                break
            images = normalize(images, device)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                metrics = ema(images, labels, optimizer_step)
            for name, value in metrics.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    metric_values.setdefault(name, []).append(value.detach().float())
            if first_images is None:
                first_images, first_labels = images, labels
    if not metric_values.get("loss"):
        raise RuntimeError("Validation loader produced no batches")
    result = {}
    for name, values in metric_values.items():
        reduced = reduce_mean(torch.stack(values).mean(), world_size)
        result["val_loss" if name == "loss" else f"val_{name}"] = float(
            reduced.item()
        )

    if is_main(rank):
        count = min(args.sample_batch, first_labels.shape[0])
        labels = first_labels[:count]
        targets = first_images[:count]
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(args.seed + 200_000)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sample_a = ema.sample(
                    labels, num_steps=args.sample_steps, cfg_scale=args.cfg_scale
                ).float()
            torch.manual_seed(args.seed + 300_000)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sample_b = ema.sample(
                    labels, num_steps=args.sample_steps, cfg_scale=args.cfg_scale
                ).float()
        result.update(
            {
                "sample_mean": float(sample_a.mean().item()),
                "sample_std": float(sample_a.std().item()),
                "sample_pairwise_mse": float((sample_a - sample_b).square().mean().item()),
                "sample_saturation": float((sample_a.abs() > 1.0).float().mean().item()),
            }
        )
        if args.architecture in {"endpoint_latent", "unified_latent"}:
            same_seed_rows = []
            for steps in (1, 2, 4, 8, 16):
                torch.manual_seed(args.seed + 400_000)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    variant = ema.sample(labels, num_steps=steps, cfg_scale=args.cfg_scale)
                same_seed_rows.append(variant.float())
            if args.architecture == "unified_latent":
                torch.manual_seed(args.seed + 500_000)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    reconstruction = ema.reconstruct(targets).float()
                result.update(
                    {
                        "reconstruction_mse": float(
                            (reconstruction - targets).square().mean().item()
                        ),
                        "reconstruction_psnr": float(
                            -10.0
                            * torch.log10(
                                ((reconstruction - targets).square().mean() / 4.0)
                                .clamp_min(1e-12)
                            ).item()
                        ),
                    }
                )
                grid = torch.cat(
                    [targets, reconstruction, *same_seed_rows, sample_b], dim=0
                )
            else:
                grid = torch.cat([targets, *same_seed_rows, sample_b], dim=0)
        else:
            grid = torch.cat([targets, sample_a, sample_b], dim=0)
        sample_path = output_dir / f"samples-step{optimizer_step:08d}.png"
        save_image(((grid.clamp(-1, 1) + 1) / 2).cpu(), sample_path, nrow=count)
        result["sample_grid"] = str(sample_path)
        numeric = [value for value in result.values() if isinstance(value, float)]
        if not all(math.isfinite(value) for value in numeric):
            raise RuntimeError(f"Validation produced non-finite metrics: {result}")
        # JiT intentionally zero-initializes its output layer. Under the exact
        # multi-epoch warmup, its first smoke-step sample can therefore be a
        # finite constant image; record that fact rather than rejecting a valid
        # train-plus-sampling path before the optimizer has moved appreciably.
    barrier(world_size)
    if is_main(rank):
        print("VALIDATION_METRICS " + json.dumps(result, sort_keys=True), flush=True)
    return result


def maybe_wandb(args: argparse.Namespace, rank: int, config: Dict):
    if not is_main(rank) or args.wandb_mode == "disabled" or not args.wandb_project:
        return None
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or None,
        dir=args.output_dir,
        config=config,
        mode=args.wandb_mode,
        resume="allow",
    )
    run.define_metric("optimizer_step")
    run.define_metric("train/*", step_metric="optimizer_step")
    run.define_metric("validation/*", step_metric="optimizer_step")
    return run


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.grad_accum <= 0:
        raise ValueError("batch-size and grad-accum must be positive")
    rank, world_size, local_rank, device = init_distributed()
    signal.signal(signal.SIGUSR1, _request_stop)
    output_dir = Path(args.output_dir)
    if is_main(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(world_size)

    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.optimize_ddp = False

    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_path, "train"), transform=transform(args.image_size)
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_path, "val"), transform=validation_transform(args.image_size)
    )
    if train_dataset.classes != val_dataset.classes:
        raise RuntimeError("ImageNet train/val class mappings differ")
    if len(train_dataset.classes) != args.num_classes:
        raise RuntimeError(
            f"Expected {args.num_classes} classes, found {len(train_dataset.classes)}"
        )
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=max(args.sample_batch, 1),
        num_workers=min(args.num_workers, 4),
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(args.architecture, args.image_size, args.num_classes).to(device)
    parameter_count = trainable_parameter_count(model)
    if args.expected_min_params and parameter_count < args.expected_min_params:
        raise RuntimeError(f"Parameter count {parameter_count} below required minimum")
    if args.expected_max_params and parameter_count > args.expected_max_params:
        raise RuntimeError(f"Parameter count {parameter_count} above required maximum")
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    distributed_model = DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False
    )
    parameter_groups = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.ndim > 1 and not name.endswith("bias")
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and (parameter.ndim <= 1 or name.endswith("bias"))
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(parameter_groups, lr=0.0, betas=(0.9, 0.95))

    effective_batch = args.batch_size * world_size * args.grad_accum
    updates_per_epoch = len(train_loader) // args.grad_accum
    if updates_per_epoch <= 0:
        raise RuntimeError("Gradient accumulation exceeds the epoch's microbatch count")
    config = {
        **vars(args),
        "world_size": world_size,
        "effective_batch": effective_batch,
        "updates_per_epoch": updates_per_epoch,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "trainable_parameters": parameter_count,
        "torch_version": torch.__version__,
    }
    if is_main(rank):
        (output_dir / "resolved_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("RESOLVED_CONFIG " + json.dumps(config, sort_keys=True), flush=True)
    run = maybe_wandb(args, rank, config)

    epoch = 0
    next_batch = 0
    optimizer_step = 0
    checkpoint_path = output_dir / "checkpoint-last.pth"
    if args.resume and checkpoint_path.exists():
        epoch, next_batch, optimizer_step = load_checkpoint(
            checkpoint_path, model, ema, optimizer, device
        )
        print(
            f"CHECKPOINT_LOADED step={optimizer_step} epoch={epoch} next_batch={next_batch}",
            flush=True,
        )

    first_overfit_loss: Optional[float] = None
    last_overfit_loss: Optional[float] = None
    cached_batch = None
    validations_run = 0
    log_path = output_dir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    training_start = time.time()
    done = False

    while epoch < args.epochs and not done:
        train_sampler.set_epoch(epoch)
        micro_in_update = 0
        for batch_index, batch in enumerate(train_loader):
            if batch_index < next_batch:
                continue
            if args.overfit_one_batch:
                if cached_batch is None:
                    cached_batch = batch
                batch = cached_batch
            images, labels = batch
            images = normalize(images, device)
            labels = labels.to(device, non_blocking=True)
            micro_in_update += 1
            sync_now = micro_in_update == args.grad_accum
            context = contextlib.nullcontext() if sync_now else distributed_model.no_sync()
            force_steps = args.overfit_force_steps if args.overfit_one_batch else None
            rng_context = (
                torch.random.fork_rng(devices=[device.index])
                if args.overfit_one_batch else contextlib.nullcontext()
            )
            with context, rng_context:
                if args.overfit_one_batch:
                    torch.manual_seed(args.seed + rank)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    metrics = distributed_model(
                        images, labels, optimizer_step + 1, force_steps=force_steps
                    )
                    scaled_loss = metrics["loss"] / args.grad_accum
                scaled_loss.backward()
            if not sync_now:
                continue

            lr = current_lr(
                args.base_lr,
                effective_batch,
                optimizer_step,
                updates_per_epoch,
                0.0 if args.overfit_one_batch else args.warmup_epochs,
            )
            set_lr(optimizer, lr)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            micro_in_update = 0
            update_ema(ema, model, args.ema_decay)
            next_batch = batch_index + 1

            reduced = {
                name: float(reduce_mean(value, world_size).item())
                for name, value in metrics.items()
            }
            reduced.update(
                {
                    "optimizer_step": optimizer_step,
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "lr": lr,
                    "elapsed_seconds": time.time() - training_start,
                }
            )
            if args.overfit_one_batch:
                if first_overfit_loss is None:
                    first_overfit_loss = reduced["loss"]
                last_overfit_loss = reduced["loss"]
            if is_main(rank) and (
                optimizer_step == 1 or optimizer_step % args.log_every_steps == 0
            ):
                print("TRAIN_METRICS " + json.dumps(reduced, sort_keys=True), flush=True)
                append_jsonl(log_path, {"split": "train", **reduced})
                if run is not None:
                    run.log(
                        {
                            "optimizer_step": optimizer_step,
                            **{f"train/{key}": value for key, value in reduced.items()},
                        },
                        step=optimizer_step,
                    )

            should_validate = (
                args.val_every_steps > 0 and optimizer_step % args.val_every_steps == 0
            )
            if should_validate:
                validation = validate(
                    ema,
                    val_loader,
                    val_sampler,
                    args,
                    optimizer_step,
                    rank,
                    world_size,
                    device,
                    output_dir,
                )
                validations_run += 1
                if is_main(rank):
                    append_jsonl(
                        log_path,
                        {"split": "validation", "optimizer_step": optimizer_step, **validation},
                    )
                    if run is not None:
                        wandb_payload = {
                            "optimizer_step": optimizer_step,
                            **{
                                f"validation/{key}": value
                                for key, value in validation.items()
                                if isinstance(value, (int, float))
                            },
                        }
                        if validation.get("sample_grid"):
                            import wandb

                            wandb_payload["validation/sample_grid"] = wandb.Image(
                                validation["sample_grid"]
                            )
                        run.log(wandb_payload, step=optimizer_step)
                model.train()

            if optimizer_step % args.save_every_steps == 0 or STOP_REQUESTED:
                save_checkpoint(
                    output_dir,
                    model,
                    ema,
                    optimizer,
                    args,
                    epoch,
                    next_batch,
                    optimizer_step,
                    rank,
                )
                barrier(world_size)

            stop_file_requested = bool(args.stop_file) and Path(args.stop_file).exists()
            if stop_file_requested:
                save_checkpoint(
                    output_dir,
                    model,
                    ema,
                    optimizer,
                    args,
                    epoch,
                    next_batch,
                    optimizer_step,
                    rank,
                )
                barrier(world_size)
            if STOP_REQUESTED or stop_file_requested:
                done = True
                break
            if args.max_optimizer_steps and optimizer_step >= args.max_optimizer_steps:
                done = True
                break
            if args.overfit_one_batch:
                next_batch = 0
                if not done:
                    continue

        if not done:
            if micro_in_update:
                optimizer.zero_grad(set_to_none=True)
            epoch += 1
            next_batch = 0

    save_checkpoint(
        output_dir, model, ema, optimizer, args, epoch, next_batch, optimizer_step, rank
    )
    barrier(world_size)
    if args.overfit_one_batch:
        if first_overfit_loss is None or last_overfit_loss is None:
            raise RuntimeError("Overfit run produced no optimizer steps")
        if not last_overfit_loss < first_overfit_loss:
            raise RuntimeError(
                f"Overfit loss did not improve: first={first_overfit_loss} last={last_overfit_loss}"
            )
        if is_main(rank):
            print(
                f"OVERFIT_GATE_PASSED first={first_overfit_loss:.8f} "
                f"last={last_overfit_loss:.8f}",
                flush=True,
            )
    if args.smoke_require_validation:
        if validations_run < 1:
            raise RuntimeError("Smoke ended without scheduled post-training validation")
        if is_main(rank):
            print(f"SMOKE_GATE_PASSED validations={validations_run}", flush=True)
    if run is not None:
        run.finish()
    barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
