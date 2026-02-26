import os
import platform
import resource
import time

import numpy as np
import psutil
import torch


def _fmt_bytes(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(n) < 1024.0:
            return f"{n:6.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def _rss_bytes():
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb if platform.system() == "Darwin" else kb * 1024


def _cuda_bytes():
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(dev)
        res = torch.cuda.memory_reserved(dev)
        peak = torch.cuda.max_memory_allocated(dev)
        return alloc, res, peak
    return 0, 0, 0


def _print_mem(label, t0):
    rss = _rss_bytes()
    alloc, res, peak = _cuda_bytes()
    print(
        f"[{label:20s}]  Δt={time.time() - t0:6.2f}s | "
        f"RSS={_fmt_bytes(rss)} | "
        f"CUDA alloc={_fmt_bytes(alloc)}, res={_fmt_bytes(res)}, peak={_fmt_bytes(peak)}"
    )


def get_memory_summary():
    """
    Returns a dictionary containing current memory statistics.

    Returns:
        dict: Dictionary with keys:
            - 'rss_gb': RSS memory in GB (float)
            - 'cuda_alloc_gb': CUDA allocated memory in GB (float, 0 if CUDA not available)
            - 'cuda_reserved_gb': CUDA reserved memory in GB (float, 0 if CUDA not available)
            - 'cuda_peak_gb': CUDA peak allocated memory in GB (float, 0 if CUDA not available)
    """
    process = psutil.Process(os.getpid())
    rss_gb = process.memory_info().rss / 1e9

    cuda_alloc, cuda_reserved, cuda_peak = _cuda_bytes()
    cuda_alloc_gb = cuda_alloc / 1e9 if cuda_alloc > 0 else 0.0
    cuda_reserved_gb = cuda_reserved / 1e9 if cuda_reserved > 0 else 0.0
    cuda_peak_gb = cuda_peak / 1e9 if cuda_peak > 0 else 0.0

    return {
        "rss_gb": rss_gb,
        "cuda_alloc_gb": cuda_alloc_gb,
        "cuda_reserved_gb": cuda_reserved_gb,
        "cuda_peak_gb": cuda_peak_gb,
    }


def print_epoch_memory_summary(epoch_stats_list):
    """
    Print a formatted summary table of memory statistics across epochs.

    Args:
        epoch_stats_list: List of dictionaries, each containing:
            - 'epoch': int, epoch number
            - 'rss_gb': float, RSS memory in GB
            - 'cuda_alloc_gb': float, CUDA allocated memory in GB
            - 'cuda_reserved_gb': float, CUDA reserved memory in GB
            - 'cuda_peak_gb': float, CUDA peak allocated memory in GB
    """
    print("\n" + "=" * 80)
    print("FINAL TRAINING MEMORY STATISTICS - PER EPOCH SUMMARY")
    print("=" * 80)

    if not epoch_stats_list:
        print("No epoch memory statistics recorded.")
        return

    # Print table header
    print(
        f"\n{'Epoch':<8} {'RSS (GB)':<12} {'GPU Alloc (GB)':<16} {'GPU Reserved (GB)':<18} {'GPU Peak (GB)':<15}"
    )
    print("-" * 80)

    # Print stats for each epoch
    for stat in epoch_stats_list:
        epoch = stat["epoch"]
        rss = stat["rss_gb"]
        alloc = stat["cuda_alloc_gb"]
        reserved = stat["cuda_reserved_gb"]
        peak = stat["cuda_peak_gb"]
        print(f"{epoch:<8} {rss:<12.2f} {alloc:<16.2f} {reserved:<18.2f} {peak:<15.2f}")

    # Calculate and print summary statistics
    if len(epoch_stats_list) > 0:
        rss_values = [s["rss_gb"] for s in epoch_stats_list]
        alloc_values = [
            s["cuda_alloc_gb"] for s in epoch_stats_list if s["cuda_alloc_gb"] > 0
        ]
        reserved_values = [
            s["cuda_reserved_gb"] for s in epoch_stats_list if s["cuda_reserved_gb"] > 0
        ]
        peak_values = [
            s["cuda_peak_gb"] for s in epoch_stats_list if s["cuda_peak_gb"] > 0
        ]

        print("-" * 80)
        print(
            f"{'Min':<8} {min(rss_values):<12.2f} {min(alloc_values) if alloc_values else 0:<16.2f} {min(reserved_values) if reserved_values else 0:<18.2f} {min(peak_values) if peak_values else 0:<15.2f}"
        )
        print(
            f"{'Max':<8} {max(rss_values):<12.2f} {max(alloc_values) if alloc_values else 0:<16.2f} {max(reserved_values) if reserved_values else 0:<18.2f} {max(peak_values) if peak_values else 0:<15.2f}"
        )
        print(
            f"{'Avg':<8} {np.mean(rss_values):<12.2f} {np.mean(alloc_values) if alloc_values else 0:<16.2f} {np.mean(reserved_values) if reserved_values else 0:<18.2f} {np.mean(peak_values) if peak_values else 0:<15.2f}"
        )

    # Final current memory stats
    final_stats = get_memory_summary()
    print("\n" + "-" * 80)
    print("Final Memory (at training end):")
    print(f"  RSS: {final_stats['rss_gb']:.2f} GB")
    print(f"  GPU Allocated: {final_stats['cuda_alloc_gb']:.2f} GB")
    print(f"  GPU Reserved: {final_stats['cuda_reserved_gb']:.2f} GB")
    print(f"  GPU Peak: {final_stats['cuda_peak_gb']:.2f} GB")

    print("=" * 80 + "\n")


def print_parameter_gradient_status(model, epoch=None, print_all=False):
    """
    Print the gradient status (requires_grad) of all model parameters.

    Args:
        model: PyTorch model, LightningModule, or any object with named_parameters() method
        epoch: Optional epoch number to include in output
        print_all: If True, print all parameters. If False, only print summary stats.
    """
    import torch.nn as nn

    total_params = 0
    trainable_params = 0
    frozen_params = 0
    param_info = []

    # Try to get named_parameters - handle different model structures
    try:
        if hasattr(model, "named_parameters"):
            named_params = model.named_parameters()
        elif hasattr(model, "nets") and isinstance(model.nets, nn.ModuleDict):
            # Handle robomimic-style models with nets dictionary
            named_params = []
            for net_name, net in model.nets.items():
                for name, param in net.named_parameters():
                    named_params.append((f"{net_name}.{name}", param))
        elif hasattr(model, "model") and hasattr(model.model, "named_parameters"):
            # Handle wrapped models
            named_params = model.model.named_parameters()
        else:
            print(
                f"\n⚠️  ERROR: Cannot access parameters from model type: {type(model)}"
            )
            print(
                f"Model has attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}"
            )
            return
    except Exception as e:
        print(f"\n⚠️  ERROR: Failed to access model parameters: {e}")
        return

    for name, param in named_params:
        total_params += 1
        if param.requires_grad:
            trainable_params += 1
        else:
            frozen_params += 1

        if print_all:
            param_info.append(
                {
                    "name": name,
                    "requires_grad": param.requires_grad,
                    "shape": tuple(param.shape),
                    "numel": param.numel(),
                }
            )

    # Print summary
    epoch_str = f" (Epoch {epoch})" if epoch is not None else ""
    print(f"\n{'=' * 80}")
    print(f"PARAMETER GRADIENT STATUS{epoch_str}")
    print(f"{'=' * 80}")
    print(f"Total parameters: {total_params}")
    print(f"Trainable (requires_grad=True): {trainable_params}")
    print(f"Frozen (requires_grad=False): {frozen_params}")

    if print_all and param_info:
        print(
            f"\n{'Parameter Name':<60} {'Requires Grad':<15} {'Shape':<30} {'Num Elements':<15}"
        )
        print("-" * 120)
        for info in param_info:
            print(
                f"{info['name']:<60} {str(info['requires_grad']):<15} {str(info['shape']):<30} {info['numel']:<15}"
            )

    # Check for potential issues: parameters that should be frozen but aren't
    if frozen_params == 0:
        print(
            "\n⚠️  WARNING: No frozen parameters found. All parameters have requires_grad=True."
        )
    elif trainable_params == 0:
        print(
            "\n⚠️  WARNING: No trainable parameters found. All parameters have requires_grad=False."
        )

    print(f"{'=' * 80}\n")
