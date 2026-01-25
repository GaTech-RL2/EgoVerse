#!/usr/bin/env python3
"""
Plot Motion Diversity Multi-Scene Experiment Results

This script extracts final validation metrics from three motion diversity experiment
log directories and generates line plots comparing metrics across different operator counts.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import glob


def find_wandb_summary(log_dir: Path) -> Optional[Path]:
    """
    Find the wandb summary JSON file in a log directory.
    
    Args:
        log_dir: Path to the log directory
        
    Returns:
        Path to wandb-summary.json if found, None otherwise
    """
    # Look for wandb-summary.json in the wandb run directory
    pattern = str(log_dir / "wandb" / "run-*" / "files" / "wandb-summary.json")
    matches = glob.glob(pattern)
    
    if matches:
        return Path(matches[0])
    return None


def extract_validation_metrics(summary_path: Path) -> Dict[str, float]:
    """
    Extract validation metrics from a wandb summary JSON file.
    
    Args:
        summary_path: Path to wandb-summary.json
        
    Returns:
        Dictionary mapping metric names to values (only metrics starting with 'Valid/')
    """
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    # Extract only validation metrics
    validation_metrics = {
        key: value for key, value in data.items()
        if key.startswith('Valid/') and isinstance(value, (int, float))
    }
    
    return validation_metrics


def plot_metric_in_subplot(
    ax,
    metric_name: str,
    operator_counts: List[int],
    metric_values: List[float],
    color=None,
):
    """
    Plot a single metric in a subplot.
    
    Args:
        ax: Matplotlib axes object
        metric_name: Name of the metric (e.g., 'Valid/aria_bimanual_actions_cartesian_paired_mse_avg')
        operator_counts: List of operator counts [4, 8, 12]
        metric_values: List of metric values corresponding to operator counts
        color: Color for the line plot
    """
    # Create a clean metric name for the title
    clean_name = metric_name.replace('Valid/aria_bimanual_actions_cartesian_', '').replace('_', ' ')
    
    # Plot line with markers
    ax.plot(operator_counts, metric_values, marker='o', linewidth=2, markersize=8, color=color)
    
    # Formatting
    ax.set_xlabel('Number of Operators', fontsize=10)
    ax.set_ylabel('Metric Value', fontsize=10)
    ax.set_title(clean_name, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(operator_counts)
    
    # Add value labels on points
    for x, y in zip(operator_counts, metric_values):
        ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=8)


def plot_all_metrics_unified(
    all_metrics_data: Dict[str, Dict[int, float]],
    operator_counts: List[int],
    output_dir: Path,
    figsize=(15, 10)
):
    """
    Create a unified figure with all metrics in subplots.
    
    Args:
        all_metrics_data: Dictionary mapping metric names to {operator_count: value}
        operator_counts: List of operator counts [4, 8, 12]
        output_dir: Directory to save the plot
        figsize: Figure size tuple
    """
    # Filter metrics that have at least 2 data points
    valid_metrics = {}
    for metric_name, metric_dict in all_metrics_data.items():
        available_operators = [op for op in operator_counts if op in metric_dict]
        if len(available_operators) >= 2:
            valid_metrics[metric_name] = {
                op: metric_dict[op] for op in available_operators
            }
    
    if not valid_metrics:
        print("No valid metrics to plot")
        return
    
    # Define the desired order for metrics
    metric_order = [
        'Valid/aria_bimanual_actions_cartesian_reverse_kl_M8',
        'Valid/aria_bimanual_actions_cartesian_paired_mse_avg',
        'Valid/aria_bimanual_actions_cartesian_frechet_gauss_min',
        'Valid/aria_bimanual_actions_cartesian_frechet_gauss_max',
        'Valid/aria_bimanual_actions_cartesian_frechet_gauss_avg',
        'Valid/aria_bimanual_actions_cartesian_final_mse_avg',
    ]
    
    # Sort metrics according to the desired order
    sorted_metrics = []
    for metric_name in metric_order:
        if metric_name in valid_metrics:
            sorted_metrics.append((metric_name, valid_metrics[metric_name]))
    
    # Add any remaining metrics that weren't in the order list
    for metric_name, metric_dict in valid_metrics.items():
        if metric_name not in metric_order:
            sorted_metrics.append((metric_name, metric_dict))
    
    # Determine grid size (2 rows, 3 columns for 6 metrics, adjust as needed)
    n_metrics = len(sorted_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
    
    # Create color palette for different metrics
    colors = sns.color_palette("husl", n_metrics)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle different cases for axes shape
    if n_metrics == 1:
        # Single subplot case - axes is a single Axes object
        axes_flat = [axes]
    else:
        # Multiple subplots - flatten to 1D array
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Plot each metric in its subplot
    for idx, (metric_name, metric_dict) in enumerate(sorted_metrics):
        if idx >= len(axes_flat):
            break
        
        ax = axes_flat[idx]
        
        # Get operator counts and values for this metric
        available_operators = sorted([op for op in operator_counts if op in metric_dict])
        metric_values = [metric_dict[op] for op in available_operators]
        
        # Plot in subplot with distinct color
        plot_metric_in_subplot(ax, metric_name, available_operators, metric_values, color=colors[idx])
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save unified plot
    output_path = output_dir / "motion_diversity_all_metrics_unified.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved unified figure: {output_path}")
    
    plt.close()


def main():
    """Main function to extract metrics and generate plots."""
    # Define log directories and their corresponding operator counts
    base_dir = Path("/coc/flash7/bli678/Shared/EgoVerse")
    log_configs = [
        {
            "log_dir": base_dir / "logs/fold-clothes/operator-4-time-15_2026-01-21_10-15-31",
            "operators": 4
        },
        {
            "log_dir": base_dir / "logs/fold-clothes/operator-8-time-7_5_2026-01-21_01-02-39",
            "operators": 8
        },
        {
            "log_dir": base_dir / "logs/fold-clothes/operator-12-time-3_75_2026-01-21_12-13-57",
            "operators": 12
        }
    ]
    
    # Extract metrics from each log directory
    all_metrics = {}  # metric_name -> {operator_count: value}
    
    for config in log_configs:
        log_dir = config["log_dir"]
        operators = config["operators"]
        
        print(f"Processing {operators} operators: {log_dir}")
        
        summary_path = find_wandb_summary(log_dir)
        if summary_path is None:
            print(f"Warning: Could not find wandb summary in {log_dir}")
            continue
        
        validation_metrics = extract_validation_metrics(summary_path)
        print(f"  Found {len(validation_metrics)} validation metrics")
        
        # Store metrics organized by metric name
        for metric_name, value in validation_metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = {}
            all_metrics[metric_name][operators] = value
    
    # Create output directory
    base_dir = Path("/coc/flash7/bli678/Shared/EgoVerse")
    output_dir = base_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unified plot with all metrics
    operator_counts = [4, 8, 12]
    
    print(f"\nGenerating unified figure with all metrics...")
    plot_all_metrics_unified(all_metrics, operator_counts, output_dir)
    
    print(f"\nUnified figure saved to: {output_dir}")


if __name__ == "__main__":
    main()

