#!/usr/bin/env python3
"""
Plot Scene Diversity Experiment Results

This script extracts final validation metrics from scene diversity experiment
log directories and generates unified line plots comparing metrics across different
numbers of scenes, with separate lines for different time-per-scene values.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob
import re


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


def parse_scene_diversity_dir(dir_name: str) -> Optional[Tuple[int, float]]:
    """
    Parse scene diversity directory name to extract number of scenes and time per scene.
    
    Args:
        dir_name: Directory name like "scenes-1-time-3_75_2026-01-21_22-03-37"
        
    Returns:
        Tuple of (num_scenes, time_per_scene) if successful, None otherwise
    """
    # Pattern: scenes-{N}-time-{T}_{timestamp}
    # The time value can be like "3_75" (for 3.75) or "60" (for 60)
    # We need to match up to the first underscore followed by a date pattern
    match = re.match(r'scenes-(\d+)-time-([\d_]+?)(?:_\d{4}-\d{2}-\d{2}|$)', dir_name)
    if match:
        num_scenes = int(match.group(1))
        time_str = match.group(2).replace('_', '.')
        try:
            time_per_scene = float(time_str)
            return (num_scenes, time_per_scene)
        except ValueError:
            return None
    return None


def plot_metric_with_multiple_lines(
    ax,
    metric_name: str,
    scene_counts: List[int],
    time_values: List[float],
    metric_data: Dict[int, Dict[float, float]],
):
    """
    Plot a metric with multiple lines (one per time value).
    
    Args:
        ax: Matplotlib axes object
        metric_name: Name of the metric
        scene_counts: List of scene counts [1, 2, 4, 8, 16]
        time_values: List of time per scene values [3.75, 7.5, 15, 30, 60]
        metric_data: Dictionary mapping scene_count -> {time_per_scene: value}
    """
    # Create a clean metric name for the title
    clean_name = metric_name.replace('Valid/aria_bimanual_actions_cartesian_', '').replace('_', ' ')
    
    # Create color palette for different time values
    colors = sns.color_palette("husl", len(time_values))
    
    # Plot a line for each time value
    has_lines = False
    for time_idx, time_val in enumerate(sorted(time_values)):
        # Get data points for this time value
        scene_vals = []
        metric_vals = []
        
        for scene_count in sorted(scene_counts):
            if scene_count in metric_data:
                # Find matching time value (with tolerance for floating point)
                matched_time = None
                for stored_time in metric_data[scene_count].keys():
                    if abs(stored_time - time_val) < 0.01:
                        matched_time = stored_time
                        break
                
                if matched_time is not None:
                    scene_vals.append(scene_count)
                    metric_vals.append(metric_data[scene_count][matched_time])
        
        # Only plot if we have at least 2 data points
        if len(scene_vals) >= 2:
            label = f'{time_val} min'
            ax.plot(scene_vals, metric_vals, marker='o', linewidth=2, markersize=6, 
                   color=colors[time_idx], label=label)
            has_lines = True
    
    # Formatting
    ax.set_xlabel('Number of Scenes', fontsize=10)
    ax.set_ylabel('Metric Value', fontsize=10)
    ax.set_title(clean_name, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(scene_counts)
    
    # Only show legend if there are lines
    if has_lines:
        ax.legend(loc='best', fontsize=8)


def plot_all_metrics_unified(
    all_metrics_data: Dict[str, Dict[int, Dict[float, float]]],
    scene_counts: List[int],
    time_values: List[float],
    output_dir: Path,
    figsize=(15, 10)
):
    """
    Create a unified figure with all metrics in subplots.
    
    Args:
        all_metrics_data: Dictionary mapping metric names to {scene_count: {time_per_scene: value}}
        scene_counts: List of scene counts [1, 2, 4, 8, 16]
        time_values: List of time per scene values [3.75, 7.5, 15, 30, 60]
        output_dir: Directory to save the plot
        figsize: Figure size tuple
    """
    # Filter metrics that have at least some data
    valid_metrics = {}
    for metric_name, metric_dict in all_metrics_data.items():
        # Check if metric has data for at least 2 scene counts
        available_scenes = [s for s in scene_counts if s in metric_dict]
        if len(available_scenes) >= 2:
            valid_metrics[metric_name] = metric_dict
    
    if not valid_metrics:
        print("No valid metrics to plot")
        return
    
    # Collect all unique time values from the data
    all_time_values = set()
    for metric_dict in valid_metrics.values():
        for scene_dict in metric_dict.values():
            all_time_values.update(scene_dict.keys())
    
    # Use the intersection of expected time values and actual time values in data
    # This ensures we only plot lines for time values that actually exist
    actual_time_values = sorted([t for t in time_values if any(abs(t - stored_t) < 0.01 for stored_t in all_time_values)])
    
    if not actual_time_values:
        print("Warning: No matching time values found in data")
        print(f"Expected: {time_values}")
        print(f"Found in data: {sorted(all_time_values)}")
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
        
        # Plot with multiple lines (using actual time values from data)
        plot_metric_with_multiple_lines(ax, metric_name, scene_counts, actual_time_values, metric_dict)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save unified plot
    output_path = output_dir / "scene_diversity_all_metrics_unified.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved unified figure: {output_path}")
    
    plt.close()


def main():
    """Main function to extract metrics and generate plots."""
    base_dir = Path("/coc/flash7/bli678/Shared/EgoVerse")
    logs_dir = base_dir / "logs/fold-clothes"
    
    # Discover all scene diversity log directories
    scene_diversity_dirs = []
    pattern = str(logs_dir / "scenes-*-time-*")
    all_dirs = glob.glob(pattern)
    
    for dir_path in all_dirs:
        dir_name = Path(dir_path).name
        parsed = parse_scene_diversity_dir(dir_name)
        if parsed:
            num_scenes, time_per_scene = parsed
            scene_diversity_dirs.append({
                "log_dir": Path(dir_path),
                "num_scenes": num_scenes,
                "time_per_scene": time_per_scene
            })
    
    print(f"Found {len(scene_diversity_dirs)} scene diversity log directories")
    
    # Extract metrics from each log directory
    # Structure: metric_name -> {num_scenes: {time_per_scene: value}}
    all_metrics = {}
    
    for config in scene_diversity_dirs:
        log_dir = config["log_dir"]
        num_scenes = config["num_scenes"]
        time_per_scene = config["time_per_scene"]
        
        print(f"Processing {num_scenes} scenes, {time_per_scene} min: {log_dir.name}")
        
        summary_path = find_wandb_summary(log_dir)
        if summary_path is None:
            print(f"  Warning: Could not find wandb summary")
            continue
        
        validation_metrics = extract_validation_metrics(summary_path)
        print(f"  Found {len(validation_metrics)} validation metrics")
        
        # Store metrics organized by metric name, then scene count, then time
        for metric_name, value in validation_metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = {}
            if num_scenes not in all_metrics[metric_name]:
                all_metrics[metric_name][num_scenes] = {}
            all_metrics[metric_name][num_scenes][time_per_scene] = value
    
    # Create output directory
    output_dir = base_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define scene counts and time values
    scene_counts = [1, 2, 4, 8, 16]
    time_values = [3.75, 7.5, 15, 30, 60]
    
    print(f"\nGenerating unified figure with all metrics...")
    plot_all_metrics_unified(all_metrics, scene_counts, time_values, output_dir)
    
    print(f"\nUnified figure saved to: {output_dir}")


if __name__ == "__main__":
    main()

