#!/usr/bin/env python3
"""
Extract wandb validation metrics from scene diversity experiment directories
and organize them into tables with scenes as rows and time as columns.
"""

import json
import re
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd


def parse_directory_name(dirname: str) -> Optional[Tuple[int, float]]:
    """
    Parse directory name to extract scenes and time.
    
    Expected format: scenes-{number}-time-{minutes}_{timestamp}
    Note: Time may use underscores instead of decimal points (e.g., "7_5" = 7.5, "3_75" = 3.75)
    
    Returns:
        Tuple of (scenes, time) or None if parsing fails
    """
    # Pattern that handles both decimal points and underscores in time
    pattern = r'scenes-(\d+)-time-([\d._]+)_'
    match = re.match(pattern, dirname)
    if match:
        scenes = int(match.group(1))
        time_str = match.group(2)
        # Convert underscore to decimal point for time (e.g., "7_5" -> 7.5, "3_75" -> 3.75)
        if '_' in time_str:
            time = float(time_str.replace('_', '.'))
        else:
            time = float(time_str)
        return (scenes, time)
    return None


def find_wandb_summary(wandb_dir: Path) -> Optional[Path]:
    """
    Find the wandb-summary.json file in the wandb directory.
    
    Checks:
    1. latest-run symlink
    2. run-* directories
    
    Returns:
        Path to wandb-summary.json or None if not found
    """
    # Try latest-run symlink first
    latest_run = wandb_dir / "latest-run"
    if latest_run.exists() and latest_run.is_symlink():
        summary_path = latest_run.resolve() / "files" / "wandb-summary.json"
        if summary_path.exists():
            return summary_path
    
    # Try to find run-* directories
    run_dirs = list(wandb_dir.glob("run-*"))
    if run_dirs:
        # Use the first run directory found
        summary_path = run_dirs[0] / "files" / "wandb-summary.json"
        if summary_path.exists():
            return summary_path
    
    return None


def extract_validation_metrics(summary_path: Path) -> Dict[str, float]:
    """
    Extract all validation metrics from wandb-summary.json.
    
    Returns:
        Dictionary mapping metric names (starting with "Valid/") to their values
    """
    try:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Extract all metrics that start with "Valid/"
        valid_metrics = {
            key: value 
            for key, value in summary.items() 
            if key.startswith("Valid/")
        }
        
        return valid_metrics
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {summary_path}: {e}")
        return {}


def sanitize_filename(metric_name: str) -> str:
    """
    Sanitize metric name for use as filename.
    
    Replaces invalid filesystem characters with underscores.
    """
    # Remove "Valid/" prefix
    name = metric_name.replace("Valid/", "")
    # Replace invalid characters
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Replace slashes with underscores
    name = name.replace('/', '_')
    return name


def main():
    """Main function to extract metrics and create tables."""
    
    # Source directory with experiment results
    source_dir = Path("/coc/cedarp-dxu345-0/bli678/EgoVerse/logs/fold_clothes/scene_diversity")
    
    # Output directory for CSV files
    output_dir = Path("/coc/flash7/bli678/Shared/EgoVerse/results/scene_diversity_fold_clothes")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all metrics by configuration
    # Structure: {(scenes, time): {metric_name: value}}
    metrics_by_config: Dict[Tuple[int, float], Dict[str, float]] = {}
    
    # Scan all scene-diversity directories
    print(f"Scanning directory: {source_dir}")
    for item in source_dir.iterdir():
        if not item.is_dir():
            continue
        
        # Parse directory name
        parsed = parse_directory_name(item.name)
        if parsed is None:
            print(f"Warning: Could not parse directory name: {item.name}")
            continue
        
        scenes, time = parsed
        print(f"Processing: {item.name} -> {scenes} scenes, {time} minutes")
        
        # Find wandb summary file
        wandb_dir = item / "wandb"
        if not wandb_dir.exists():
            print(f"Warning: wandb directory not found in {item.name}")
            continue
        
        summary_path = find_wandb_summary(wandb_dir)
        if summary_path is None:
            print(f"Warning: wandb-summary.json not found in {item.name}")
            continue
        
        # Extract validation metrics
        metrics = extract_validation_metrics(summary_path)
        if not metrics:
            print(f"Warning: No validation metrics found in {item.name}")
            continue
        
        # Store metrics by configuration
        config = (scenes, time)
        metrics_by_config[config] = metrics
        print(f"  Found {len(metrics)} validation metrics")
    
    if not metrics_by_config:
        print("Error: No metrics found in any directory!")
        return
    
    # Collect all unique metric names
    all_metrics = set()
    for metrics in metrics_by_config.values():
        all_metrics.update(metrics.keys())
    
    print(f"\nFound {len(all_metrics)} unique validation metrics")
    print(f"Found {len(metrics_by_config)} experiment configurations")
    
    # Define expected scenes and time values
    expected_scenes = [1, 2, 4, 8, 16]
    expected_times = [3.75, 7.5, 15, 30, 60]
    
    # Report which configurations were found
    print("\nExperiment configurations found:")
    for config in sorted(metrics_by_config.keys()):
        scenes, time = config
        num_metrics = len(metrics_by_config[config])
        print(f"  {scenes} scenes, {time} minutes: {num_metrics} metrics")
    
    # Check for missing configurations
    missing_configs = []
    for scenes in expected_scenes:
        for time in expected_times:
            config = (scenes, time)
            if config not in metrics_by_config:
                missing_configs.append(f"{scenes} scenes, {time} minutes")
    
    if missing_configs:
        print(f"\nWarning: Missing configurations (will show as empty/NaN in tables):")
        for config in missing_configs[:10]:  # Show first 10
            print(f"  - {config}")
        if len(missing_configs) > 10:
            print(f"  ... and {len(missing_configs) - 10} more")
    
    # Create a table for each metric
    for metric_name in sorted(all_metrics):
        print(f"\nProcessing metric: {metric_name}")
        
        # Create DataFrame with scenes as rows and time as columns
        data = {}
        missing_values = []
        for time in expected_times:
            data[time] = []
            for scenes in expected_scenes:
                config = (scenes, time)
                value = metrics_by_config.get(config, {}).get(metric_name, None)
                data[time].append(value)
                if value is None:
                    missing_values.append(f"({scenes}s, {time}m)")
        
        df = pd.DataFrame(data, index=expected_scenes)
        df.index.name = "Scenes"
        df.columns.name = "Time (minutes)"
        
        if missing_values:
            print(f"  Warning: Missing values for {metric_name} at: {', '.join(missing_values[:5])}")
            if len(missing_values) > 5:
                print(f"    ... and {len(missing_values) - 5} more")
        
        # Sanitize metric name for filename
        filename = sanitize_filename(metric_name)
        output_path = output_dir / f"{filename}.csv"
        
        # Save to CSV
        df.to_csv(output_path)
        print(f"  Saved to: {output_path}")
        print(f"  Table shape: {df.shape}")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  - Processed {len(metrics_by_config)} experiment configurations")
    print(f"  - Found {len(all_metrics)} unique validation metrics")
    print(f"  - Created {len(all_metrics)} CSV files")
    print(f"  - Output directory: {output_dir}")
    
    if missing_configs:
        print(f"\n  Note: {len(missing_configs)} expected configurations were not found.")
        print(f"        These will appear as empty/NaN values in the CSV tables.")
    else:
        print(f"\n  All expected configurations were found - no empty values expected.")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
