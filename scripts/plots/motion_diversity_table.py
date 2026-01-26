#!/usr/bin/env python3
"""
Extract wandb validation metrics from motion diversity experiment directories
and organize them into a table with operators as rows and metrics as columns.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional
import pandas as pd


def parse_directory_name(dirname: str) -> Optional[int]:
    """
    Parse directory name to extract operator number.
    
    Expected format: operator-{number}-time-{minutes}_{timestamp}
    
    Returns:
        Operator number or None if parsing fails
    """
    pattern = r'operator-(\d+)-time-'
    match = re.match(pattern, dirname)
    if match:
        operators = int(match.group(1))
        return operators
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


def format_metric_name(metric_name: str) -> str:
    """
    Format metric name for use as column header.
    
    Converts metric name to readable format.
    """
    # Remove "Valid/" prefix
    name = metric_name.replace("Valid/", "")
    name = name.replace("aria_bimanual_actions_cartesian_", "")
    
    # Handle specific metric names
    if "reverse_kl_M8" in name:
        return "Reverse KL (M=8)"
    elif "paired_mse_avg" in name:
        return "Paired MSE Avg"
    elif "final_mse_avg" in name:
        return "Final MSE Avg"
    elif "frechet_gauss_min" in name:
        return "Frechet Gauss Min"
    elif "frechet_gauss_avg" in name:
        return "Frechet Gauss Avg"
    elif "frechet_gauss_max" in name:
        return "Frechet Gauss Max"
    
    # Fallback: convert snake_case to Title Case
    name = name.replace('_', ' ').title()
    return name


def main():
    """Main function to extract metrics and create table."""
    
    # Source directory with experiment results
    source_dir = Path("/coc/cedarp-dxu345-0/bli678/EgoVerse/logs/eval-fold-clothes-motion-diversity")
    
    # Output directory for CSV file
    output_dir = Path("/coc/flash7/bli678/Shared/EgoVerse/results/motion_diversity_fold_clothes")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all metrics by operator count
    # Structure: {operators: {metric_name: value}}
    metrics_by_operator: Dict[int, Dict[str, float]] = {}
    
    # Define metric order (as they appear in wandb)
    metric_order = [
        "Valid/aria_bimanual_actions_cartesian_reverse_kl_M8",
        "Valid/aria_bimanual_actions_cartesian_paired_mse_avg",
        "Valid/aria_bimanual_actions_cartesian_final_mse_avg",
        "Valid/aria_bimanual_actions_cartesian_frechet_gauss_min",
        "Valid/aria_bimanual_actions_cartesian_frechet_gauss_avg",
        "Valid/aria_bimanual_actions_cartesian_frechet_gauss_max",
    ]
    
    # Scan all operator directories
    print(f"Scanning directory: {source_dir}")
    for item in source_dir.iterdir():
        if not item.is_dir():
            continue
        
        # Parse directory name
        operators = parse_directory_name(item.name)
        if operators is None:
            print(f"Warning: Could not parse directory name: {item.name}")
            continue
        
        print(f"Processing: {item.name} -> {operators} operators")
        
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
        
        # Store metrics by operator count
        metrics_by_operator[operators] = metrics
        print(f"  Found {len(metrics)} validation metrics")
    
    if not metrics_by_operator:
        print("Error: No metrics found in any directory!")
        return
    
    print(f"\nFound {len(metrics_by_operator)} experiment configurations")
    
    # Define expected operators (sorted)
    expected_operators = sorted(metrics_by_operator.keys())
    print(f"Operators found: {expected_operators}")
    
    # Create DataFrame with operators as rows and metrics as columns
    data = {}
    for metric_name in metric_order:
        column_name = format_metric_name(metric_name)
        data[column_name] = []
        for operators in expected_operators:
            value = metrics_by_operator.get(operators, {}).get(metric_name, None)
            data[column_name].append(value)
    
    df = pd.DataFrame(data, index=expected_operators)
    df.index.name = "Operators"
    
    # Save to CSV
    output_path = output_dir / "motion_diversity_metrics.csv"
    df.to_csv(output_path)
    print(f"\nSaved table to: {output_path}")
    print(f"Table shape: {df.shape}")
    print(f"\nTable preview:")
    print(df)
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  - Processed {len(metrics_by_operator)} experiment configurations")
    print(f"  - Found {len(metric_order)} metrics")
    print(f"  - Output file: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
