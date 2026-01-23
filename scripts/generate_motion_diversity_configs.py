#!/usr/bin/env python3
"""
Generate Motion Diversity Multi Scene Config Files

This script generates 3 YAML config files based on the diversity_fold_clothes_hashes.csv file,
selecting specific operators and episode counts per scene as specified in the plan.
"""

import pandas as pd
import re
from pathlib import Path


def parse_episode_hashes(cell_value):
    """
    Parse comma-separated episode hashes from CSV cell.
    
    Args:
        cell_value: Cell value from CSV (string or NaN)
        
    Returns:
        List of episode hash strings
    """
    if pd.isna(cell_value) or not str(cell_value).strip() or str(cell_value).strip().lower() == 'nan':
        return []
    
    hashes = [h.strip() for h in str(cell_value).split(',') if h.strip() and h.strip().lower() != 'nan']
    return hashes


def get_operator_number(operator_name):
    """
    Extract operator number from CSV operator name format.
    
    Args:
        operator_name: Operator name like "Operator 5 (Ryan)"
        
    Returns:
        Operator number (int) or None if not found
    """
    match = re.match(r'Operator\s+(\d+)', operator_name)
    if match:
        return int(match.group(1))
    return None


def get_operator_display_name(operator_name):
    """
    Extract display name from CSV operator name format.
    
    Args:
        operator_name: Operator name like "Operator 5 (Ryan)"
        
    Returns:
        Display name like "Ryan" or full name if no parentheses
    """
    match = re.search(r'\((.+)\)', operator_name)
    if match:
        return match.group(1)
    return operator_name.replace('Operator ', '').strip()


# Define config specifications
CONFIG_SPECS = {
    '4_15': {
        'operators': ['Ryan', 'Pranav', 'Nadun', 'Yangcen'],
        'episode_counts': {
            'Ryan': [4, 4, 4, 4, 4, 4, 4, 4],  # All scenes: 4 episodes
            'Pranav': [2, 3, 2, 3, 1, 3, 2, 3],  # Scenes 1-8
            'Nadun': [4, 4, 4, 4, 4, 4, 4, 4],  # All scenes: 4 episodes
            'Yangcen': [2, 2, 2, 2, 2, 2, 2, 2],  # All scenes: 2 episodes
        }
    },
    '8_7_5': {
        'operators': ['Ryan', 'Pranav', 'Nadun', 'Yangcen', 'Xinchen', 'Rohan', 'David', 'Vaibhav'],
        'episode_counts': {
            'Ryan': [2, 2, 2, 2, 2, 2, 2, 2],
            'Pranav': [2, 2, 2, 2, 1, 2, 2, 2],
            'Nadun': [2, 2, 2, 2, 2, 2, 2, 2],
            'Yangcen': [1, 1, 1, 1, 1, 1, 1, 1],
            'Xinchen': [2, 2, 2, 2, 2, 2, 2, 2],
            'Rohan': [2, 2, 2, 2, 2, 2, 2, 2],
            'David': [2, 2, 2, 2, 2, 2, 2, 2],
            'Vaibhav': [2, 2, 2, 2, 2, 2, 2, 2],
        }
    },
    '12_3_75': {
        'operators': ['Ryan', 'Pranav', 'Nadun', 'Yangcen', 'Xinchen', 'Rohan', 'David', 'Vaibhav', 
                     'Aniketh', 'Jenny', 'Baoyu', 'Lawrence'],
        'episode_counts': {
            'Ryan': [1, 1, 1, 1, 1, 1, 1, 1],
            'Pranav': [1, 1, 1, 1, 1, 1, 1, 1],
            'Nadun': [1, 1, 1, 1, 1, 1, 1, 1],
            'Yangcen': [1, 1, 1, 1, 1, 1, 1, 1],
            'Xinchen': [1, 1, 1, 1, 1, 1, 1, 1],
            'Rohan': [1, 1, 1, 1, 1, 1, 1, 1],
            'David': [1, 1, 1, 1, 1, 1, 1, 1],
            'Vaibhav': [1, 1, 1, 1, 1, 1, 1, 1],
            'Aniketh': [1, 1, 1, 1, 1, 1, 1, 1],
            'Jenny': [1, 1, 1, 1, 1, 1, 1, 1],
            'Baoyu': [1, 1, 1, 1, 1, 1, 1, 1],
            'Lawrence': [1, 1, 1, 1, 1, 1, 1, 1],
        }
    }
}

# Map operator names to CSV operator numbers
OPERATOR_TO_NUM = {
    'Aniketh': 1,
    'Jenny': 2,
    'Baoyu': 3,
    'Lawrence': 4,
    'Ryan': 5,
    'Pranav': 6,
    'Nadun': 7,
    'Yangcen': 8,
    'Zhenyang': 9,
    'Woolchul': 10,
    'Shuo': 11,
    'Liqian': 12,
    'Xinchen': 13,
    'Rohan': 14,
    'David': 15,
    'Vaibhav': 16,
    'Mengying': 17,
    'Elmo': 18,
}


def generate_config(config_key, df, output_dir):
    """
    Generate a YAML config file for the specified config.
    
    Args:
        config_key: Key from CONFIG_SPECS (e.g., '4_15')
        df: DataFrame containing episode hashes
        output_dir: Directory to save the config file
    """
    spec = CONFIG_SPECS[config_key]
    operators = spec['operators']
    episode_counts = spec['episode_counts']
    
    # Generate YAML content
    yaml_content = []
    yaml_content.append("_target_: egomimic.pl_utils.pl_data_utils.MultiDataModuleWrapper")
    yaml_content.append("")
    yaml_content.append("train_datasets:")
    yaml_content.append("  dataset1:")
    yaml_content.append("    _target_: egomimic.rldb.utils.MultiRLDBDataset")
    yaml_content.append("    datasets:")
    yaml_content.append("")
    
    # Process each operator
    for op_idx, operator_name in enumerate(operators, start=1):
        operator_num = OPERATOR_TO_NUM.get(operator_name)
        if operator_num is None:
            print(f"Warning: Operator '{operator_name}' not found in mapping. Skipping.")
            continue
        
        # Find operator row in DataFrame
        operator_row_name = None
        for idx in df.index:
            idx_str = str(idx)
            op_num = get_operator_number(idx_str)
            if op_num == operator_num:
                operator_row_name = idx
                break
        
        if operator_row_name is None:
            print(f"Warning: Operator {operator_name} (op{operator_num}) not found in CSV. Skipping.")
            continue
        
        row_data = df.loc[operator_row_name]
        
        # Add newline before new operator (except first)
        if op_idx > 1:
            yaml_content.append("")
        
        # Add comment with operator name
        yaml_content.append(f"      # {operator_name}")
        
        # Process scenes 1-8
        first_scene_in_operator = True
        for scene_num in range(1, 9):  # Scenes 1-8
            scenario_col = f'Scenario {scene_num}'
            cell_value = row_data[scenario_col]
            
            # Get episode count for this operator-scene
            count = episode_counts[operator_name][scene_num - 1]
            
            # Parse episode hashes
            hashes = parse_episode_hashes(cell_value)
            
            if hashes and count > 0:
                # Select first N episodes
                selected_hashes = hashes[:count]
                
                # Add newline before new scene (except first scene of operator)
                if not first_scene_in_operator:
                    yaml_content.append("")
                first_scene_in_operator = False
                
                # Add episodes
                for ep_idx, hash_val in enumerate(selected_hashes, start=1):
                    dataset_name = f"op{op_idx}_scene{scene_num}_ep{ep_idx}"
                    yaml_content.append(f"      {dataset_name}:")
                    yaml_content.append("        _target_: egomimic.rldb.utils.S3RLDBDataset")
                    yaml_content.append('        bucket_name: "rldb"')
                    yaml_content.append('        mode: total')
                    yaml_content.append('        embodiment: "aria_bimanual"')
                    yaml_content.append("        local_files_only: True")
                    yaml_content.append('        temp_root: "/coc/cedarp-dxu345-0/datasets/egoverse/offline_eval_diversity"')
                    yaml_content.append(f"        filters: {{episode_hash: '{hash_val}'}}")
    
    # Add closing sections
    yaml_content.append("")
    yaml_content.append('    embodiment: "aria_bimanual"')
    yaml_content.append("")
    yaml_content.append("train_dataloader_params:")
    yaml_content.append("  dataset1:")
    yaml_content.append("    batch_size: 32")
    yaml_content.append("    num_workers: 10")
    yaml_content.append("")
    yaml_content.append("valid_dataloader_params:")
    yaml_content.append("  dataset1:")
    yaml_content.append("    batch_size: 32")
    yaml_content.append("    num_workers: 10")
    
    # Write to file
    filename = f"motion_diversity_multi_scene_{config_key}.yaml"
    output_path = Path(output_dir) / filename
    with open(output_path, 'w') as f:
        f.write('\n'.join(yaml_content))
    
    print(f"Generated: {output_path}")


def main():
    """Main entry point."""
    # Read CSV file
    csv_file = 'results/diversity_fold_clothes_hashes.csv'
    df = pd.read_csv(csv_file, index_col=0)
    
    # Remove the "Total" row if it exists
    if 'Total' in df.index:
        df = df.drop('Total')
    
    # Output directory
    output_dir = Path('egomimic/hydra_configs/data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all 3 config files
    print("Generating motion diversity multi-scene config files...")
    for config_key in CONFIG_SPECS.keys():
        print(f"\nGenerating config: {config_key}")
        generate_config(config_key, df, output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()

