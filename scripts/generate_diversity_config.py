import pandas as pd
import csv

# Read the CSV file
csv_file = 'results/diversity_fold_clothes_hashes.csv'
df = pd.read_csv(csv_file, index_col=0)

# Remove the "Total" row if it exists
if 'Total' in df.index:
    df = df.drop('Total')

# Generate YAML content
yaml_content = []
yaml_content.append("_target_: egomimic.pl_utils.pl_data_utils.MultiDataModuleWrapper")
yaml_content.append("")
yaml_content.append("train_datasets:")
yaml_content.append("  dataset1:")
yaml_content.append("    _target_: egomimic.rldb.utils.MultiRLDBDataset")
yaml_content.append("    datasets:")
yaml_content.append("")

# Process train datasets (all episodes)
episode_counter = 1
for operator_idx, operator_row in enumerate(df.iterrows(), start=1):
    operator_name = operator_row[0]
    row_data = operator_row[1]
    
    # Add newline before new operator (except first)
    if operator_idx > 1:
        yaml_content.append("")
    
    # Add comment with operator name
    operator_comment = operator_name.replace('Operator ', '').replace('(', '# ').replace(')', '')
    yaml_content.append(f"      {operator_comment}")
    
    first_scene_in_operator = True
    for scene_num in range(1, 17):
        scenario_col = f'Scenario {scene_num}'
        cell_value = str(row_data[scenario_col])
        
        # Parse episode hashes (comma-separated)
        if pd.notna(cell_value) and cell_value.strip() and cell_value.strip().lower() != 'nan':
            # Add newline before new scene (except first scene of operator)
            if not first_scene_in_operator:
                yaml_content.append("")
            first_scene_in_operator = False
            
            hashes = [h.strip() for h in cell_value.split(',') if h.strip() and h.strip().lower() != 'nan']
            for ep_idx, hash_val in enumerate(hashes, start=1):
                # Skip if hash_val is NaN or empty
                if pd.isna(hash_val) or not hash_val or hash_val.lower() == 'nan':
                    continue
                
                dataset_name = f"op{operator_idx}_scene{scene_num}_ep{ep_idx}"
                yaml_content.append(f"      {dataset_name}:")
                yaml_content.append("        _target_: egomimic.rldb.utils.S3RLDBDataset")
                yaml_content.append('        bucket_name: "rldb"')
                yaml_content.append('        mode: total')
                yaml_content.append('        embodiment: "aria_bimanual"')
                yaml_content.append("        local_files_only: True")
                yaml_content.append('        temp_root: "/coc/cedarp-dxu345-0/datasets/egoverse/offline_eval_diversity"')
                yaml_content.append(f"        filters: {{episode_hash: '{hash_val}'}}")

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
output_file = 'egomimic/hydra_configs/data/diversity_fold_clothes_all.yaml'
with open(output_file, 'w') as f:
    f.write('\n'.join(yaml_content))

print(f"Generated config file: {output_file}")

