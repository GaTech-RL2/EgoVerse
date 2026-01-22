from sqlalchemy import (
    MetaData,
    Table,
    Column,
    Integer,
    String,
    Boolean,
    Float,
    text,
)
from egomimic.utils.aws.aws_sql import (
    TableRow,
    add_episode,
    update_episode,
    create_default_engine,
    episode_hash_to_table_row,
    delete_episodes,
    episode_table_to_df,
    delete_all_episodes,
)
import pandas as pd

engine = create_default_engine()

df = episode_table_to_df(engine)

# Filter for task "fold clothes" and robot_name "aria_bimanual"
df_fold = df[
    (df['task'].isin(['fold_clothes', 'fold clothes'])) & 
    (df['robot_name'] == 'aria_bimanual')
].copy()

# Convert scene to string for consistent comparison
df_fold['scene'] = df_fold['scene'].astype(str)

# Expected denominators (C) from image data
# Format: (operator_name, scene): denominator
# Updated for scenes 1-16 based on new image data
expected_denominators = {
    # Operator 1 (Aniketh)
    ('Aniketh', '1'): 32, ('Aniketh', '2'): 1, ('Aniketh', '3'): 1, ('Aniketh', '4'): 1,
    ('Aniketh', '5'): 1, ('Aniketh', '6'): 1, ('Aniketh', '7'): 1, ('Aniketh', '8'): 1,
    ('Aniketh', '9'): 0, ('Aniketh', '10'): 0, ('Aniketh', '11'): 0, ('Aniketh', '12'): 4,
    ('Aniketh', '13'): 0, ('Aniketh', '14'): 4, ('Aniketh', '15'): 4, ('Aniketh', '16'): 0,
    # Operator 2 (Jenny)
    ('Jenny', '1'): 16, ('Jenny', '2'): 1, ('Jenny', '3'): 1, ('Jenny', '4'): 1,
    ('Jenny', '5'): 1, ('Jenny', '6'): 1, ('Jenny', '7'): 1, ('Jenny', '8'): 1,
    ('Jenny', '9'): 4, ('Jenny', '10'): 4, ('Jenny', '11'): 0, ('Jenny', '12'): 4,
    ('Jenny', '13'): 0, ('Jenny', '14'): 4, ('Jenny', '15'): 4, ('Jenny', '16'): 0,
    # Operator 3 (Baoyu)
    ('Baoyu', '1'): 8, ('Baoyu', '2'): 1, ('Baoyu', '3'): 1, ('Baoyu', '4'): 1,
    ('Baoyu', '5'): 1, ('Baoyu', '6'): 1, ('Baoyu', '7'): 1, ('Baoyu', '8'): 1,
    ('Baoyu', '9'): 4, ('Baoyu', '10'): 4, ('Baoyu', '11'): 4, ('Baoyu', '12'): 4,
    ('Baoyu', '13'): 4, ('Baoyu', '14'): 0, ('Baoyu', '15'): 0, ('Baoyu', '16'): 0,
    # Operator 4 (Lawrence)
    ('Lawrence', '1'): 8, ('Lawrence', '2'): 1, ('Lawrence', '3'): 1, ('Lawrence', '4'): 1,
    ('Lawrence', '5'): 1, ('Lawrence', '6'): 1, ('Lawrence', '7'): 1, ('Lawrence', '8'): 1,
    ('Lawrence', '9'): 4, ('Lawrence', '10'): 0, ('Lawrence', '11'): 0, ('Lawrence', '12'): 4,
    ('Lawrence', '13'): 4, ('Lawrence', '14'): 4, ('Lawrence', '15'): 4, ('Lawrence', '16'): 0,
    # Operator 5 (Ryan)
    ('Ryan', '1'): 4, ('Ryan', '2'): 4, ('Ryan', '3'): 4, ('Ryan', '4'): 4,
    ('Ryan', '5'): 4, ('Ryan', '6'): 4, ('Ryan', '7'): 4, ('Ryan', '8'): 4,
    ('Ryan', '9'): 4, ('Ryan', '10'): 4, ('Ryan', '11'): 4, ('Ryan', '12'): 0,
    ('Ryan', '13'): 0, ('Ryan', '14'): 0, ('Ryan', '15'): 4, ('Ryan', '16'): 4,
    # Operator 6 (Pranav)
    ('Pranav', '1'): 3, ('Pranav', '2'): 3, ('Pranav', '3'): 3, ('Pranav', '4'): 3,
    ('Pranav', '5'): 3, ('Pranav', '6'): 3, ('Pranav', '7'): 3, ('Pranav', '8'): 3,
    ('Pranav', '9'): 0, ('Pranav', '10'): 4, ('Pranav', '11'): 0, ('Pranav', '12'): 0,
    ('Pranav', '13'): 4, ('Pranav', '14'): 0, ('Pranav', '15'): 0, ('Pranav', '16'): 4,
    # Operator 7 (Nadun - previously Elmo)
    ('Nadun', '1'): 4, ('Nadun', '2'): 4, ('Nadun', '3'): 4, ('Nadun', '4'): 4,
    ('Nadun', '5'): 4, ('Nadun', '6'): 4, ('Nadun', '7'): 4, ('Nadun', '8'): 4,
    ('Nadun', '9'): 0, ('Nadun', '10'): 0, ('Nadun', '11'): 0, ('Nadun', '12'): 0,
    ('Nadun', '13'): 0, ('Nadun', '14'): 0, ('Nadun', '15'): 0, ('Nadun', '16'): 0,
    # Operator 8 (Yangcen)
    ('Yangcen', '1'): 2, ('Yangcen', '2'): 2, ('Yangcen', '3'): 2, ('Yangcen', '4'): 2,
    ('Yangcen', '5'): 2, ('Yangcen', '6'): 2, ('Yangcen', '7'): 2, ('Yangcen', '8'): 2,
    ('Yangcen', '9'): 0, ('Yangcen', '10'): 0, ('Yangcen', '11'): 2, ('Yangcen', '12'): 0,
    ('Yangcen', '13'): 0, ('Yangcen', '14'): 2, ('Yangcen', '15'): 0, ('Yangcen', '16'): 2,
    # Operator 9 (Zhenyang)
    ('Zhenyang', '1'): 2, ('Zhenyang', '2'): 2, ('Zhenyang', '3'): 2, ('Zhenyang', '4'): 2,
    ('Zhenyang', '5'): 2, ('Zhenyang', '6'): 2, ('Zhenyang', '7'): 2, ('Zhenyang', '8'): 2,
    ('Zhenyang', '9'): 0, ('Zhenyang', '10'): 0, ('Zhenyang', '11'): 0, ('Zhenyang', '12'): 0,
    ('Zhenyang', '13'): 0, ('Zhenyang', '14'): 0, ('Zhenyang', '15'): 0, ('Zhenyang', '16'): 0,
    # Operator 10 (Woolchul)
    ('Woolchul', '1'): 2, ('Woolchul', '2'): 2, ('Woolchul', '3'): 2, ('Woolchul', '4'): 2,
    ('Woolchul', '5'): 2, ('Woolchul', '6'): 2, ('Woolchul', '7'): 2, ('Woolchul', '8'): 2,
    ('Woolchul', '9'): 0, ('Woolchul', '10'): 0, ('Woolchul', '11'): 0, ('Woolchul', '12'): 0,
    ('Woolchul', '13'): 0, ('Woolchul', '14'): 0, ('Woolchul', '15'): 0, ('Woolchul', '16'): 0,
    # Operator 11 (Shuo)
    ('Shuo', '1'): 2, ('Shuo', '2'): 2, ('Shuo', '3'): 2, ('Shuo', '4'): 2,
    ('Shuo', '5'): 2, ('Shuo', '6'): 2, ('Shuo', '7'): 2, ('Shuo', '8'): 2,
    ('Shuo', '9'): 0, ('Shuo', '10'): 0, ('Shuo', '11'): 0, ('Shuo', '12'): 0,
    ('Shuo', '13'): 0, ('Shuo', '14'): 0, ('Shuo', '15'): 0, ('Shuo', '16'): 0,
    # Operator 12 (Liqian)
    ('Liqian', '1'): 2, ('Liqian', '2'): 2, ('Liqian', '3'): 2, ('Liqian', '4'): 2,
    ('Liqian', '5'): 2, ('Liqian', '6'): 2, ('Liqian', '7'): 2, ('Liqian', '8'): 2,
    ('Liqian', '9'): 0, ('Liqian', '10'): 0, ('Liqian', '11'): 0, ('Liqian', '12'): 0,
    ('Liqian', '13'): 0, ('Liqian', '14'): 0, ('Liqian', '15'): 0, ('Liqian', '16'): 0,
    # Operator 13 (Xinchen)
    ('Xinchen', '1'): 2, ('Xinchen', '2'): 2, ('Xinchen', '3'): 2, ('Xinchen', '4'): 2,
    ('Xinchen', '5'): 2, ('Xinchen', '6'): 2, ('Xinchen', '7'): 2, ('Xinchen', '8'): 2,
    ('Xinchen', '9'): 0, ('Xinchen', '10'): 0, ('Xinchen', '11'): 0, ('Xinchen', '12'): 0,
    ('Xinchen', '13'): 0, ('Xinchen', '14'): 0, ('Xinchen', '15'): 0, ('Xinchen', '16'): 0,
    # Operator 14 (Rohan)
    ('Rohan', '1'): 2, ('Rohan', '2'): 2, ('Rohan', '3'): 2, ('Rohan', '4'): 2,
    ('Rohan', '5'): 2, ('Rohan', '6'): 2, ('Rohan', '7'): 2, ('Rohan', '8'): 2,
    ('Rohan', '9'): 0, ('Rohan', '10'): 0, ('Rohan', '11'): 0, ('Rohan', '12'): 0,
    ('Rohan', '13'): 0, ('Rohan', '14'): 0, ('Rohan', '15'): 0, ('Rohan', '16'): 0,
    # Operator 15 (David)
    ('David', '1'): 2, ('David', '2'): 2, ('David', '3'): 2, ('David', '4'): 2,
    ('David', '5'): 2, ('David', '6'): 2, ('David', '7'): 2, ('David', '8'): 2,
    ('David', '9'): 0, ('David', '10'): 0, ('David', '11'): 0, ('David', '12'): 0,
    ('David', '13'): 0, ('David', '14'): 0, ('David', '15'): 0, ('David', '16'): 0,
    # Operator 16 (Vaibhav)
    ('Vaibhav', '1'): 2, ('Vaibhav', '2'): 2, ('Vaibhav', '3'): 2, ('Vaibhav', '4'): 2,
    ('Vaibhav', '5'): 2, ('Vaibhav', '6'): 2, ('Vaibhav', '7'): 2, ('Vaibhav', '8'): 2,
    ('Vaibhav', '9'): 0, ('Vaibhav', '10'): 0, ('Vaibhav', '11'): 0, ('Vaibhav', '12'): 0,
    ('Vaibhav', '13'): 0, ('Vaibhav', '14'): 0, ('Vaibhav', '15'): 0, ('Vaibhav', '16'): 0,
    # Operator 17 (Mengying)
    ('Mengying', '1'): 2, ('Mengying', '2'): 2, ('Mengying', '3'): 2, ('Mengying', '4'): 2,
    ('Mengying', '5'): 2, ('Mengying', '6'): 2, ('Mengying', '7'): 2, ('Mengying', '8'): 2,
    ('Mengying', '9'): 0, ('Mengying', '10'): 0, ('Mengying', '11'): 0, ('Mengying', '12'): 0,
    ('Mengying', '13'): 0, ('Mengying', '14'): 0, ('Mengying', '15'): 0, ('Mengying', '16'): 0,
    # Operator 18 (Elmo) - same expectations as Operator 7 (Nadun)
    ('Elmo', '1'): 0, ('Elmo', '2'): 0, ('Elmo', '3'): 0, ('Elmo', '4'): 0,
    ('Elmo', '5'): 0, ('Elmo', '6'): 0, ('Elmo', '7'): 0, ('Elmo', '8'): 0,
    ('Elmo', '9'): 0, ('Elmo', '10'): 0, ('Elmo', '11'): 4, ('Elmo', '12'): 0,
    ('Elmo', '13'): 4, ('Elmo', '14'): 0, ('Elmo', '15'): 0, ('Elmo', '16'): 4,
}

# Operator mapping to Operator N format
operator_mapping = {
    'Aniketh': 'Operator 1 (Aniketh)',
    'Jenny': 'Operator 2 (Jenny)',
    'Baoyu': 'Operator 3 (Baoyu)',
    'Lawrence': 'Operator 4 (Lawrence)',
    'Ryan': 'Operator 5 (Ryan)',
    'Pranav': 'Operator 6 (Pranav)',
    'Nadun': 'Operator 7 (Nadun)',
    'Yangcen': 'Operator 8 (Yangcen)',
    'Zhenyang': 'Operator 9 (Zhenyang)',
    'Woolchul': 'Operator 10 (Woolchul)',
    'Shuo': 'Operator 11 (Shuo)',
    'Liqian': 'Operator 12 (Liqian)',
    'Xinchen': 'Operator 13 (Xinchen)',
    'Rohan': 'Operator 14 (Rohan)',
    'David': 'Operator 15 (David)',
    'Vaibhav': 'Operator 16 (Vaibhav)',
    'Mengying': 'Operator 17 (Mengying)',
    'Elmo': 'Operator 18 (Elmo)',
}

# Check if processed_path is not empty (not null and not empty string)
df_fold['has_processed_path'] = (
    df_fold['processed_path'].notna() & 
    (df_fold['processed_path'].astype(str).str.strip() != '')
)

# Calculate metrics for each operator-scene combination
results = []
for operator in operator_mapping.keys():
    for scene in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']:
        # Filter for this operator-scene combination
        op_scene_df = df_fold[
            (df_fold['operator'] == operator) & 
            (df_fold['scene'] == scene)
        ]
        
        # A: Count where processed_path is not empty
        A = op_scene_df['has_processed_path'].sum()
        
        # Collect episode hashes where processed_path is not empty
        processed_episodes = op_scene_df[op_scene_df['has_processed_path']]
        episode_hashes = processed_episodes['episode_hash'].tolist() if len(processed_episodes) > 0 else []
        
        # B: Total count for operator-scene
        B = len(op_scene_df)
        
        # C: Expected denominator from image data
        C = expected_denominators.get((operator, scene), 0)
        
        results.append({
            'operator': operator_mapping[operator],
            'scene': scene,
            'A': A,
            'B': B,
            'C': C,
            'episode_hashes': episode_hashes,
        })

# Create DataFrame from results
results_df = pd.DataFrame(results)

# Create formatted output DataFrame directly from results
scenes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
output_data = {}
hash_output_data = {}  # Separate data structure for episode hashes

for operator_display in operator_mapping.values():
    output_data[operator_display] = {}
    hash_output_data[operator_display] = {}
    for scene in scenes:
        # Find the result for this operator-scene combination
        result = results_df[
            (results_df['operator'] == operator_display) & 
            (results_df['scene'] == scene)
        ]
        
        if len(result) > 0:
            A_val = int(result['A'].iloc[0])
            B_val = int(result['B'].iloc[0])
            C_val = int(result['C'].iloc[0])
            episode_hashes = result['episode_hashes'].iloc[0]
        else:
            # Get operator name from display name
            operator_name = [k for k, v in operator_mapping.items() if v == operator_display][0]
            A_val = 0
            B_val = 0
            C_val = expected_denominators.get((operator_name, scene), 0)
            episode_hashes = []
        
        # Format cell value without episode hashes for main CSV
        output_data[operator_display][f'Scenario {scene}'] = f"{A_val} / {B_val} / {C_val}"
        
        # Store hashes separately for the hash CSV
        if len(episode_hashes) > 0:
            hash_str = ', '.join(str(h) for h in episode_hashes)
            hash_output_data[operator_display][f'Scenario {scene}'] = hash_str
        else:
            # Empty string for no hashes
            hash_output_data[operator_display][f'Scenario {scene}'] = ""

# Create final output DataFrame
output_df = pd.DataFrame(output_data).T
output_df = output_df.reindex(columns=[f'Scenario {s}' for s in scenes])
# Ensure operators are in the correct order
operator_order = [operator_mapping[op] for op in operator_mapping.keys()]
output_df = output_df.reindex(operator_order)

# Calculate totals for each scenario (sum of A, B, C)
total_row = {}
for scene in scenes:
    total_A = 0
    total_B = 0
    total_C = 0
    
    # Sum A, B, C values from all operators for this scene
    for operator_display in operator_order:
        cell_value = output_df.loc[operator_display, f'Scenario {scene}']
        # Parse the "A / B / C" format
        parts = cell_value.split(' / ')
        if len(parts) == 3:
            try:
                total_A += int(parts[0])
                total_B += int(parts[1])
                total_C += int(parts[2])
            except ValueError:
                pass
    
    total_row[f'Scenario {scene}'] = f"{total_A} / {total_B} / {total_C}"

# Add total row to the DataFrame
total_df = pd.DataFrame([total_row], index=['Total'])
output_df = pd.concat([output_df, total_df])

# Create episode hash DataFrame with same structure
hash_output_df = pd.DataFrame(hash_output_data).T
hash_output_df = hash_output_df.reindex(columns=[f'Scenario {s}' for s in scenes])
hash_output_df = hash_output_df.reindex(operator_order)

# Add empty row for totals in hash CSV (or could collect all hashes)
hash_total_row = {f'Scenario {s}': '' for s in scenes}
hash_total_df = pd.DataFrame([hash_total_row], index=['Total'])
hash_output_df = pd.concat([hash_output_df, hash_total_df])

# Save statistics CSV with proper formatting
# Add index name for the operator column
output_df.index.name = 'Time (minutes)'
output_df.to_csv('results/diversity_fold_clothes.csv')
print("CSV file saved as 'results/diversity_fold_clothes.csv'")

# Save episode hashes CSV with same structure
hash_output_df.index.name = 'Time (minutes)'
hash_output_df.to_csv('results/diversity_fold_clothes_hashes.csv')
print("Episode hashes CSV file saved as 'results/diversity_fold_clothes_hashes.csv'")

print(f"\nTotal operators: {len(output_df)}")
print(f"Total scenarios: {len(output_df.columns)}")
print("\nFirst few rows of statistics:")
print(output_df.head())
print("\nFirst few rows of episode hashes:")
print(hash_output_df.head())
print("\nLast few rows:")
print(output_df.tail())