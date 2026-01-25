#!/usr/bin/env python3
"""Generate scene diversity config files by sampling from scene_diversity_8_60.yaml."""

import yaml
import re

# Read base config
with open('egomimic/hydra_configs/data/scene_diversity/scene_diversity_8_60.yaml', 'r') as f:
    base = yaml.safe_load(f)

base_datasets = base['train_datasets']['dataset1']['datasets']

def extract_datasets(num_scenes, num_operators_per_scene):
    """Extract datasets for given number of scenes and operators per scene"""
    filtered_datasets = {}
    
    for key, value in base_datasets.items():
        match = re.match(r'op(\d+)_scene(\d+)_ep(\d+)', key)
        if match:
            op = int(match.group(1))
            scene = int(match.group(2))
            
            # Filter by scene (first N scenes)
            if scene <= num_scenes:
                # Filter by operator (first N operators)
                if op <= num_operators_per_scene:
                    filtered_datasets[key] = value
    
    return filtered_datasets

def create_config_file(num_scenes, minutes_per_scene, num_operators):
    """Create a config file with specified scenes and operators"""
    datasets = extract_datasets(num_scenes, num_operators)
    
    # Format filename
    time_str = str(minutes_per_scene).replace('.', '_')
    filename = f'egomimic/hydra_configs/data/scene_diversity/scene_diversity_{num_scenes}_{time_str}.yaml'
    
    # Write file with proper formatting matching base file
    with open(filename, 'w') as f:
        f.write('_target_: egomimic.pl_utils.pl_data_utils.MultiDataModuleWrapper\n')
        f.write('\n')
        f.write('valid_datasets:\n')
        f.write('  dataset1: \n')
        f.write('    _target_: egomimic.rldb.utils.MultiRLDBDataset\n')
        f.write('    datasets:\n')
        f.write('      # ood scene\n')
        f.write('      op1_scene1_ep1:\n')
        f.write('        _target_: egomimic.rldb.utils.S3RLDBDataset\n')
        f.write('        bucket_name: "rldb"\n')
        f.write('        mode: total\n')
        f.write('        embodiment: "aria_bimanual"\n')
        f.write('        local_files_only: True\n')
        f.write('        temp_root: "/coc/cedarp-dxu345-0/datasets/egoverse/offline_eval_diversity"\n')
        f.write('        filters: {episode_hash: \'2025-11-11-22-56-48-683000\'}\n')
        f.write('      # op1_scene1_ep2:\n')
        f.write('      #   _target_: egomimic.rldb.utils.S3RLDBDataset\n')
        f.write('      #   bucket_name: "rldb"\n')
        f.write('      #   mode: total\n')
        f.write('      #   embodiment: "aria_bimanual"\n')
        f.write('      #   local_files_only: True\n')
        f.write('      #   temp_root: "/coc/cedarp-dxu345-0/datasets/egoverse/offline_eval_diversity"\n')
        f.write('      #   filters: {episode_hash: \'2025-11-11-22-56-28-321000\'}\n')
        f.write('      # op1_scene1_ep3:\n')
        f.write('      #   _target_: egomimic.rldb.utils.S3RLDBDataset\n')
        f.write('      #   bucket_name: "rldb"\n')
        f.write('      #   mode: total\n')
        f.write('      #   embodiment: "aria_bimanual"\n')
        f.write('      #   local_files_only: True\n')
        f.write('      #   temp_root: "/coc/cedarp-dxu345-0/datasets/egoverse/offline_eval_diversity"\n')
        f.write('      #   filters: {episode_hash: \'2025-11-11-22-56-13-083000\'}\n')
        f.write('      # op1_scene1_ep4:\n')
        f.write('      #   _target_: egomimic.rldb.utils.S3RLDBDataset\n')
        f.write('      #   bucket_name: "rldb"\n')
        f.write('      #   mode: total\n')
        f.write('      #   embodiment: "aria_bimanual"\n')
        f.write('      #   local_files_only: True\n')
        f.write('      #   temp_root: "/coc/cedarp-dxu345-0/datasets/egoverse/offline_eval_diversity"\n')
        f.write('      #   filters: {episode_hash: \'2025-11-11-22-55-56-880000\'}\n')
        f.write('\n')
        f.write('    embodiment: "aria_bimanual"\n')
        f.write('\n')
        f.write('\n')
        f.write('\n')
        f.write('\n')
        f.write('train_datasets:\n')
        f.write('  dataset1:\n')
        f.write('    _target_: egomimic.rldb.utils.MultiRLDBDataset\n')
        f.write('    datasets:\n')
        f.write('\n')
        
        # Sort by scene, then by operator
        sorted_keys = sorted(datasets.keys(), key=lambda k: (
            int(re.match(r'op\d+_scene(\d+)_ep\d+', k).group(1)),
            int(re.match(r'op(\d+)_scene\d+_ep\d+', k).group(1))
        ))
        
        last_scene = 0
        for key in sorted_keys:
            match = re.match(r'op(\d+)_scene(\d+)_ep(\d+)', key)
            scene = int(match.group(2))
            
            if scene != last_scene:
                f.write(f'      # scene {scene}\n')
                last_scene = scene
            
            value = datasets[key]
            f.write(f'      {key}:\n')
            f.write('        _target_: egomimic.rldb.utils.S3RLDBDataset\n')
            f.write(f'        bucket_name: "{value["bucket_name"]}"\n')
            f.write(f'        mode: {value["mode"]}\n')
            f.write(f'        embodiment: "{value["embodiment"]}"\n')
            local_files_only_str = "True" if value["local_files_only"] else "False"
            f.write(f'        local_files_only: {local_files_only_str}\n')
            f.write(f'        temp_root: "{value["temp_root"]}"\n')
            episode_hash = value['filters']['episode_hash']
            f.write(f'        filters: {{episode_hash: \'{episode_hash}\'}}\n')
            f.write('\n')
        
        f.write('    embodiment: "aria_bimanual"\n')
        f.write('\n')
        f.write('train_dataloader_params:\n')
        f.write('  dataset1:\n')
        f.write('    batch_size: 32\n')
        f.write('    num_workers: 10\n')
        f.write('\n')
        f.write('valid_dataloader_params:\n')
        f.write('  dataset1:\n')
        f.write('    batch_size: 32\n')
        f.write('    num_workers: 10\n')
    
    print(f'Created: {filename} ({len(datasets)} entries)')

# Configurations to create (from bottom to top of table)
configs = [
    # 8 scene configs
    (8, 30, 8),   # 8 scenes, 30 min/scene = 8 operators
    (8, 15, 4),   # 8 scenes, 15 min/scene = 4 operators
    (8, 7.5, 2),  # 8 scenes, 7.5 min/scene = 2 operators
    (8, 3.75, 1), # 8 scenes, 3.75 min/scene = 1 operator
    # 4 scene configs
    (4, 60, 16),  # 4 scenes, 60 min/scene = 16 operators
    (4, 30, 8),   # 4 scenes, 30 min/scene = 8 operators
    (4, 15, 4),   # 4 scenes, 15 min/scene = 4 operators
    (4, 7.5, 2),  # 4 scenes, 7.5 min/scene = 2 operators
    (4, 3.75, 1), # 4 scenes, 3.75 min/scene = 1 operator
    # 2 scene configs
    (2, 60, 16),  # 2 scenes, 60 min/scene = 16 operators
    (2, 30, 8),   # 2 scenes, 30 min/scene = 8 operators
    (2, 15, 4),   # 2 scenes, 15 min/scene = 4 operators
    (2, 7.5, 2),  # 2 scenes, 7.5 min/scene = 2 operators
    (2, 3.75, 1), # 2 scenes, 3.75 min/scene = 1 operator
    # 1 scene configs
    (1, 60, 16),  # 1 scene, 60 min/scene = 16 operators
    (1, 30, 8),   # 1 scene, 30 min/scene = 8 operators
    (1, 15, 4),   # 1 scene, 15 min/scene = 4 operators
    (1, 7.5, 2),  # 1 scene, 7.5 min/scene = 2 operators
    (1, 3.75, 1), # 1 scene, 3.75 min/scene = 1 operator
]

print('Generating config files by sampling from scene_diversity_8_60.yaml...')
for num_scenes, minutes_per_scene, num_operators in configs:
    create_config_file(num_scenes, minutes_per_scene, num_operators)

print('\nAll config files created!')

