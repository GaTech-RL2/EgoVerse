#!/usr/bin/env python3
"""Generate scene diversity config files by sampling from scene_diversity_16_60.yaml."""

import yaml
import re
from collections import defaultdict

# Read base config
with open('egomimic/hydra_configs/data/scene_diversity/scene_diversity_16_60.yaml', 'r') as f:
    base = yaml.safe_load(f)

base_datasets = base['train_datasets']['dataset1']['datasets']

def extract_datasets(num_entries_per_scene):
    """Extract datasets for given number of entries per scene.
    Prioritizes having as many operators as possible (one episode per operator first)."""
    filtered_datasets = {}
    
    # Group by scene first
    scenes_dict = defaultdict(lambda: defaultdict(list))
    
    for key, value in base_datasets.items():
        if key.startswith('#'):
            continue
        match = re.match(r'op(\d+)_scene(\d+)_ep(\d+)', key)
        if match:
            op = int(match.group(1))
            scene = int(match.group(2))
            ep = int(match.group(3))
            
            scenes_dict[scene][op].append((ep, key, value))
    
    # Sort scenes
    sorted_scenes = sorted(scenes_dict.keys())
    
    for scene_num in sorted_scenes:
        # Sort operators
        sorted_operators = sorted(scenes_dict[scene_num].keys())
        
        # First pass: Take one episode from as many operators as possible
        entries_collected = 0
        operators_used = []
        
        for op in sorted_operators:
            if entries_collected >= num_entries_per_scene:
                break
            
            # Sort episodes for this operator
            episodes = sorted(scenes_dict[scene_num][op], key=lambda x: x[0])
            
            # Take first episode from this operator
            if episodes:
                ep, key, value = episodes[0]
                filtered_datasets[key] = value
                operators_used.append(op)
                entries_collected += 1
        
        # Second pass: If we still need more entries, take additional episodes
        # from operators we've already used (in order)
        if entries_collected < num_entries_per_scene:
            for op in operators_used:
                if entries_collected >= num_entries_per_scene:
                    break
                
                # Sort episodes for this operator
                episodes = sorted(scenes_dict[scene_num][op], key=lambda x: x[0])
                
                # Take additional episodes (skip the first one we already took)
                for ep, key, value in episodes[1:]:
                    if entries_collected >= num_entries_per_scene:
                        break
                    filtered_datasets[key] = value
                    entries_collected += 1
    
    return filtered_datasets

def create_config_file(minutes_per_scene, num_entries_per_scene):
    """Create a config file with specified entries per scene"""
    datasets = extract_datasets(num_entries_per_scene)
    
    # Format filename
    time_str = str(minutes_per_scene).replace('.', '_')
    filename = f'egomimic/hydra_configs/data/scene_diversity/scene_diversity_16_{time_str}.yaml'
    
    # Group datasets by scene for proper formatting
    scenes_dict = defaultdict(lambda: defaultdict(list))
    for key, value in datasets.items():
        match = re.match(r'op(\d+)_scene(\d+)_ep(\d+)', key)
        if match:
            scene_num = int(match.group(2))
            op_num = int(match.group(1))
            ep_num = int(match.group(3))
            scenes_dict[scene_num][op_num].append((ep_num, key, value))
    
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
        f.write('train_datasets:\n')
        f.write('  dataset1:\n')
        f.write('    _target_: egomimic.rldb.utils.MultiRLDBDataset\n')
        f.write('    datasets:\n')
        f.write('\n')
        
        # Write datasets grouped by scene
        sorted_scenes = sorted(scenes_dict.keys())
        for scene_num in sorted_scenes:
            f.write(f'      # scene {scene_num}\n')
            # Sort operators
            sorted_operators = sorted(scenes_dict[scene_num].keys())
            for op_num in sorted_operators:
                # Sort episodes for this operator
                episodes = sorted(scenes_dict[scene_num][op_num], key=lambda x: x[0])
                for ep_num, key, value in episodes:
                    f.write(f'      {key}:\n')
                    f.write(f'        _target_: {value["_target_"]}\n')
                    f.write(f'        bucket_name: "{value["bucket_name"]}"\n')
                    f.write(f'        mode: {value["mode"]}\n')
                    f.write(f'        embodiment: "{value["embodiment"]}"\n')
                    f.write(f'        local_files_only: {value["local_files_only"]}\n')
                    f.write(f'        temp_root: "{value["temp_root"]}"\n')
                    episode_hash = value["filters"]["episode_hash"]
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

# Configurations to create
# Each entry is 3.75 minutes, so:
# 30 min = 8 entries, 15 min = 4 entries, 7.5 min = 2 entries, 3.75 min = 1 entry
configs = [
    (30, 8),   # 30 min/scene = 8 entries
    (15, 4),   # 15 min/scene = 4 entries
    (7.5, 2),  # 7.5 min/scene = 2 entries
    (3.75, 1), # 3.75 min/scene = 1 entry
]

print('Generating config files by sampling from scene_diversity_16_60.yaml...')
for minutes_per_scene, num_entries_per_scene in configs:
    create_config_file(minutes_per_scene, num_entries_per_scene)

print('\nAll config files created!')
