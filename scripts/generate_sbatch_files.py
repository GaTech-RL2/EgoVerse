#!/usr/bin/env python3
"""Generate SBATCH files for all scene diversity configs."""

# Read template
with open('sbatch/scene_diversity/scene_diversity_8_60.sh', 'r') as f:
    template = f.read()

# Configurations
configs = [
    # 8 scene configs
    (8, 60, 'scenes-8-time-60'),
    (8, 30, 'scenes-8-time-30'),
    (8, 15, 'scenes-8-time-15'),
    (8, 7.5, 'scenes-8-time-7_5'),
    (8, 3.75, 'scenes-8-time-3_75'),
    # 4 scene configs
    (4, 60, 'scenes-4-time-60'),
    (4, 30, 'scenes-4-time-30'),
    (4, 15, 'scenes-4-time-15'),
    (4, 7.5, 'scenes-4-time-7_5'),
    (4, 3.75, 'scenes-4-time-3_75'),
    # 2 scene configs
    (2, 60, 'scenes-2-time-60'),
    (2, 30, 'scenes-2-time-30'),
    (2, 15, 'scenes-2-time-15'),
    (2, 7.5, 'scenes-2-time-7_5'),
    (2, 3.75, 'scenes-2-time-3_75'),
    # 1 scene configs
    (1, 60, 'scenes-1-time-60'),
    (1, 30, 'scenes-1-time-30'),
    (1, 15, 'scenes-1-time-15'),
    (1, 7.5, 'scenes-1-time-7_5'),
    (1, 3.75, 'scenes-1-time-3_75'),
]

print('Generating SBATCH files...')
for num_scenes, minutes_per_scene, description in configs:
    time_str = str(minutes_per_scene).replace('.', '_')
    job_name = f'scene_diversity_{num_scenes}_{time_str}'
    config_name = f'scene_diversity_{num_scenes}_{time_str}'
    
    # Replace in template
    content = template.replace('scene_diversity_8_60', job_name)
    content = content.replace('scenes-8-time-60', description)
    # Ensure partition is hoffman-lab (in case template was updated)
    content = content.replace('"rl2-lab"', '"hoffman-lab"')
    
    filename = f'sbatch/scene_diversity/{job_name}.sh'
    with open(filename, 'w') as f:
        f.write(content)
    
    print(f'Created: {filename}')

print('\nAll SBATCH files created!')
