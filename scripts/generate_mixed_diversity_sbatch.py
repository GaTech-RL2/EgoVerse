#!/usr/bin/env python3
"""Generate SBATCH files for all mixed_diversity configs."""

import os
from pathlib import Path

# Config files in mixed_diversity directory
configs = [
    "mixed_diversity_4_4_15",
    "mixed_diversity_4_8_7_5",
    "mixed_diversity_6_4_10",
    "mixed_diversity_6_8_5",
    "mixed_diversity_8_4_7_5",
    "mixed_diversity_8_8_3_75",
]

# Template for sbatch file
template = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=sbatch_logs/{job_name}.out
#SBATCH --error=sbatch_logs/{job_name}.err
#SBATCH --partition="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --exclude="clippy"

source /coc/flash7/bli678/Shared/emimic/bin/activate

# Extract number of GPUs from SLURM_GPUS_PER_NODE (format: "l40s:4" -> 4)
NUM_GPUS_PER_NODE=$(echo ${{SLURM_GPUS_PER_NODE}} | cut -d: -f2)
export SLURM_GPUS=$((NUM_GPUS_PER_NODE * SLURM_NNODES))
echo "Using node: $SLURM_NODELIST, GPUs per node: $NUM_GPUS_PER_NODE, total GPUs: $SLURM_GPUS"

# Set PyTorch memory allocation to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python egomimic/trainHydra.py \\
    --config-name=train.yaml \\
    data=mixed_diversity/{config_name} \\
    logger.wandb.project=everse_mixed_diversity_fold_clothes \\
    name=fold-clothes \\
    description={description}
"""

# Create output directory
output_dir = Path('sbatch/mixed_diversity')
output_dir.mkdir(parents=True, exist_ok=True)

print('Generating SBATCH files for mixed_diversity configs...')
for config_name in configs:
    job_name = config_name
    # Create description from config name (e.g., mixed_diversity_4_4_15 -> mixed-4-4-15)
    description = config_name.replace('_', '-')
    
    content = template.format(
        job_name=job_name,
        config_name=config_name,
        description=description
    )
    
    filename = output_dir / f"{job_name}.sh"
    with open(filename, 'w') as f:
        f.write(content)
    
    # Make executable
    os.chmod(filename, 0o755)
    
    print(f'Created: {filename}')

print('\nAll SBATCH files created!')
