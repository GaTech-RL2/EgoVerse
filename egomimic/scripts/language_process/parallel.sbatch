#!/bin/bash
#SBATCH --job-name=ray_zarr_convert
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12         # Ray will automatically see and use these 24 cores
#SBATCH --mem-per-cpu=8G           # 192GB total RAM to handle 24 concurrent conversions
#SBATCH --time=8:00:00
#SBATCH -A gts-dxu345-rl2
#SBATCH --output=logs/conversion/ray_convert_%j.out

cd /storage/home/hcoda1/4/paphiwetsa3/r-dxu345-0/projects/EgoVerse
source ~/.bashrc
source .venv/bin/activate && export PATH="/storage/project/r-dxu345-0/paphiwetsa3/install/bin:$PATH"

# --num-cpus-per-task defaults to 1, meaning 24 episodes process at the same time.
python egomimic/scripts/language_process/scale_to_zarr_annotation_parallel.py \
    --scale-annotation-dir annotations_test \
    --dataset-config-path egomimic/hydra_configs/data/cotrain_pi_lang.yaml \
    --conversion-mode pick_place_llm \
    --prompt-filepath egomimic/scripts/language_process/prompt.txt \
    --augment-prompt-filepath egomimic/scripts/language_process/augment_prompt.txt \
    --project-name "dense-language"