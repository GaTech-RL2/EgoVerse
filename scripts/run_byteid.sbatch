#!/bin/bash
#SBATCH --job-name=byteid
#SBATCH --time=00:20:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/byteid_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/byteid_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
srun python scripts/byteid_proof.py
echo "BYTEID_EXIT=$?"
