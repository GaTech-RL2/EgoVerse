# Activate the EgoVerse-main venv (lives next to faive2lerobot's venv on
# the same image; the container ships the python interpreter).
source /capstor/store/cscs/swissai/a144/jiaqchen/egoverse/faive2lerobot/faive2lerobot/bin/activate

# Code root. Default is the capstor checkout; set EGOMAIN to use another copy
# (e.g. the iopsstor staging copy while capstor store is over its inode quota).
EGOMAIN=${EGOMAIN:-/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse-main}
export PYTHONPATH=${EGOMAIN}:$PYTHONPATH
export PYTHONPATH=${EGOMAIN}/external/lerobot:$PYTHONPATH

export HF_HOME=/iopsstor/scratch/cscs/jiaqchen/.hf_cache
export HF_DATASETS_CACHE=$HF_HOME/hf_datasets_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers_cache
export HF_HUB_CACHE=$HF_HOME/hub_cache

echo "[clariden] HF_HOME=$HF_HOME"
echo "[clariden] PYTHONPATH set for EGOMAIN=${EGOMAIN}"
