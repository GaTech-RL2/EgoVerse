import os
from pathlib import Path

# Pilot data root: zarr/, lance/ and deliveries/ live under it.
PILOT = Path(os.environ.get("EGOVERSE_LANCE_PILOT", "/coc/flash7/scratch/lance_pilot"))
