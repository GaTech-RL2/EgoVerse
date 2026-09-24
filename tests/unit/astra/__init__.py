"""Make the standalone experiment importable with the pytest console command."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
