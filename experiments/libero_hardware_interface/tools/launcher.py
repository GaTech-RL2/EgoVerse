"""Run from this study directory or from the repository root."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from astra_reversal.hardware_interface.launcher import main

if __name__ == "__main__":
    main()
