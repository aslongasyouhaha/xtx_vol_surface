"""Shared import paths; scripts retain caller control over the working directory."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / 'src', ROOT / 'scripts'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
