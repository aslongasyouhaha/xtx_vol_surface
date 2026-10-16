"""Make src importable when an entry script is executed directly."""
from pathlib import Path
from importlib.util import find_spec
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

# Prefer the installed dependency; also support this workspace's native build.
native_dependencies = ROOT / 'data' / 'outputs' / 'native_deps'
if find_spec('highspy') is None and (native_dependencies / 'highspy').is_dir():
    sys.path.append(str(native_dependencies))
