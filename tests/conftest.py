import sys
from pathlib import Path

# Make the sco2_ptc package importable without requiring PYTHONPATH to be set.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
