"""Dry-run launcher with the same interface as chase_paced.py (module name first) but no Adaption client."""
import importlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
if __name__ == "__main__":
    mod = importlib.import_module(sys.argv[1])
    sys.argv = [sys.argv[1] + ".py"] + sys.argv[2:]
    mod.main()
