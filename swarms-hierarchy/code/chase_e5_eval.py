"""E5 evaluation: the frozen E3 evaluation (chase_e3_eval.py, unchanged) applied to the one exploratory cell of
amendment C6, Gemma-4-26B `norm0`, against results/chase_e5_predictions.json.

  chase_e5_eval.py [--reps 200]   -> results/chase_e5_eval.json

Nothing is analysed unless every group of the cell is present. The cell is exploratory and is not added to the
Holm family of E3; with one cell the adjusted p equals the raw p.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import chase_e3_eval as V

ROOT = Path(__file__).resolve().parents[1]
CELL = ("adaption:gemma-4-26b-a4b-it", "norm0")


def main():
    if any(x in sys.argv for x in ("--pop", "--predictions", "--out")):
        sys.exit("chase_e5_eval fixes its own files")
    V.TARGET = {CELL: {4: {"OWN": 40, "LOG": 40, "DYAD": 20, "REDACT": 20}, 8: {"OWN": 6, "LOG": 6}}}
    V.CONTROL = set()
    sys.argv += ["--pop", str(ROOT / "results" / "chase_e3_pop.jsonl"),
                 "--predictions", str(ROOT / "results" / "chase_e5_predictions.json"),
                 "--out", str(ROOT / "results" / "chase_e5_eval.json")]
    V.main()


if __name__ == "__main__":
    main()
