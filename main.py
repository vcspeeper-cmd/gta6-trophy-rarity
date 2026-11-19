"""gta6-trophy-rarity-forecast-01 entry: one case in, one forecast out.

Reads every <id>.in in the input directory (the grader runs one case at a time)
and writes <id>.out beside the output directory: for each of the case's trophies
the 5%, 25%, 50%, 75% and 95% quantiles of its earn rate at the case's target.
The model lives in model.py beside this file.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import forecast_case  # noqa: E402

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
