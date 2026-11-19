"""gta6-trophy-rarity-forecast-01 entry: refresh-aware trophy-rate quantiles.

The grader runs `python3 -P -s /bot/main.py` (Python 3.13, standard library only, PYTHONHASHSEED=0),
one case per run: every `*.in` in $MAKEFUN_INPUT_DIR (default /input) becomes one `*.out` in
$MAKEFUN_OUTPUT_DIR (default /output),

    {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}

one row per case trophy with five finite, non-decreasing percentages in [0, 100], at most 65,536 bytes.
The forecast is refresh-aware: on a flat history it applies a launch-refresh prior, on a history with
real refreshes it fits players and earned counts across the refresh points and extrapolates them.

-P keeps this directory off sys.path, so the sys.path line below is needed for `import model`;
-s drops user site-packages.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds model.py beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
SIGMA = 0.5


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, like JavaScript's Math.round(x * 10000) / 10000."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


import model


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    if not case["trophies"]:
        raise ValueError(f"{name} has no trophies")
    return {"trophies": model.forecast_case(case, clamp, logit, round4, Z)}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
