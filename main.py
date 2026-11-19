"""gta6-trophy-rarity-forecast-01 entry point (Python 3.13, standard library only).

Keeps the kit starter's I/O conventions and its clamp / logit / round4 helpers; the
actual forecast lives in model.py.  Each run sees one case, reads every <id>.in in
MAKEFUN_INPUT_DIR (default /input) and writes <id>.out to MAKEFUN_OUTPUT_DIR
(default /output).  No randomness, no wall clock, sorted iteration: two runs write
byte-identical files.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds
# model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import forecast_case as model_forecast_case  # noqa: E402  (sys.path fix first)

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Persistence fallback spread, in logit units per square-root day (the house baseline).
SIGMA = 0.5


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 does.

    A whole number comes back as an int, so json.dumps writes 100 where JavaScript
    writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def persistence_case(case: dict) -> dict:
    """The house baseline, used only if the model cannot produce a forecast."""
    last = case["history"][-1]
    spread = SIGMA * math.sqrt(case["horizon_minutes"] / 1440)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        lg = logit(rate if rate is not None else 0)
        rows.append({"id": trophy["id"], "q": [round4(100 / (1 + math.exp(-(lg + z * spread)))) for z in Z]})
    return {"trophies": rows}


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    if not case.get("history"):
        raise ValueError(f"{name} has no history")
    try:
        result = model_forecast_case(case)
    except Exception:
        # A forecast that is valid but mediocre beats an invalid case: never crash a case.
        result = persistence_case(case)
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
