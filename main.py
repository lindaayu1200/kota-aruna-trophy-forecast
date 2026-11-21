"""Entry point for the make.fun GTA VI trophy-rarity challenge.

The judge invokes this script directly (``python3 -P -s`` with a fixed hash
seed).  Each ``*.in`` document found in MAKEFUN_INPUT_DIR (default ``/input``)
is one case, and one ``<stem>.out`` file is produced in MAKEFUN_OUTPUT_DIR
(default ``/output``).  The emitted object contains a single ``trophies`` key:
a list of ``{"id": n, "q": [...]}`` records, one per trophy of the case.  Every
``q`` holds five finite, ascending percentages in the closed 0.01..99.99 range.
The numbers come from model.forecast_case: flat histories receive a
launch-refresh prior, while histories with real refreshes are fitted at the
refresh points and extrapolated.  Only the standard library is used and the
program is fully deterministic.
"""

import json
import math
import os
import sys
from pathlib import Path

# Under ``-P`` the script's own directory is absent from sys.path; restore it
# so that the sibling ``model`` module remains importable.
_SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SOURCE_DIR))

import model

CASE_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Five requested tail levels, expressed as fractions rather than percentages.
LEVELS = [fraction / 100 for fraction in (5, 25, 50, 75, 95)]
# Inverse standard-normal CDF values for those same five tail levels.
Z_SCORES = (-1.645, -0.674, 0.0, 0.674, 1.645)

MIN_PCT = 0.01
MAX_PCT = 99.99


def bound_pct(value: float) -> float:
    """Keep a percentage off the exact 0/100 endpoints before logging odds."""
    return min(MAX_PCT, max(MIN_PCT, value))


def to_logit(pct: float) -> float:
    """Convert a bounded percentage into the log-odds spread coordinate."""
    safe = bound_pct(pct)
    return math.log(safe / (100.0 - safe))


def round_four(value: float) -> float | int:
    """Round half-up to four decimals, mirroring JavaScript's Math.round."""
    times_ten_k = value * 10000
    truncated = math.floor(times_ten_k)
    if times_ten_k - truncated >= 0.5:
        truncated += 1
    result = truncated / 10000
    return int(result) if result.is_integer() else result


def check_case(case: dict, label: str) -> None:
    """Guard against malformed cases that cannot yield a valid forecast."""
    if case["levels"] != LEVELS:
        raise ValueError(f"{label} asks for levels {case['levels']}, not {LEVELS}")
    if len(case["history"]) == 0:
        raise ValueError(f"{label} has no history")
    if len(case["trophies"]) == 0:
        raise ValueError(f"{label} has no trophies")


def predict(case: dict, label: str) -> dict:
    """Return the output envelope built from the model's per-trophy rows."""
    check_case(case, label)
    rows = model.forecast_case(case, bound_pct, to_logit, round_four, Z_SCORES)
    return {"trophies": rows}


def run() -> None:
    """Read cases in filename order and write one compact JSON file per case."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for source in sorted(CASE_DIR.glob("*.in")):
        case = json.loads(source.read_text(encoding="utf-8"))
        result = predict(case, source.name)
        target = OUT_DIR / f"{source.stem}.out"
        target.write_text(
            json.dumps(result, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    run()
