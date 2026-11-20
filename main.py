"""Launcher for the GTA VI trophy-rarity entry on make.fun.

Judging calls this script directly with ``python3 -P -s`` and a fixed hash seed.
Every ``*.in`` file under MAKEFUN_INPUT_DIR is one case; each case becomes a
``*.out`` file under MAKEFUN_OUTPUT_DIR. The payload is a dict with one
"trophies" key, whose value is a list of {"id", "q"} rows for all case trophies.
Each q list holds five finite percentages between 0 and 100 in ascending order.
Forecasts come from model.py: flat histories get a launch-refresh prior, while
histories with real refreshes are fit at the refresh points and extrapolated.
The program uses only the standard library and is fully deterministic.
"""

import json, math, os, sys
from pathlib import Path as Folder

# -P hides the script's own folder, so put it back to allow `import model`.
HERE = Folder(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import model

CASE_FOLDER = Folder(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
RESULT_FOLDER = Folder(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Requested quantiles, rebuilt from whole percentages to avoid typos.
LEVELS = [step / 100 for step in (5, 25, 50, 75, 95)]
# Standard-normal z values matching those five tail probabilities.
Z_SCORES = (-1.645, -0.674, 0.0, 0.674, 1.645)


def bound_pct(value: float) -> float:
    """Trim a percentage away from the exact endpoints before logging its odds."""
    return min(99.99, max(0.01, value))


def to_logit(pct: float) -> float:
    """Turn a bounded percentage into the log-odds coordinate used for spread."""
    trimmed = bound_pct(pct)
    return math.log(trimmed / (100.0 - trimmed))


def round_four(value: float) -> float | int:
    """Round to four decimals, halves upward, matching JS Math.round."""
    scaled = value * 10000
    floored = math.floor(scaled)
    if scaled - floored >= 0.5:
        floored += 1
    result = floored / 10000
    return int(result) if result.is_integer() else result


def check_case(case: dict, label: str) -> None:
    """Reject inputs that cannot produce a well-formed forecast."""
    expected = LEVELS
    if case["levels"] != expected:
        raise ValueError(f"{label} asks for levels {case['levels']}, not {expected}")
    if len(case["history"]) == 0:
        raise ValueError(f"{label} has no history")
    if len(case["trophies"]) == 0:
        raise ValueError(f"{label} has no trophies")


def predict(case: dict, label: str) -> dict:
    """Wrap the model's per-trophy rows in the output envelope."""
    check_case(case, label)
    rows = model.forecast_case(case, bound_pct, to_logit, round_four, Z_SCORES)
    return {"trophies": rows}


def run() -> None:
    """Process all cases in name order and write compact JSON files."""
    RESULT_FOLDER.mkdir(parents=True, exist_ok=True)
    for source in sorted(CASE_FOLDER.glob("*.in")):
        payload = json.loads(source.read_text(encoding="utf-8"))
        forecast = predict(payload, source.name)
        destination = RESULT_FOLDER / f"{source.stem}.out"
        destination.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == '__main__':
    run()
