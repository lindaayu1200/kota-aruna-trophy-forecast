#!/usr/bin/env python3
"""Local validator for the gta6-trophy-rarity-forecast-01 entry.

Usage:  python3 tools/validate.py

Runs main.py on the local practice fixtures in a temporary directory with
MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR set, checks the output contract, scores the
forecast against (a) a persistence answer, (b) the first-refresh answer and (c) the
live-page launch refresh, checks determinism over two independent runs, and prints
the numbers.  Standard library
only.  Override the fixture path with MAKEFUN_FIXTURE_DIR if needed.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(os.environ.get("MAKEFUN_FIXTURE_DIR", ROOT / "tools" / "fixtures"))
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
HOUSE_LOSS = 1.51  # the house baselines' reported practice loss
LIVE_PLAYERS = 3868
LIVE_EARNED = 4276
LIVE_PLATINUM_RATE = 0.10


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    x = clamp(pct)
    return math.log(x / (100 - x))


def round2(x: float) -> float:
    return math.floor(x * 100 + 0.5) / 100


def run_program(input_dir: Path, output_dir: Path) -> float:
    env = os.environ.copy()
    env["MAKEFUN_INPUT_DIR"] = str(input_dir)
    env["MAKEFUN_OUTPUT_DIR"] = str(output_dir)
    env["PYTHONHASHSEED"] = "0"
    start = time.monotonic()
    subprocess.run([sys.executable, "-P", "-s", str(ROOT / "main.py")],
                   env=env, check=True, timeout=30)
    return time.monotonic() - start


def check_contract(case: dict, path: Path) -> dict:
    raw = path.read_bytes()
    assert 0 < len(raw) <= 65536, f"{path.name}: output size {len(raw)}"
    obj = json.loads(raw)
    assert isinstance(obj, dict) and set(obj) == {"trophies"}, f"{path.name}: top-level keys"
    rows = obj["trophies"]
    assert isinstance(rows, list), f"{path.name}: trophies is not a list"
    expected_ids = sorted(t["id"] for t in case["trophies"])
    assert len(rows) == len(expected_ids), f"{path.name}: {len(rows)} rows, expected {len(expected_ids)}"
    assert sorted(r["id"] for r in rows) == expected_ids, f"{path.name}: ids"
    for row in rows:
        assert set(row) == {"id", "q"}, f"{path.name}: row keys {set(row)}"
        assert isinstance(row["id"], int) and not isinstance(row["id"], bool), f"{path.name}: id type"
        q = row["q"]
        assert isinstance(q, list) and len(q) == len(LEVELS), f"{path.name}: q length"
        for value in q:
            assert isinstance(value, (int, float)) and not isinstance(value, bool), f"{path.name}: q type"
            assert math.isfinite(value) and 0.0 <= value <= 100.0, f"{path.name}: q range {value}"
        assert all(q[i] <= q[i + 1] for i in range(len(q) - 1)), f"{path.name}: q not non-decreasing"
    return obj


def score(case: dict, forecast: dict, actual: dict[int, float]) -> dict:
    """Exactly the competition's pinball loss, typical miss and 90% coverage."""
    total, items, miss, covered = 0.0, 0, 0.0, 0
    by_id = {row["id"]: row["q"] for row in forecast["trophies"]}
    for trophy in case["trophies"]:
        y = actual[trophy["id"]]
        q = by_id[trophy["id"]]
        for tau, quantile in zip(LEVELS, q):
            d = logit(y) - logit(quantile)
            total += 2 * max(tau * d, (tau - 1) * d)
            items += 1
        miss += abs(y - q[2])
        covered += int(q[0] <= y <= q[4])
    n = len(case["trophies"])
    return {"loss": round(total / items, 6),
            "typical_miss_pp": round(miss / n, 2),
            "coverage_90": round(covered / n, 4)}


def last_rates(case: dict) -> list[float]:
    return [clamp(float(r)) if r is not None else 0.01 for r in case["history"][-1]["rates"]]


def persistence_forecast(case: dict) -> dict:
    rates = last_rates(case)
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440)
    rows = []
    for trophy, rate in zip(case["trophies"], rates):
        centre = logit(rate)
        q = [100 / (1 + math.exp(-(centre + z * spread))) for z in Z]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def scenario_persistence(case: dict) -> dict[int, float]:
    return {t["id"]: r for t, r in zip(case["trophies"], last_rates(case))}


def scenario_refresh(case: dict) -> dict[int, float]:
    """(b): platinum 100, every other rate x0.75 rounded to 2 decimals."""
    answer = {}
    for trophy, rate in zip(case["trophies"], last_rates(case)):
        answer[trophy["id"]] = 100.0 if trophy["tier"] == "platinum" else round2(0.75 * rate)
    return answer


def scenario_high_rate_stays(case: dict) -> dict[int, float]:
    """Diagnostic: the trophy already at 100% stays there, the rest x0.75."""
    return {t["id"]: (100.0 if r >= 99.5 else round2(0.75 * r))
            for t, r in zip(case["trophies"], last_rates(case))}


def scenario_house_like(case: dict) -> dict[int, float]:
    """Diagnostic matching the house's 1.51 / 0.68 practice figures.

    The 100%-rate trophy staying at 100 forces the other rates down by about
    (4276/3868 - 1) / (943/638 - 1) ~ 0.22, not 0.75: 0.75 is the fall in the
    average earned per tracked player, not the multiplier on each rare trophy.
    """
    return {t["id"]: (100.0 if r >= 99.5 else round2(0.22 * r))
            for t, r in zip(case["trophies"], last_rates(case))}


def scenario_launch_refresh(case: dict) -> dict[int, float]:
    """(C): the live-page refresh answer, rebuilt from the observed aggregates.

    The 100% trophy stays at 100 and the platinum takes its observed 0.10%.  The
    other earned counts grow as c_i**0.25, scaled so their sum plus the saturated
    trophy and the platinum sums to 4,276 earned trophies over 3,868 players.
    """
    rates = last_rates(case)
    players1 = float(case["history"][-1]["players_tracked"])
    counts = [rate / 100.0 * players1 for rate in rates]
    grown = {i: counts[i] ** 0.25
             for i, (trophy, rate) in enumerate(zip(case["trophies"], rates))
             if trophy["tier"] != "platinum" and rate < 99.5}
    remaining = LIVE_EARNED - LIVE_PLAYERS - LIVE_PLATINUM_RATE / 100.0 * LIVE_PLAYERS
    total = math.fsum(grown.values())
    scale = remaining / total if total > 0 else 0.0
    answer = {}
    for i, trophy in enumerate(case["trophies"]):
        if trophy["tier"] == "platinum":
            answer[trophy["id"]] = LIVE_PLATINUM_RATE
        elif rates[i] >= 99.5:
            answer[trophy["id"]] = 100.0
        else:
            answer[trophy["id"]] = round2(scale * grown.get(i, counts[i]) / LIVE_PLAYERS * 100.0)
    return answer


def synthetic_case() -> dict:
    """A small multi-refresh case so the validator also exercises the fitted branch.

    The two practice cases are entirely flat, so on their own they cannot check the
    refresh detection, the two OLS fits or their extrapolation.  This case has five
    refresh points, duplicated unchanged reads between them and a 4.5-day horizon.
    """
    trophies = [
        {"id": 1, "name": "Platinum", "description": "Finish everything", "tier": "platinum",
         "points": 300, "secret": False},
        {"id": 2, "name": "Chapter One", "description": "Complete Chapter 1", "tier": "bronze",
         "points": 15, "secret": True},
        {"id": 3, "name": "Collector", "description": "Discover 100 locations", "tier": "silver",
         "points": 30, "secret": False},
    ]
    origin = datetime(2026, 11, 18, 23, 50, tzinfo=timezone.utc)

    def stamp(hours: float) -> str:
        return (origin + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")

    refreshes = [(0.0, 638), (4.5, 3868), (11.0, 5100), (26.0, 7200), (52.0, 9100)]
    exponents = [0.0, 0.8, 0.55]
    reads = []
    for hours, players in refreshes:
        earned = [4.0] + [4.0 * ((1.0 + hours) ** e) for e in exponents[1:]]
        rates = [round(min(100.0, 100.0 * c / players), 2) for c in earned]
        for slot in (hours, hours + 1.0, hours + 2.0):
            if slot >= 58.0:
                break
            reads.append({"slot": stamp(slot), "at": stamp(slot), "players_tracked": players,
                          "trophies_earned": sum(int(r / 100 * players) for r in rates),
                          "rates": list(rates)})
    return {"schema": "makefun/gta6-trophies-case/v1", "id": "synth", "at": stamp(58.0),
            "horizon_minutes": 6510, "target": stamp(166.5),
            "levels": LEVELS, "game": {"name": "Synthetic", "release_date": "2026-11-19"},
            "trophies": trophies, "history": reads, "notes": "multi-refresh smoke test"}


def fmt(metrics: dict) -> str:
    return (f"loss={metrics['loss']:.6f}  "
            f"typical_miss_pp={metrics['typical_miss_pp']:.2f}  "
            f"coverage_90={metrics['coverage_90']:.4f}")


def main() -> None:
    assert FIXTURES.is_dir(), f"fixture directory not found: {FIXTURES}"
    case_paths = sorted(FIXTURES.glob("*.in"))
    assert case_paths, f"no practice cases in {KIT_INPUT}"
    names = [p.stem for p in case_paths]
    assert "p00" in names and "p01" in names, f"practice cases missing: {names}"
    practice = list(names)
    synthetic = synthetic_case()
    names = sorted(names + [synthetic["id"]])

    cases = {}
    forecasts = {}
    elapsed = []
    blobs = {}
    with tempfile.TemporaryDirectory(prefix="makefun-validate-") as tmp:
        root = Path(tmp)
        for run_no in (1, 2):
            work = root / f"run{run_no}"
            in_dir, out_dir = work / "input", work / "output"
            in_dir.mkdir(parents=True)
            for src in case_paths:
                shutil.copyfile(src, in_dir / src.name)
            (in_dir / f"{synthetic['id']}.in").write_text(
                json.dumps(synthetic, separators=(",", ":")), encoding="utf-8")
            elapsed.append(run_program(in_dir, out_dir))
            out_paths = sorted(out_dir.glob("*.out"))
            assert [p.stem for p in out_paths] == names, f"run {run_no}: output files"
            blobs[run_no] = {p.name: p.read_bytes() for p in out_paths}
            if run_no == 1:
                for src in sorted(in_dir.glob("*.in")):
                    case = json.loads(src.read_text(encoding="utf-8"))
                    cases[src.stem] = case
                    forecasts[src.stem] = check_contract(case, out_dir / f"{src.stem}.out")

    assert blobs[1] == blobs[2], "two runs produced different bytes"
    assert max(elapsed) < 8.0, f"run took {max(elapsed):.3f}s"

    print(f"fixtures:  {FIXTURES}")
    print(f"contract:  OK for {', '.join(names)} "
          f"({', '.join(str(len(f['trophies'])) for f in forecasts.values())} rows; "
          f"{', '.join(str(len(blobs[1][n + '.out'])) for n in names)} bytes)")
    print(f"runtime:   {elapsed[0]:.3f}s / {elapsed[1]:.3f}s (limit 8s)")
    print("determinism: two runs byte-identical  OK")

    flat = {name: score(cases[name], forecasts[name], scenario_persistence(cases[name]))
            for name in practice}
    refresh = {name: score(cases[name], forecasts[name], scenario_refresh(cases[name]))
               for name in practice}
    launch_answers = {name: scenario_launch_refresh(cases[name]) for name in practice}
    launch = {name: score(cases[name], forecasts[name], launch_answers[name])
              for name in practice}
    base_launch = {name: score(cases[name], persistence_forecast(cases[name]), launch_answers[name])
                   for name in practice}
    high = {name: score(cases[name], forecasts[name], scenario_high_rate_stays(cases[name]))
            for name in practice}
    house = {name: score(cases[name], forecasts[name], scenario_house_like(cases[name]))
             for name in practice}
    base_refresh = {name: score(cases[name], persistence_forecast(cases[name]),
                                scenario_refresh(cases[name])) for name in practice}
    base_house = {name: score(cases[name], persistence_forecast(cases[name]),
                              scenario_house_like(cases[name])) for name in practice}

    def averaged(scores: dict[str, dict]) -> dict:
        keys = scores[practice[0]]
        return {key: round(sum(scores[name][key] for name in practice) / len(practice), 6) for key in keys}

    print("\n(A) persistence answer — rates unchanged")
    for name in practice:
        print(f"  {name} model: {fmt(flat[name])}")
    print(f"  both: {fmt(averaged(flat))}")
    print("\n(B) refresh answer — platinum 100, other rates x0.75")
    for name in practice:
        print(f"  {name} model: {fmt(refresh[name])}")
        print(f"  {name} house persistence baseline: {fmt(base_refresh[name])}")
    print(f"  both model: {fmt(averaged(refresh))}")
    print(f"  both persistence: {fmt(averaged(base_refresh))}")
    print("\n(C) live-page launch refresh — platinum 0.10%, 100% trophy unchanged, "
          "others grown as c_i^0.25 to 4,276 earned / 3,868 players")
    for name in practice:
        print(f"  {name} model: {fmt(launch[name])}")
        print(f"  {name} house persistence baseline: {fmt(base_launch[name])}")
    print(f"  both model: {fmt(averaged(launch))}")
    print(f"  both persistence: {fmt(averaged(base_launch))}")
    print("\nDiagnostics (the two readings of 'the 100% platinum' in the kit prose)")
    print(f"  100%-rate trophy stays 100, others x0.75: model {fmt(averaged(high))}")
    print(f"  house-like x0.22 (matches the 1.51 / 0.68 report):")
    print(f"    model {fmt(averaged(house))}")
    print(f"    persistence {fmt(averaged(base_house))}")

    for scenario, model_scores, base_scores in (
        ("C", launch, base_launch),
        ("house-like", house, base_house),
    ):
        for name in practice:
            assert model_scores[name]["loss"] < base_scores[name]["loss"], \
                f"{name}: model does not beat persistence under scenario ({scenario})"
    for name in practice:
        assert launch[name]["loss"] < 1.2, \
            f"{name}: model loss {launch[name]['loss']} >= 1.2 on scenario (C)"

    print("\nPASS: contract, determinism, runtime, and all loss comparisons.")


if __name__ == "__main__":
    main()
