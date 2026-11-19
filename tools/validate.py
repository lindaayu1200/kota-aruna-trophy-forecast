"""Local validation for the GTA VI trophy rarity forecast entry."""

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main.py"
FIXTURES = ROOT / "tools" / "fixtures"
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
MAX_BYTES = 65536


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct)
    return math.log(p / (100 - p))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def round2(x):
    whole = math.floor(x * 100 + 0.5)
    value = whole / 100
    return int(value) if value.is_integer() else value


def inv_logit(x):
    return 100 / (1 + math.exp(-x))


def item_loss(y, q, tau):
    d = logit(y) - logit(q)
    return 2 * max(tau * d, (tau - 1) * d)


def score(forecast, answer):
    rows = forecast["trophies"]
    values = [row["q"] for row in rows]
    total = 0.0
    miss = 0.0
    covered = 0
    for qs, y in zip(values, answer):
        for tau, q in zip(LEVELS, qs):
            total += item_loss(y, q, tau)
        miss += abs(y - qs[2])
        covered += qs[0] <= y <= qs[4]
    n = len(answer) * len(LEVELS)
    return {
        "loss": total / n,
        "typical_miss_pp": miss / len(answer),
        "coverage_90": covered / len(answer),
    }


def persistence(case):
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i] if i < len(last["rates"]) else None
        lg = logit(rate if rate is not None else 0)
        qs = [round4(100 / (1 + math.exp(-(lg + z * spread)))) for z in Z]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def answers(case):
    last = case["history"][-1]
    p1 = float(last.get("players_tracked") or 0)
    rates = list(last["rates"])
    platinum = next((i for i, t in enumerate(case["trophies"]) if t["tier"] == "platinum"), 0)
    full = [i for i, r in enumerate(rates) if r is not None and r >= 99.5]
    counts = [max(0.0, r) / 100 * p1 if r is not None else 0.0 for r in rates]
    weights = [count ** 0.25 for count in counts]
    other = [i for i in range(len(rates)) if i != platinum and i not in full]
    remaining = 4276 - (3868 if full else 0) - round(0.001 * 3868)
    total_weight = sum(weights[i] for i in other) or 1.0
    unchanged = [r if r is not None else 0.01 for r in rates]
    scaled = [100.0 if i in full else round2(r * 0.75) if r is not None else 0.01 for i, r in enumerate(rates)]
    if platinum not in full:
        scaled[platinum] = 100.0
    refreshed = []
    for i in range(len(rates)):
        if i in full:
            refreshed.append(100.0)
        elif i == platinum:
            refreshed.append(0.10)
        else:
            refreshed.append(round2(100 * remaining * weights[i] / total_weight / 3868))
    return {"A": unchanged, "B": scaled, "C": refreshed}


def synthetic_case():
    base = datetime(2026, 11, 19, tzinfo=timezone.utc)
    trophies = [
        {"id": 1, "name": "Leonida Legend", "description": "Congratulations!", "tier": "platinum", "points": 300, "secret": False},
        {"id": 2, "name": "Chapter One", "description": "Complete Chapter 1.", "tier": "bronze", "points": 15, "secret": True},
        {"id": 3, "name": "Card Shark", "description": "Complete 3 Collectible Card sets.", "tier": "silver", "points": 30, "secret": False},
        {"id": 4, "name": "Crew Races", "description": "Win 20 online multiplayer races with your crew.", "tier": "bronze", "points": 15, "secret": False},
    ]
    history = []
    for slot in range(10):
        refresh = slot // 3
        players = [1000, 1450, 2050, 2750][refresh]
        rates = [
            [0.05, 0.3, 0.22, 0.18][refresh],
            100.0,
            [2.0, 2.7, 3.5, 4.4][refresh],
            [0.8, 1.1, 1.4, 1.8][refresh] if slot not in (6, 9) else None,
        ]
        moment = base + timedelta(hours=slot)
        history.append({
            "slot": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "at": (moment + timedelta(seconds=12)).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "players_tracked": players,
            "trophies_earned": round(sum(r / 100 * players for r in rates if r is not None)),
            "rates": rates,
        })
    checkpoint = base + timedelta(hours=9, minutes=30)
    return {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": "s00",
        "at": checkpoint.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_minutes": 1440,
        "target": (checkpoint + timedelta(minutes=1440)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "levels": LEVELS,
        "game": {"name": "Grand Theft Auto VI", "platform": "PS5", "release_date": "2026-11-19", "source": ""},
        "trophies": trophies,
        "history": history,
        "notes": "Synthetic multi-refresh contract check.",
    }


def run_entry(input_dir, output_dir):
    env = dict(os.environ)
    env["MAKEFUN_INPUT_DIR"] = str(input_dir)
    env["MAKEFUN_OUTPUT_DIR"] = str(output_dir)
    env["PYTHONHASHSEED"] = "0"
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-P", "-s", str(MAIN)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return time.perf_counter() - started, result


def check_contract(case, output_dir):
    path = output_dir / f"{case['id']}.out"
    raw = path.read_bytes()
    assert len(raw) <= MAX_BYTES, f"{path.name} exceeds {MAX_BYTES} bytes"
    forecast = json.loads(raw)
    assert set(forecast) == {"trophies"}, f"{path.name} has unexpected top-level keys"
    rows = forecast["trophies"]
    expected = [t["id"] for t in case["trophies"]]
    assert isinstance(rows, list) and len(rows) == len(expected), f"{path.name} has the wrong row count"
    assert sorted(r["id"] for r in rows) == sorted(expected), f"{path.name} has the wrong ids"
    for row in rows:
        assert set(row) == {"id", "q"}, f"{path.name} has unexpected row keys"
        qs = row["q"]
        assert isinstance(qs, list) and len(qs) == len(LEVELS), f"{path.name} has the wrong quantile count"
        assert all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 100
            for v in qs
        ), f"{path.name} has an invalid quantile"
        assert all(a <= b for a, b in zip(qs, qs[1:])), f"{path.name} has unsorted quantiles"
    return raw


def main():
    cases = [json.loads((FIXTURES / f"{name}.in").read_text(encoding="utf-8")) for name in ("p00", "p01")]
    cases.append(synthetic_case())
    forecasts = {}
    durations = []

    with tempfile.TemporaryDirectory(prefix="trophy-forecast-") as tmp:
        root = Path(tmp)
        for run_index in range(2):
            input_dir = root / f"in{run_index}"
            output_dir = root / f"out{run_index}"
            input_dir.mkdir()
            for case in cases:
                (input_dir / f"{case['id']}.in").write_text(json.dumps(case), encoding="utf-8")
            elapsed, _ = run_entry(input_dir, output_dir)
            durations.append(elapsed)
            for case in cases:
                raw = check_contract(case, output_dir)
                if run_index == 0:
                    forecasts[case["id"]] = json.loads(raw)
                else:
                    assert raw == (input_dir.parent / f"out0" / f"{case['id']}.out").read_bytes(), (
                        f"{case['id']} changed between runs"
                    )

    print(f"contract: ok for {', '.join(c['id'] for c in cases)} (exact keys, ids, five sorted quantiles, size cap)")
    print(f"determinism: two runs byte-identical; wall time {durations[0]:.3f}s / {durations[1]:.3f}s (8s cap)")

    practice = cases[:2]
    scenario_scores = {}
    baseline_scores = {}
    for name in ("A", "B", "C"):
        model_losses = [score(forecasts[c["id"]], answers(c)[name])["loss"] for c in practice]
        base_losses = [score(persistence(c), answers(c)[name])["loss"] for c in practice]
        scenario_scores[name] = sum(model_losses) / len(model_losses)
        baseline_scores[name] = sum(base_losses) / len(base_losses)
        detail = score(forecasts[practice[0]["id"]], answers(practice[0])[name])
        print(
            f"scenario {name}: model {scenario_scores[name]:.6f} "
            f"(per case {model_losses[0]:.6f}, {model_losses[1]:.6f}), "
            f"persistence {baseline_scores[name]:.6f}; "
            f"p00 typical_miss {detail['typical_miss_pp']:.2f}pp "
            f"coverage90 {detail['coverage_90']:.3f}"
        )

    assert scenario_scores["C"] < 1.2, "model must score below 1.2 on the launch-refresh answer"
    assert scenario_scores["C"] < baseline_scores["C"], "model must beat persistence on the refresh answer"
    for name in ("A", "B", "C"):
        assert scenario_scores[name] < baseline_scores["C"], (
            f"model loss on {name} must beat the persistence baseline's refresh-answer loss"
        )
    print(
        f"assertions: every scenario's model loss < persistence's refresh-answer loss "
        f"({baseline_scores['C']:.6f}); model < 1.2 on C; all passed"
    )


if __name__ == "__main__":
    main()
