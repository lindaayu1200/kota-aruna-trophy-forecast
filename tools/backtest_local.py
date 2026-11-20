#!/usr/bin/env python3
"""Synthetic, seed-varying backtest for the trophy-rarity forecaster.

Simulates honest Exophase dynamics: logistic/power-law earned-count accrual,
tracked-pool jumps (dilution), 10-minute stale reads, a slow refresh schedule and
a stale answer read at the target.  Scores the entry against the house random
walk and a logit-linear trend, using the competition's pinball loss on logits.
"""

import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model

FIXTURE = ROOT / "tools" / "fixtures" / "p00.in"
LAUNCH = datetime(2026, 11, 19, tzinfo=timezone.utc)
CUT_H = 59.5
TARGET_H = 168.0
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)


def clamp(p):
    return min(99.99, max(0.01, p))


def logit(p):
    p = clamp(p)
    return math.log(p / (100.0 - p))


def inv_logit(x):
    x = max(-30.0, min(30.0, x))
    return 100.0 / (1.0 + math.exp(-x))


def stamp(hours):
    return (LAUNCH + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_simulation(seed, regime="sat"):
    rng = random.Random(seed)
    base = json.loads(FIXTURE.read_text(encoding="utf-8"))
    n = len(base["trophies"])
    families = [model.classify(t, i) for i, t in enumerate(base["trophies"])]
    full = [i for i, t in enumerate(base["trophies"]) if t["tier"] == "platinum"]

    history_refresh_h = [0.0, 4.5, 14.0, 38.0]
    future = []
    t = 46.0
    while t < TARGET_H:
        future.append(t)
        t += rng.uniform(7.0, 19.0)
    future.append(TARGET_H)
    refresh_h = history_refresh_h + future
    answer_index = max(k for k, h in enumerate(refresh_h) if h < TARGET_H)
    staleness = TARGET_H - refresh_h[answer_index]

    pool_growth = rng.uniform(0.0030, 0.0065)
    pool = [638.0]
    for k in range(1, len(refresh_h)):
        dt = refresh_h[k] - refresh_h[k - 1]
        if k == 1:
            jump = 3868.0 / 638.0
        elif k == 2:
            jump = 14028.0 / 3868.0
        else:
            jump = rng.uniform(1.10, 1.35) if rng.random() < 0.55 else 1.0
        grown = pool[-1] * math.exp(pool_growth * dt) * jump
        pool.append(min(100000.0, grown))

    counts = []
    ceiling = {
        "story": 1200.0, "platinum": 900.0, "collectible": 650.0, "stunt": 700.0,
        "grind": 450.0, "online": 420.0, "other": 600.0,
    }
    for i in range(n):
        family = families[i]
        tau = rng.uniform(22.0, 55.0) * (0.8 if family in ("story", "platinum") else 1.15)
        gamma = rng.uniform(0.50, 0.80)
        limit = ceiling[family] * rng.uniform(0.7, 1.4)
        seed_count = rng.uniform(2.0, 8.0)
        counts.append((limit, tau, gamma, seed_count))

    truth_counts = {}
    for i, (limit, tau, gamma, seed_count) in enumerate(counts):
        if regime == "sat":
            truth_counts[i] = [seed_count + limit * (1.0 - math.exp(-((h / tau) ** gamma))) for h in refresh_h]
        else:
            beta = 0.55 + 0.40 * gamma
            truth_counts[i] = [seed_count + limit * 0.5 * ((1.0 + h / (0.6 * tau)) ** beta - 1.0) for h in refresh_h]

    readings = {}
    for k, h in enumerate(refresh_h):
        rates = []
        for i in range(n):
            if i in full:
                rates.append(100.0)
                continue
            noisy = max(0.0, rng.gauss(truth_counts[i][k], 0.55 * math.sqrt(max(1.0, truth_counts[i][k]))))
            rate = round(10000.0 * noisy / pool[k]) / 100.0
            rates.append(min(99.99, max(0.0, rate)))
        readings[k] = rates

    history = []
    slot = 0.0
    while slot < CUT_H - 1e-9:
        k = max(idx for idx, h in enumerate(history_refresh_h) if h <= slot + 1e-9)
        rates = readings[k]
        history.append({
            "slot": stamp(slot), "at": stamp(slot),
            "players_tracked": int(round(pool[k])),
            "trophies_earned": int(round(sum(rates))),
            "rates": list(rates),
        })
        slot += 1.0 / 6.0

    case = dict(base)
    case["id"] = f"s{seed:02d}"
    case["at"] = stamp(CUT_H)
    case["history"] = history
    case["horizon_minutes"] = 6510
    case["target"] = stamp(TARGET_H)
    debug = {
        "answer_h": refresh_h[answer_index],
        "answer_pool": pool[answer_index],
        "answer_counts": {i: truth_counts[i][answer_index] for i in range(n)},
        "refresh_h": refresh_h,
        "pools": pool,
    }
    return case, readings[answer_index], staleness, debug


def quantile_rows(medians, sigmas):
    rows = []
    for median, sigma in zip(medians, sigmas):
        center = logit(median)
        qs = [clamp(inv_logit(center + z * sigma)) for z in Z]
        qs = sorted(round(q, 4) for q in qs)
        rows.append(qs)
    return rows


def house_baseline(case):
    days = case["horizon_minutes"] / 1440.0
    spread = 0.5 * math.sqrt(days)
    rates = case["history"][-1]["rates"]
    return quantile_rows(rates, [spread] * len(rates))


def linear_baseline(case):
    days = case["horizon_minutes"] / 1440.0
    spread = 0.5 * math.sqrt(days)
    anchor = model.launch_hours(case, case["history"])
    points = []
    for read in case["history"]:
        sig = (read["players_tracked"], tuple(read["rates"]))
        if points and sig == points[-1][0]:
            continue
        points.append((sig, model.read_hours(read, anchor)))
    xs = [math.log1p(h) for _, h in points]
    x_end = xs[-1]
    rows = []
    n = len(case["trophies"])
    for i in range(n):
        ys = [logit(sig[1][i]) for sig, _ in points if i < len(sig[1]) and sig[1][i] is not None]
        if len(ys) < 2:
            ys = [logit(case["history"][-1]["rates"][i])]
        slope, intercept, _ = model.ols(list(zip(xs[-len(ys):], ys)))
        median = inv_logit(intercept + slope * (x_end + 0.5 * math.sqrt(days)))
        rows.append((median, spread))
    return quantile_rows([m for m, _ in rows], [s for _, s in rows])


def model_forecast(case):
    doc = model.forecast_case(case, clamp, logit, lambda x: round(x, 4), Z)
    return [row["q"] for row in doc]


def loss(forecast, truth):
    total = 0.0
    covered = 0
    for qs, y in zip(forecast, truth):
        ly = logit(y)
        for tau, q in zip(TAUS, qs):
            d = ly - logit(q)
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
        covered += qs[0] <= y <= qs[4]
    return total / (len(truth) * len(TAUS)), covered / len(truth)


def main():
    seeds = [int(a) for a in sys.argv[1:]] or [11, 22, 33, 44, 55]
    for regime in ("sat", "power"):
        results = {"model": [], "house": [], "linear": []}
        print(f"--- regime {regime} ---")
        for seed in seeds:
            case, truth, staleness, _ = build_simulation(seed, regime)
            scores = {
                "model": loss(model_forecast(case), truth),
                "house": loss(house_baseline(case), truth),
                "linear": loss(linear_baseline(case), truth),
            }
            for name, (value, coverage) in scores.items():
                results[name].append(value)
            print(
                f"seed {seed:02d} staleness {staleness:5.1f}h  "
                f"model {scores['model'][0]:.4f} (cov {scores['model'][1]:.2f})  "
                f"house {scores['house'][0]:.4f}  linear {scores['linear'][0]:.4f}"
            )
        for name in results:
            print(f"MEAN {name:6s} {sum(results[name]) / len(results[name]):.4f}")
        winners = sum(
            1 for a, b, c in zip(results["model"], results["house"], results["linear"])
            if a < b and a < c
        )
        print(f"model wins {winners}/{len(seeds)} seeds")


if __name__ == "__main__":
    main()
