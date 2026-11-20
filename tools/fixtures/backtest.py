#!/usr/bin/env python3
"""Backtest harness: fit a forecaster on snapshots up to a cutoff, score the
forecast at cutoff+delta against the nearest later snapshot, with the
competition's exact pinball loss on the logit scale.

Usage: python3 backtest.py --delta 6 [--end-hours 30] [--cutoffs N]
"""
import argparse
import json
import math
import os
import sys

LAUNCH = "2026-11-19T00:00:00Z"
SNAP = "/root/makefun-setup/gta6-trf/snapshots.ndjson"
KIT_CASE = "/root/makefun-setup/gta6-trf/gta6-trophy-rarity-forecast-01-kit/input/p01.in"
Z = (-1.6449, -0.6745, 0.0, 0.6745, 1.6449)
TAUS = (0.05, 0.25, 0.50, 0.75, 0.95)


def ts(s):
    y = int(s[0:4]); mo = int(s[5:7]); d = int(s[8:10])
    h = int(s[11:13]) if len(s) > 12 else 0
    mi = int(s[14:16]) if len(s) > 15 else 0
    se = int(s[17:19]) if len(s) > 18 else 0
    return ((d - 1) + (mo - 1) * 30 + (y - 2026) * 365) * 86400.0 + h * 3600 + mi * 60 + se


def logit(p):
    p = min(max(p, 0.01), 99.99) / 100.0
    return math.log(p / (1 - p))


def inv(x):
    x = min(max(x, math.log(0.0001 / 0.9999)), math.log(0.9999 / 0.0001))
    return 100.0 / (1.0 + math.exp(-x))


def ols(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 0.0
    mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    res = [y - (my + b * (x - mx)) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in res) / max(n - 2, 1))
    return b, sd, sxx


def load_snaps():
    rows = []
    with open(SNAP) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_case(snaps, cut_s, base):
    """case dict shaped like the kit's input, cut at cut_s (epoch)."""
    hist = []
    for r in snaps:
        if ts(r["at"]) < cut_s:
            hist.append({"at": r["at"], "players_tracked": r["players"],
                         "trophies_earned": r["earned"], "rates": r["rates"]})
    case = dict(base)
    case["history"] = hist
    case["at"] = snaps[0]["at"]
    return case


# ---------------- model variants ----------------
def q_from_median_sigma(m, sig):
    q = []
    prev = -1
    for z in Z:
        v = min(max(round(inv(m + z * sig), 4), 0.01), 99.99)
        v = max(v, prev); prev = v
        q.append(v)
    return q


def model_persist(case, tgt_s, days):
    out = []
    for j, t in enumerate(case["trophies"]):
        r = case["history"][-1]["rates"][j] if case["history"] else 1.0
        out.append({"id": t["id"], "q": [r] * 5})
    return out


def model_curve(case, tgt_s, days, window_h=24.0, pseudo=0.0, sig_k=1.0):
    """logit vs ln(1+h) OLS per trophy; pseudo>0 shrinks to the pooled slope.
    sig_k scales the spread."""
    hist = case["history"]
    t0 = ts(LAUNCH)
    reads = [(math.log(1 + max((ts(h["at"]) - t0) / 3600, 0)), h) for h in hist]
    xt = math.log(1 + max((tgt_s - t0) / 3600, 0))
    last_t = ts(reads[-1][1]["at"])
    win = [r for r in reads if ts(r[1]["at"]) >= last_t - window_h * 3600]
    if len(win) < 5:
        win = reads[-5:] if len(reads) >= 5 else reads
    xs_all = [r[0] for r in win]
    fits = {}
    for j, t in enumerate(case["trophies"]):
        xs, ys = [], []
        for x, h in win:
            r = h["rates"][j]
            if r is None:
                continue
            xs.append(x); ys.append(logit(r))
        fits[j] = (xs, ys)
    bs = [ols(*fits[j])[0] for j in fits if len(fits[j][0]) >= 3 and max(fits[j][1]) - min(fits[j][1]) > 1e-9]
    pooled = sum(bs) / len(bs) if bs else 0.0
    step = []
    for j in fits:
        xs, ys = fits[j]
        if len(xs) < 3:
            continue
        b = ols(xs, ys)[0]
        for k in range(1, len(xs)):
            dx = xs[k] - xs[k - 1]; dy = ys[k] - ys[k - 1]
            if dx > 1e-9 and abs(dy) > 1e-9:
                step.append((dy - b * dx) / math.sqrt(dx))
    if len(step) >= 8:
        mu = sum(step) / len(step)
        scatter = math.sqrt(sum((s - mu) ** 2 for s in step) / len(step))
    else:
        scatter = 0.30
    dx = xt - xs_all[-1]
    out = []
    for j, t in enumerate(case["trophies"]):
        r_last = case["history"][-1]["rates"][j]
        xs, ys = fits[j]
        if r_last >= 99.995:
            out.append({"id": t["id"], "q": [100.0] * 5}); continue
        if len(xs) >= 3:
            b_raw, sd_fit, sxx = ols(xs, ys)
            w = len(set(ys))
            b = (w * b_raw + pseudo * pooled) / (w + pseudo) if pseudo > 0 else b_raw
            ex = sd_fit * math.sqrt(1 + dx * dx / max(sxx, 1e-9))
        else:
            b = pooled if pseudo > 0 else 0.0
            ex = 0.0
            sd_fit = 0.0
        m = logit(r_last) + max(-2.5, min(2.5, b * dx))
        sig = math.sqrt(ex ** 2 + (scatter * abs(dx)) ** 2 + (0.45 * math.sqrt(days)) ** 2 + (0.45 * abs(dx)) ** 2)
        sig = max(0.40, min(2.2, sig)) * sig_k
        out.append({"id": t["id"], "q": q_from_median_sigma(m, sig)})
    return out


def model_decomp(case, tgt_s, days, window_h=24.0, pseudo=4.0, sig_k=1.0):
    """rate trend = per-trophy earner growth (shrunk) - shared pool growth."""
    hist = case["history"]
    t0 = ts(LAUNCH)
    reads = [(math.log(1 + max((ts(h["at"]) - t0) / 3600, 0)), h) for h in hist]
    xt = math.log(1 + max((tgt_s - t0) / 3600, 0))
    last_t = ts(reads[-1][1]["at"])
    win = [r for r in reads if ts(r[1]["at"]) >= last_t - window_h * 3600]
    if len(win) < 5:
        win = reads[-5:]
    xs_all = [r[0] for r in win]
    # pool growth on the same axis
    bN, _, _ = ols(xs_all, [math.log(r[1]["players_tracked"]) for r in win])
    bE = {}
    for j, t in enumerate(case["trophies"]):
        xs, ys = [], []
        for x, h in win:
            r = h["rates"][j]
            if r is None or r <= 0:
                continue
            xs.append(x); ys.append(math.log(max(r * h["players_tracked"] / 100.0, 0.5)))
        bE[j] = ols(xs, ys) if len(xs) >= 3 else (0.0, 0.0, 0.0)
    bs = [v[0] for v in bE.values() if v[2] > 1e-9]
    pooled = sum(bs) / len(bs) if bs else 0.0
    dx = xt - xs_all[-1]
    out = []
    for j, t in enumerate(case["trophies"]):
        r_last = case["history"][-1]["rates"][j]
        if r_last >= 99.995:
            out.append({"id": t["id"], "q": [100.0] * 5}); continue
        b_raw, sd_fit, sxx = bE[j]
        if sxx > 1e-9:
            b = (len(set(h["rates"][j] for h in win if h["rates"][j] is not None)) * b_raw + pseudo * pooled) / (len(set(h["rates"][j] for h in win if h["rates"][j] is not None)) + pseudo)
            ex = sd_fit * math.sqrt(1 + dx * dx / max(sxx, 1e-9))
        else:
            b = pooled; ex = 0.0
        shift = (b - bN) * dx
        m = logit(r_last) + max(-2.5, min(2.5, shift))
        sig = math.sqrt(ex ** 2 + (0.45 * math.sqrt(days)) ** 2 + (0.45 * abs(dx)) ** 2)
        sig = max(0.40, min(2.2, sig)) * sig_k
        out.append({"id": t["id"], "q": q_from_median_sigma(m, sig)})
    return out


def score(forecast, actual_rates):
    tot = 0.0
    n = 0
    cover = 0
    for row in forecast:
        j = row["id"] - 1
        y = actual_rates[j]
        ly = logit(y)
        q = row["q"]
        for tau, qi in zip(TAUS, q):
            d = ly - logit(qi)
            tot += 2 * max(tau * d, (tau - 1) * d)
            n += 1
        if q[0] <= y <= q[4]:
            cover += 1
    return tot / n, cover / len(forecast)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=6.0)
    ap.add_argument("--min-hours", type=float, default=8.0)
    ap.add_argument("--step", type=float, default=3.0)
    args = ap.parse_args()
    snaps = load_snaps()
    if len(snaps) < 3:
        print("not enough snapshots", len(snaps)); return
    base = json.load(open(KIT_CASE))
    t0 = ts(LAUNCH)
    variants = {
        "persist": lambda c, t, d: model_persist(c, t, d),
        "curve24": lambda c, t, d: model_curve(c, t, d, 24.0, 0.0),
        "curve24x0.7": lambda c, t, d: model_curve(c, t, d, 24.0, 0.0, 0.7),
        "curve24x1.4": lambda c, t, d: model_curve(c, t, d, 24.0, 0.0, 1.4),
        "shrunk24": lambda c, t, d: model_curve(c, t, d, 24.0, 4.0),
        "shrunk36": lambda c, t, d: model_curve(c, t, d, 36.0, 4.0),
        "shrunk24x0.7": lambda c, t, d: model_curve(c, t, d, 24.0, 4.0, 0.7),
        "shrunk24x1.4": lambda c, t, d: model_curve(c, t, d, 24.0, 4.0, 1.4),
        "decomp24": lambda c, t, d: model_decomp(c, t, d, 24.0, 4.0),
        "decomp36": lambda c, t, d: model_decomp(c, t, d, 36.0, 4.0),
    }
    first = ts(snaps[0]["at"]); last = ts(snaps[-1]["at"])
    cut = first + args.min_hours * 3600
    results = {k: [] for k in variants}
    n_case = 0
    while cut + args.delta * 3600 <= last:
        tgt = cut + args.delta * 3600
        later = [r for r in snaps if ts(r["at"]) >= tgt]
        if not later:
            break
        actual = later[0]["rates"]
        case = build_case(snaps, cut, base)
        days = args.delta / 24.0
        for name, fn in variants.items():
            fc = fn(case, tgt, days)
            loss, cov = score(fc, actual)
            results[name].append((loss, cov))
        n_case += 1
        cut += args.step * 3600
    print("delta %.1fh, %d cutoffs" % (args.delta, n_case))
    for name, rs in results.items():
        if not rs:
            continue
        loss = sum(r[0] for r in rs) / len(rs)
        cov = sum(r[1] for r in rs) / len(rs)
        print("%-14s loss %.4f  coverage %.3f" % (name, loss, cov))


if __name__ == "__main__":
    main()
