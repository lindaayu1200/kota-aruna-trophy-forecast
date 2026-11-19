"""Refresh-aware quantile forecast for the GTA VI trophy case."""

import math
from datetime import datetime

FAMILY_PRIOR = {
    "platinum": 0.0,
    "story": 0.85,
    "collectible": 0.55,
    "stunt": 0.45,
    "online": 0.30,
    "other": 0.50,
}

ONLINE_WORDS = ("online", "multiplayer", "crew", "co-op", "coop", "app", "scooter bros")
STUNT_WORDS = ("stunt", "trick", "race", "drift", "jump", "drive-by", "bowling")
COLLECT_WORDS = (
    "collect", "grind", "discover", "purchase", "unlock", "find card",
    "sets", "properties", "100%", "all shops", "all garage", "meditate at every",
)
STORY_WORDS = ("chapter", "mission", "story", "strand", "heist", "final mission")


def hours_between(first, second):
    def parse(value):
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except (ValueError, TypeError):
            return None

    a, b = parse(first), parse(second)
    if a is None or b is None:
        return 0.0
    return (b - a) / 3600.0


def classify(trophy, index):
    text = f"{trophy.get('name', '')} {trophy.get('description', '')}".lower()
    tier = str(trophy.get("tier", "")).lower()
    if tier == "platinum" or (index == 0 and tier == ""):
        return "platinum"
    if any(word in text for word in ONLINE_WORDS):
        return "online"
    if any(word in text for word in STORY_WORDS):
        return "story"
    if any(word in text for word in STUNT_WORDS):
        return "stunt"
    if any(word in text for word in COLLECT_WORDS):
        return "collectible"
    if trophy.get("secret"):
        return "story"
    return "other"


def ols(points):
    n = len(points)
    if n < 2:
        return 0.0, (points[-1][1] if points else 0.0), 0.0
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    mx, my = sx / n, sy / n
    den = sum((x - mx) ** 2 for x, _ in points)
    if den <= 1e-12:
        return 0.0, my, math.sqrt(sum((y - my) ** 2 for _, y in points) / n)
    slope = sum((x - mx) * (y - my) for x, y in points) / den
    intercept = my - slope * mx
    rmse = math.sqrt(sum((y - slope * x - intercept) ** 2 for x, y in points) / n)
    return slope, intercept, rmse


def last_known_rates(case):
    n = len(case["trophies"])
    rates = [None] * n
    for read in reversed(case["history"]):
        values = read.get("rates") or []
        for i in range(n):
            if rates[i] is None and i < len(values) and values[i] is not None:
                rates[i] = float(values[i])
        if all(r is not None for r in rates):
            break
    return rates


def refresh_points(case):
    out = []
    previous = None
    for read in case["history"]:
        rates = read.get("rates") or []
        if previous is not None:
            changed = read.get("players_tracked") != previous.get("players_tracked")
            old_rates = previous.get("rates") or []
            if not changed and len(rates) == len(old_rates):
                changed = any(a != b for a, b in zip(rates, old_rates))
            elif not changed:
                changed = True
            if not changed:
                continue
        out.append(read)
        previous = read
    return out


def inv_logit(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return 100.0 * exp_value / (1.0 + exp_value)


def make_quantiles(median, sigma, clamp, logit, round4, z_values):
    center = logit(median)
    qs = sorted(clamp(inv_logit(center + z * sigma)) for z in z_values)
    return [round4(q) for q in qs]


def flat_branch(case, clamp, logit, round4, z_values):
    history = case["history"]
    first, last = history[0], history[-1]
    p1 = float(last.get("players_tracked") or 0.0)
    E1 = float(last.get("trophies_earned") or 0.0)
    last_rates = last_known_rates(case)
    h1 = max(0.1, hours_between(first.get("at", first.get("slot")), last.get("at", last.get("slot"))))
    h2 = max(0.1, h1 + max(0.0, case.get("horizon_minutes", 0)) / 60.0)
    ratio = h2 / h1
    grown_players = ratio ** 1.21
    grown_earned = ratio ** 0.55
    if p1 > 0 and E1 > p1:
        m = ((E1 * grown_earned - p1 * grown_players) / (E1 - p1)) / grown_players
    else:
        m = 0.22
    m = min(1.5, max(0.05, m))
    counts = [max(0.01, (rate or 0.0) / 100.0 * p1) for rate in last_rates]
    non100 = [i for i, rate in enumerate(last_rates) if rate is None or rate < 99.5]
    if non100:
        mean_count = sum(counts[i] for i in non100) / len(non100)
    else:
        mean_count = 1.0
    mean_count = max(mean_count, 0.01)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last_rates[i] if i < len(last_rates) else None
        if rate is not None and rate >= 99.5:
            rows.append({"id": trophy["id"], "q": [100] * len(z_values)})
            continue
        shape = (counts[i] / mean_count) ** 0.25 if i < len(counts) else 1.0
        shape = min(1.6, max(0.6, shape))
        base = float(rate) if rate is not None else 0.0
        median = min(99.99, max(0.01, base * m * shape))
        rows.append({
            "id": trophy["id"],
            "q": make_quantiles(median, 0.55, clamp, logit, round4, z_values),
        })
    return rows


def nonflat_branch(case, clamp, logit, round4, z_values):
    points = refresh_points(case)
    first = points[0]
    first_at = first.get("at", first.get("slot"))
    target_at = case.get("target", first_at)
    h_target = max(0.0, hours_between(first_at, target_at))
    h_last = max(0.0, hours_between(first_at, points[-1].get("at", points[-1].get("slot"))))
    if h_target <= h_last:
        h_target = h_last + max(0.1, (case.get("horizon_minutes", 0) or 0) / 60.0)
    x_target = math.log1p(h_target)

    player_points = []
    last_players = 1.0
    for read in points:
        p = float(read.get("players_tracked") or 0.0)
        if p <= 0:
            continue
        last_players = p
        h = max(0.0, hours_between(first_at, read.get("at", read.get("slot"))))
        player_points.append((math.log1p(h), math.log(p)))
    if not player_points:
        player_points = [(0.0, 0.0)]
        last_players = 1.0
    player_slope, player_intercept, _ = ols(player_points)
    player_slope = max(0.0, player_slope)
    players_target = math.exp(player_intercept + player_slope * x_target)
    if h_last > 0:
        ceiling = 0.7 + 3.0 * x_target / math.log1p(h_last)
    else:
        ceiling = 3.0
    ceiling = min(5.0, max(1.0, ceiling))
    players_target = min(ceiling * last_players, max(0.7 * last_players, players_target))
    players_target = max(players_target, 0.01)

    last_rates = last_known_rates(case)
    residuals = {}
    fits = []
    families = [classify(t, i) for i, t in enumerate(case["trophies"])]
    for i, trophy in enumerate(case["trophies"]):
        family = families[i]
        prior = FAMILY_PRIOR[family]
        earned = []
        for read in points:
            rates = read.get("rates") or []
            p = float(read.get("players_tracked") or 0.0)
            if i >= len(rates) or rates[i] is None or p <= 0:
                continue
            count = clamp(float(rates[i])) / 100.0 * p
            if count < 1.0:
                continue
            h = max(0.0, hours_between(first_at, read.get("at", read.get("slot"))))
            earned.append((math.log1p(h), math.log(count)))
        if not earned:
            fits.append(None)
            continue
        raw_slope, raw_intercept, rmse = ols(earned)
        residuals.setdefault(family, []).append(rmse ** 2)
        weight = 0.3 * min(1.0, 3.0 / len(earned))
        slope = (1.0 - weight) * raw_slope + weight * prior
        if family == "online" and slope > prior:
            slope = prior + 0.85 * (slope - prior)
        slope = min(1.6, max(0.0, slope))
        cx = sum(x for x, _ in earned) / len(earned)
        cy = sum(y for _, y in earned) / len(earned)
        rx, ry = earned[-1]
        center_intercept = cy - slope * cx
        recent_intercept = ry - slope * rx
        anchor_weight = 0.1 + 0.1 * min(1.0, len(earned) / 5.0)
        intercept = (1.0 - anchor_weight) * center_intercept + anchor_weight * recent_intercept
        fits.append((slope, intercept))

    horizon_days = max(0.0, case.get("horizon_minutes", 0)) / 1440.0
    floor_sigma = 0.18 * math.sqrt(horizon_days) + 0.15
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last_rates[i] if i < len(last_rates) else None
        if rate is not None and rate >= 99.5:
            rows.append({"id": trophy["id"], "q": [100] * len(z_values)})
            continue
        fit = fits[i]
        if fit is None:
            base = float(rate) if rate is not None else 0.0
            median = clamp(base)
        else:
            count = math.exp(fit[0] * x_target + fit[1])
            median = clamp(100.0 * count / players_target)
        values = residuals.get(families[i]) or [floor_sigma ** 2]
        family_rmse = math.sqrt(sum(values) / len(values))
        sigma = min(1.5, max(family_rmse, floor_sigma))
        rows.append({
            "id": trophy["id"],
            "q": make_quantiles(median, sigma, clamp, logit, round4, z_values),
        })
    return rows


def forecast_case(case, clamp, logit, round4, z_values):
    points = refresh_points(case)
    if len(points) <= 1:
        return flat_branch(case, clamp, logit, round4, z_values)
    return nonflat_branch(case, clamp, logit, round4, z_values)
