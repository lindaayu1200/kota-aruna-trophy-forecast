"""Refresh-aware quantile forecast for the GTA VI trophy case."""

# Forecast notes — refresh-aware accrual
# --------------------------------------
# Exophase publishes new numbers only at sporadic refresh instants; the reads
# between two refreshes merely repeat the same snapshot. We therefore compress
# the history to those instants and treat each one as an observation of two
# things: the tracked-player pool and the implicit earned count of every
# trophy (rate times pool, floored at one because rates are rounded).
# Earned counts are forced to be monotone; a genuine drop in a rate marks the
# trophy as reset, which relaxes a lower bound used when blending medians.
# For the median we extrapolate counts forward with a power law in hours since
# launch. Its decay exponent is estimated jointly from the observed refresh
# intervals, then nudged per trophy by a structure class: story chains cool
# off quickly, while grind, collectible, stunt and online trophies keep
# accruing for longer. A log-linear fit of each trophy's own history is blended
# with the accrual estimate, and sparse trophies shrink toward a class-specific
# base rate (the prior) so tiny or fresh counts do not produce wild rates.
# Dividing the projected count by a projected pool accounts for tracker
# expansion: the pool fit mixes a global log-slope with the most recent
# interval growth and is bounded near the last observed value.
# Staleness matters: the answer read can be hours behind the last refresh, so
# the target horizon is pulled back by a fraction of the typical refresh gap.
# Quantiles are logit-normal around the median. The logit sigma combines four
# variances: Poisson noise in small counts, scatter of the interval decay
# estimates, uncertainty about how stale the snapshot is, and pool-expansion
# dilution. It grows with remaining time and is scaled by family-specific
# width factors, then clamped. Trophies already at 100% stay pinned there.
# Everything is arithmetic on the input case, with no clocks or randomness.

import math
from datetime import datetime, timezone

FAMILY_PRIOR = {
    "platinum": 0.55,
    "story": 1.15,
    "grind": 0.55,
    "collectible": 0.75,
    "stunt": 0.85,
    "online": 0.50,
    "other": 0.75,
}

FAMILY_DECAY = {
    "platinum": 1.00,
    "story": 1.10,
    "grind": 0.85,
    "collectible": 0.95,
    "stunt": 1.05,
    "online": 0.80,
    "other": 1.00,
}

FAMILY_BASE = {
    "platinum": 0.65,
    "story": 1.55,
    "grind": 0.80,
    "collectible": 1.05,
    "stunt": 0.95,
    "online": 0.80,
    "other": 1.00,
}

FAMILY_SPREAD = {
    "platinum": 1.10,
    "story": 0.90,
    "grind": 1.10,
    "collectible": 1.00,
    "stunt": 1.05,
    "online": 1.15,
    "other": 1.00,
}

ONLINE_WORDS = (
    "online", "multiplayer", "crew", "co-op", "coop", "scooter bros", "partner",
)
GRIND_WORDS = (
    "250 stunts", "100 locations", "100 vehicles", "50 challenges", "20 different races",
    "20 races", "$250,000", "$100,000", "all shops", "all garage", "all shooting range",
    "every location", "every spot", "achieve 100%", "attain 100%", "collectible card",
    "card sets", "purchase all", "discover 100", "meditate at every", "explore all",
    "win 20", "complete 50", "complete 250", "earn $", "spend $", "rob 5", "8 mission strands",
)
STUNT_WORDS = (
    "stunt", "trick", "race", "drift", "jump", "drive-by", "bowling", "bowl", "fishing",
    "arcade", "challenge", "medal",
)
COLLECT_WORDS = (
    "collect", "discover", "purchase", "unlock", "find card", "sets", "properties",
    "all shops", "all garage", "100%",
)
STORY_WORDS = (
    "chapter", "mission", "story", "strand", "heist", "breakfast run", "child's play",
    "headliners", "vcia takedown", "operation savage glory", "gone fishing",
)


def hours_between(first, second):
    a, b = parse_time(first), parse_time(second)
    if a is None or b is None:
        return 0.0
    return (b - a) / 3600.0


def parse_time(value):
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.timestamp()
    except (ValueError, TypeError):
        return None


def classify(trophy, index):
    text = f"{trophy.get('name', '')} {trophy.get('description', '')}".lower()
    tier = str(trophy.get("tier", "")).lower()
    if tier == "platinum" or (index == 0 and tier == ""):
        return "platinum"
    if any(word in text for word in ONLINE_WORDS):
        return "online"
    if any(word in text for word in STORY_WORDS):
        return "story"
    if any(word in text for word in GRIND_WORDS):
        return "grind"
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


def launch_hours(case, reads):
    game = case.get("game") or {}
    anchor = parse_time(game.get("release_date")) if game.get("release_date") else None
    if anchor is None:
        for read in reads:
            anchor = parse_time(read.get("at", read.get("slot")))
            if anchor is not None:
                break
    if anchor is None:
        anchor = parse_time(case.get("at")) or 0.0
    return anchor


def read_hours(read, anchor):
    moment = parse_time(read.get("at", read.get("slot")))
    if moment is None:
        moment = parse_time(read.get("slot"))
    if moment is None:
        return 0.0
    return max(0.0, (moment - anchor) / 3600.0)


def observed_counts(case, reads, hours, n):
    samples = [[] for _ in range(n)]
    dropped = [False] * n
    last_rates = [None] * n
    for k, read in enumerate(reads):
        values = read.get("rates") or []
        pool = float(read.get("players_tracked") or 0.0)
        for i in range(n):
            if i >= len(values) or values[i] is None or pool <= 0:
                continue
            rate = float(values[i])
            count = min(99.99, max(0.0, rate)) / 100.0 * pool
            if samples[i]:
                prior = last_rates[i]
                tolerance = 0.005 / 100.0 * pool + 0.5
                if (prior is not None and rate < prior - 0.005) or count < samples[i][-1][1] - tolerance:
                    dropped[i] = True
                count = max(count, samples[i][-1][1])
            last_rates[i] = rate
            samples[i].append((hours[k], count))
    return samples, dropped, last_rates


def estimate_decay(samples):
    grouped = []
    for points in samples:
        local = []
        for k in range(1, len(points)):
            dh = points[k][0] - points[k - 1][0]
            if dh <= 1e-6:
                continue
            speed = max(0.0, points[k][1] - points[k - 1][1]) / dh
            if speed <= 1e-9:
                continue
            mid = 0.5 * (points[k][0] + points[k - 1][0]) + 1.0
            local.append((math.log(mid), math.log(speed)))
        if local:
            grouped.append(local)
    if len(grouped) < 2:
        return 1.10, 0.28
    mean_log = [sum(y for _, y in g) / len(g) for g in grouped]
    points = []
    for g, mean in zip(grouped, mean_log):
        for x, y in g:
            points.append((x, y - mean))
    distinct = len(set(round(x, 3) for x, _ in points))
    if len(points) < 3 or distinct < 2:
        return 1.10, 0.28
    slope, _, rmse = ols(points)
    alpha = 0.4 * min(1.6, max(0.0, -slope)) + 0.6 * 1.10
    return alpha, min(0.30, max(0.06, rmse))


def recent_slope(points, window=4):
    selected = points[-window:]
    if len(selected) < 2:
        selected = points
    return ols(selected)[0]


def pool_forecast(reads, hours, h_answer):
    pool_points = []
    for k, read in enumerate(reads):
        pool = float(read.get("players_tracked") or 0.0)
        if pool > 0:
            h = max(0.0, hours[k])
            pool_points.append((math.log1p(h), math.log(pool), h, pool))
    if not pool_points:
        return 1.0, 0.5, 0.0
    h_last, pool_last = pool_points[-1][2], pool_points[-1][3]
    u_points = [(math.log1p(h), y) for _, y, h, _ in pool_points]
    global_slope = min(1.6, max(0.0, ols(u_points)[0]))
    recent = min(1.6, max(0.0, recent_slope(u_points)))
    blended = min(1.3, 0.4 * global_slope + 0.6 * recent)
    if len(pool_points) >= 2:
        dh = h_last - pool_points[-2][2]
        hourly = (pool_points[-1][1] - pool_points[-2][1]) / dh if dh > 1e-6 else 0.0
    else:
        hourly = 0.0
    horizon = max(0.0, h_answer - h_last)
    ratio_power = ((h_answer + 1.0) / (h_last + 1.0)) ** blended
    hourly = min(0.08, max(0.0, hourly))
    ratio_hourly = math.exp(hourly * min(96.0, horizon))
    ratio = 0.7 * ratio_power + 0.3 * ratio_hourly
    ratio = min(5.0, max(0.85, ratio))
    return max(1.0, pool_last * ratio), blended, hourly


def accrual_integral(h_start, h_end, alpha, mid):
    if h_end <= h_start:
        return 0.0
    if abs(alpha - 1.0) < 1e-6:
        return mid * math.log((h_end + 1.0) / (h_start + 1.0))
    return (mid ** alpha) * (
        (h_end + 1.0) ** (1.0 - alpha) - (h_start + 1.0) ** (1.0 - alpha)
    ) / (1.0 - alpha)


def last_speed(points, h_last):
    target = h_last - max(6.0, 0.10 * h_last)
    anchor = points[0]
    for point in points:
        if point[0] <= target:
            anchor = point
        else:
            break
    dh = h_last - anchor[0]
    speed = max(0.0, points[-1][1] - anchor[1]) / dh if dh > 1e-6 else 0.0
    mid = 0.5 * (h_last + anchor[0]) + 1.0
    return speed, mid, anchor[0]


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
        multiplier = ((E1 * grown_earned - p1 * grown_players) / (E1 - p1)) / grown_players
    else:
        multiplier = 0.22
    multiplier = min(1.5, max(0.05, multiplier))
    counts = [max(0.01, (rate or 0.0) / 100.0 * p1) for rate in last_rates]
    non100 = [i for i, rate in enumerate(last_rates) if rate is None or rate < 99.5]
    mean_count = sum(counts[i] for i in non100) / len(non100) if non100 else 1.0
    mean_count = max(mean_count, 0.01)
    days = max(0.0, case.get("horizon_minutes", 0)) / 1440.0
    sigma = max(0.55, 0.35 * math.sqrt(days))
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last_rates[i] if i < len(last_rates) else None
        if rate is not None and rate >= 99.5:
            rows.append({"id": trophy["id"], "q": [100] * len(z_values)})
            continue
        shape = (counts[i] / mean_count) ** 0.25 if i < len(counts) else 1.0
        shape = min(1.6, max(0.6, shape))
        base = float(rate) if rate is not None else 0.0
        median = min(99.99, max(0.01, base * multiplier * shape))
        rows.append({
            "id": trophy["id"],
            "q": make_quantiles(median, sigma, clamp, logit, round4, z_values),
        })
    return rows


def trend_branch(case, clamp, logit, round4, z_values, reads, hours, samples, dropped, last_rates, families):
    anchor = launch_hours(case, reads)
    target = parse_time(case.get("target"))
    if target is None:
        target = parse_time(case.get("at"))
    h_target = max(hours[-1] + 0.1, (target - anchor) / 3600.0) if target is not None else hours[-1] + 24.0
    h_last = hours[-1]
    alpha_global, accrual_scatter = estimate_decay(samples)
    refresh_gaps = [b - a for a, b in zip(hours, hours[1:]) if b - a > 0.5]
    mean_gap = sum(refresh_gaps) / len(refresh_gaps) if refresh_gaps else 12.0
    staleness = min(12.0, max(2.0, 0.5 * mean_gap))
    h_answer = max(h_last, h_target - staleness)
    pool_answer, pool_slope, pool_hourly = pool_forecast(reads, hours, h_answer)
    days = max(0.0, case.get("horizon_minutes", 0)) / 1440.0
    remaining = max(0.0, h_answer - h_last)
    time_growth = (1.0 + remaining / 240.0) ** 0.4

    valid_rates = [r for i, r in enumerate(last_rates) if r is not None and r < 99.5 and 0 < r]
    reference = sorted(valid_rates)[len(valid_rates) // 2] if valid_rates else 0.05
    reference = max(0.01, reference)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last_rates[i] if i < len(last_rates) else None
        if rate is not None and rate >= 99.5:
            rows.append({"id": trophy["id"], "q": [100] * len(z_values)})
            continue
        family = families[i]
        points = samples[i]
        log_hc = [(math.log1p(h), math.log(max(0.5, c))) for h, c in points]
        beta_recent = min(1.5, max(0.0, recent_slope(log_hc, window=2)))
        alpha = 0.35 * alpha_global + 0.65 * (1.0 - beta_recent)
        alpha = min(1.6, max(0.05, alpha * FAMILY_DECAY[family]))
        if len(points) >= 2:
            speed, mid, ref_h = last_speed(points, h_last)
            if speed <= 1e-9:
                speed = points[-1][1] / max(1.0, h_last) * 0.5
            added = speed * accrual_integral(ref_h, h_answer, alpha, mid)
            c_accrual = max(0.5, points[-1][1] + added)
            beta, _, _ = ols(log_hc)
            shrink = 0.25 * min(1.0, 3.0 / len(points))
            beta = (1.0 - shrink) * beta_recent + shrink * FAMILY_PRIOR[family]
            beta = min(1.5, max(0.0, beta))
            h_ratio = (h_answer + 1.0) / (points[-1][0] + 1.0)
            c_fit = points[-1][1] * min(4.0, max(0.85, h_ratio ** beta))
            fit_weight = 0.40 if len(points) >= 4 else 0.50
            c_est = math.exp((1.0 - fit_weight) * math.log(c_accrual) + fit_weight * math.log(max(0.5, c_fit)))
        else:
            c_est = max(0.5, (rate or 0.1) / 100.0 * (reads[-1].get("players_tracked") or 1.0))

        median_extrap = min(99.0, max(0.01, 100.0 * c_est / pool_answer))
        prior = reference * FAMILY_BASE[family]
        weight = 0.10 + 0.25 / (1.0 + len(points))
        if c_est < 8.0:
            weight += 0.15
        weight = min(0.5, weight)
        log_median = (1.0 - weight) * math.log(median_extrap) + weight * math.log(max(0.01, prior))
        if not dropped[i]:
            log_median = max(log_median, math.log(median_extrap) - 0.25)
        median = min(99.0, max(0.01, math.exp(log_median)))

        poisson = min(0.8, max(0.03, 1.0 / math.sqrt(c_est + 0.5)))
        stale = (0.06 + 0.10 * math.sqrt(staleness / 12.0)) * (1.0 + 0.3 * alpha)
        pool_uncertainty = (0.10 + 0.20 / math.sqrt(max(2.0, len(reads)))) * (1.0 + 0.3 * pool_slope)
        accrual = accrual_scatter * min(1.0, math.sqrt(1.0 + remaining / max(24.0, h_last)))
        sigma = math.sqrt(poisson ** 2 + stale ** 2 + pool_uncertainty ** 2 + accrual ** 2)
        sigma *= time_growth * FAMILY_SPREAD[family] * (0.90 + 0.10 * math.sqrt(days))
        sigma = min(1.2, max(0.15, sigma))
        rows.append({
            "id": trophy["id"],
            "q": make_quantiles(median, sigma, clamp, logit, round4, z_values),
        })
    return rows


def forecast_case(case, clamp, logit, round4, z_values):
    reads = refresh_points(case)
    if len(reads) <= 1:
        return flat_branch(case, clamp, logit, round4, z_values)
    n = len(case["trophies"])
    anchor = launch_hours(case, reads)
    hours = [read_hours(read, anchor) for read in reads]
    for k in range(1, len(hours)):
        if hours[k] <= hours[k - 1]:
            hours[k] = hours[k - 1] + 1e-6
    families = [classify(trophy, i) for i, trophy in enumerate(case["trophies"])]
    samples, dropped, last_rates = observed_counts(case, reads, hours, n)
    return trend_branch(case, clamp, logit, round4, z_values, reads, hours, samples, dropped, last_rates, families)
