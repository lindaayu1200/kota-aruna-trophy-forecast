"""Refresh-aware trophy-rarity forecast for gta6-trophy-rarity-forecast-01.

The page Exophase shows is piecewise constant: between its own refreshes every rate
and the players-tracked count are frozen, and at a refresh both jump.  So the model
works on *refresh points*, not on the raw reads:

  * ln(players_tracked) ~ a + b * ln(1 + hours since the first read),
  * ln(earned count of trophy i) ~ a_i + s_i * ln(1 + hours),

both by ordinary least squares over the refresh points.  The target rate is the
fitted earned count divided by the fitted player count.  Per-trophy slopes are
shrunk toward a tier/keyword prior, and the quantiles are a logit-normal band
whose width is the larger of the group fit residual and a horizon term.  A history
with no refresh at all uses the launch-refresh prior: the non-100% rate mass is
scaled by the fitted players/earned growth, with a c_i**0.25 per-trophy shape and
a fixed logit spread.

Pure float math, no randomness and no wall clock: identical input gives identical
bytes.  Standard library only.
"""

import math
from datetime import datetime, timezone

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Prior slopes on ln(1 + hours) for ln(earned count), by trophy family.
PRIORS = {
    "platinum": 0.0,
    "story": 0.85,
    "collectible": 0.55,
    "stunt": 0.45,
    "online": 0.30,
}
PRIOR_WEIGHT = 0.3       # weight on the prior with the minimum useful data
PRIOR_PIVOT = 3.0        # extra refresh points reduce that weight
ONLINE_ATTENUATION = 0.85
PLAYER_EXPONENT = 1.21
EARNED_EXPONENT = 0.55
SHAPE_POWER = 0.25
MIN_SHAPE, MAX_SHAPE = 0.6, 1.6
MIN_MASS_MULT, MAX_MASS_MULT = 0.05, 1.5
FLAT_SIGMA = 0.55
HIGH_RATE = 99.5
MIN_RATE, MAX_RATE = 0.01, 99.99
MAX_GROUP_SIGMA = 1.5     # a very noisy fit may not swallow the whole band


def clamp(x: float) -> float:
    return min(MAX_RATE, max(MIN_RATE, x))


def logit(pct: float) -> float:
    x = clamp(pct)
    return math.log(x / (100 - x))


def inv_logit(value: float) -> float:
    return 100.0 / (1.0 + math.exp(-value))


def round4(x: float) -> float | int:
    """Half-up to 4 decimals, like JavaScript's Math.round(x * 10000) / 10000."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    value = text[:-1] + "+00:00" if text.endswith("Z") else text
    moment = datetime.fromisoformat(value)
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def hours_since(origin: datetime, text: str) -> float:
    return max(0.0, (parse_time(text) - origin).total_seconds() / 3600.0)


def family(trophy: dict) -> str:
    """Tier + keyword classification: the prior slope group of one trophy."""
    tier = str(trophy.get("tier", "")).lower()
    if tier == "platinum":
        return "platinum"
    text = (str(trophy.get("name", "")) + " " + str(trophy.get("description", ""))).lower()
    if any(k in text for k in ("online", "multiplayer", "crew", "ppv", "replay", "co-op", "coop")):
        return "online"
    if any(k in text for k in ("stunt", "trick", "drift", "jump", "wheelie", "stoppie", "race")):
        return "stunt"
    if any(k in text for k in ("collect", "discover", "purchase", "all locations", "every location",
                               "100%", "gold medal", "earn $", "spend $", "attain", "garage properties")):
        return "collectible"
    if any(k in text for k in ("chapter", "final mission", "mission strand", "heist", "complete the",
                               "complete final", "story")) or trophy.get("secret"):
        return "story"
    if tier in ("gold", "silver"):
        return "story"
    return "collectible"


def ols(points: list[tuple[float, float]]) -> tuple[float, float, float] | None:
    """Slope, intercept and RMSE of y = slope*x + intercept; None if degenerate."""
    n = len(points)
    if n < 2:
        return None
    sx = math.fsum(x for x, _ in points)
    sy = math.fsum(y for _, y in points)
    sxx = math.fsum(x * x for x, _ in points)
    sxy = math.fsum(x * y for x, y in points)
    denominator = n * sxx - sx * sx
    if denominator <= 1e-12:
        return None
    slope = (n * sxy - sx * sy) / denominator
    intercept = (sy - slope * sx) / n
    rmse = math.sqrt(math.fsum((y - (slope * x + intercept)) ** 2 for x, y in points) / n)
    return slope, intercept, rmse


def refresh_indices(history: list[dict]) -> list[int]:
    """The first read plus every read at which tracked players or any rate changed."""
    indices = [0]
    for i in range(1, len(history)):
        before, after = history[i - 1], history[i]
        changed = (before.get("players_tracked") != after.get("players_tracked")
                   or list(before.get("rates") or []) != list(after.get("rates") or []))
        if changed:
            indices.append(i)
    return indices


def last_rate(history: list[dict], index: int) -> float:
    for read in reversed(history):
        rates = read.get("rates") or []
        if index < len(rates) and rates[index] is not None:
            return clamp(float(rates[index]))
    return MIN_RATE


def flat_launch_shapes(rates: list[float], players: float) -> list[float]:
    """Per-trophy variation of the launch-refresh multiplier, normalised by rate mass."""
    counts = [rate / 100.0 * players for rate in rates]
    indices = [i for i, rate in enumerate(rates) if rate < HIGH_RATE]
    if not indices:
        return [1.0] * len(rates)
    mean_count = math.fsum(counts[i] for i in indices) / len(indices)
    shapes = [1.0] * len(rates)
    for i in indices:
        ratio = counts[i] / mean_count if mean_count > 0 else 1.0
        shapes[i] = min(MAX_SHAPE, max(MIN_SHAPE, ratio ** SHAPE_POWER))
    mass = math.fsum(rates[i] for i in indices)
    weighted = math.fsum(rates[i] * shapes[i] for i in indices)
    if mass > 0 and weighted > 0:
        factor = mass / weighted
        shapes = [shape * factor for shape in shapes]
    return shapes


def forecast_case(case: dict) -> dict:
    history = case["history"]
    trophies = case["trophies"]
    first_time = parse_time(history[0].get("slot") or history[0]["at"])
    target_time = parse_time(case["target"]) if case.get("target") else parse_time(
        history[-1].get("slot") or history[-1]["at"])
    hours_target = max(0.0, (target_time - first_time).total_seconds() / 3600.0)
    x_target = math.log1p(hours_target)
    horizon_days = max(0.0, float(case["horizon_minutes"]) / 1440.0)
    base_sigma = 0.18 * math.sqrt(horizon_days) + 0.15

    indices = refresh_indices(history)
    refresh_x = [math.log1p(hours_since(first_time, history[j].get("slot") or history[j]["at"]))
                 for j in indices]
    flat = len(indices) < 2

    flat_multiplier = 1.0
    flat_shapes: list[float] = []
    if flat:
        last_rates = [last_rate(history, i) for i in range(len(trophies))]
        players1 = max(1.0, float(history[-1]["players_tracked"]))
        earned1 = max(1.0, float(history[-1]["trophies_earned"]))
        t1 = max(0.1, hours_since(first_time, history[-1].get("slot") or history[-1]["at"]))
        t2 = max(0.1, hours_target)
        growth = t2 / t1
        players2 = players1 * growth ** PLAYER_EXPONENT
        earned2 = earned1 * growth ** EARNED_EXPONENT
        excess1 = max(1e-9, earned1 - players1)
        flat_multiplier = ((earned2 - players2) / excess1) * (players1 / players2)
        flat_multiplier = min(MAX_MASS_MULT, max(MIN_MASS_MULT, flat_multiplier))
        flat_shapes = flat_launch_shapes(last_rates, players1)

    # Player-count fit on the refresh points.
    ln_players_last = math.log(max(1.0, float(history[-1]["players_tracked"])))
    ln_players_target = ln_players_last
    if not flat:
        fit = ols([(x, math.log(max(1.0, float(history[j]["players_tracked"]))))
                   for x, j in zip(refresh_x, indices)])
        if fit is not None:
            slope, intercept, _ = fit
            # Revision 2: the tracked pool of a just-launched game does not shrink;
            # a negative fitted slope is an artefact of the irregular refresh grid.
            slope = max(0.0, slope)
            ln_players_target = intercept + slope * x_target
            # Guard the extrapolation: the tracked pool cannot explode or collapse
            # arbitrarily over a horizon of a few days.
            ln_players_target = min(ln_players_last + 2.0,
                                    max(ln_players_last - 0.05, ln_players_target))

    # Fit every trophy on earned counts at the refresh points; pool residuals by family.
    residuals: dict[str, list[float]] = {key: [] for key in PRIORS}
    slopes: list[tuple[float, float, str]] = []
    for i, trophy in enumerate(trophies):
        group = family(trophy)
        prior = PRIORS[group]
        points = []
        for x, j in zip(refresh_x, indices):
            read = history[j]
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            earned = clamp(float(rates[i])) / 100.0 * max(1.0, float(read["players_tracked"]))
            # Revision 3: 0.00% is below the page's useful resolution at small
            # player counts; log(0.01% * players) would dominate the fit as noise.
            if earned >= 1.0:
                points.append((x, math.log(earned)))
        fit = ols(points) if not flat else None
        if fit is None:
            slopes.append((prior, 0.0, group))
            continue
        fit_slope, _, _ = fit
        n = len(points)
        weight = PRIOR_WEIGHT * PRIOR_PIVOT / (PRIOR_PIVOT + max(0, n - 2))
        slope = (1.0 - weight) * fit_slope + weight * prior
        if group == "online" and slope > prior:
            slope = prior + ONLINE_ATTENUATION * (slope - prior)
        # Revision 2: an earned count cannot fall.  A negative slope is rounding
        # noise on the small counts of the first refreshes, not a real decline.
        slope = min(1.6, max(0.0, slope))
        # Re-anchor the shrunken line at the data centroid, then lean a little
        # toward the most recent refresh: revision 3 keeps the line passing near
        # the state the forecast starts from when the early counts are tiny.
        mean_x = math.fsum(x for x, _ in points) / n
        mean_y = math.fsum(y for _, y in points) / n
        centroid_intercept = mean_y - slope * mean_x
        last_x, last_y = points[-1]
        recent_intercept = last_y - slope * last_x
        anchor = 0.20 if n <= 4 else 0.10
        intercept = (1.0 - anchor) * centroid_intercept + anchor * recent_intercept
        residuals[group].extend(y - (slope * x + intercept) for x, y in points)
        slopes.append((slope, intercept, group))

    rmse = {key: (math.sqrt(math.fsum(v * v for v in values) / len(values))
                  if values and math.isfinite(math.fsum(v * v for v in values)) else 0.0)
            for key, values in residuals.items()}

    rows = []
    for i, trophy in enumerate(trophies):
        rate = last_rate(history, i)
        group = family(trophy)
        if flat:
            if rate >= HIGH_RATE:
                rows.append({"id": trophy["id"], "q": [100.0] * len(LEVELS)})
                continue
            median = rate * flat_multiplier * flat_shapes[i]
            sigma = FLAT_SIGMA
        else:
            if trophy.get("tier") == "platinum" or rate >= HIGH_RATE:
                rows.append({"id": trophy["id"], "q": [100.0] * len(LEVELS)})
                continue
            slope, intercept, _ = slopes[i]
            median = 100.0 * math.exp(slope * x_target + intercept - ln_players_target)
            median = clamp(median)
            sigma = max(base_sigma, min(MAX_GROUP_SIGMA, rmse.get(group, 0.0)))
        centre = logit(median)
        quantiles = sorted(clamp(inv_logit(centre + z * sigma)) for z in Z)
        # No floor at the last rate: when the fit says a rate falls while earned
        # counts grow, the fit is the forecast.
        rows.append({"id": trophy["id"], "q": [round4(q) for q in quantiles]})
    return {"trophies": rows}
