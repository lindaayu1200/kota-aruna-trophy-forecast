# Sharpen the week-one trophy rarity forecast (round 3)

Competition: gta6-trophy-rarity-forecast-01 on make.fun. Grading truth: each trophy's earn
rate on Exophase's GTA VI PSN trophy page at the case target, scored by mean pinball loss
on the logit scale over 5 quantiles (tau = 0.05, 0.25, 0.5, 0.75, 0.95) per trophy. Lower
is better. Rank column = `loss` only; typical-miss and coverage are informational.

## Real-world observations (ground truth measured today, use to calibrate)
- Launch night 2026-11-19: page showed 638 players tracked until ~04:00, then a refresh at
  ~04:30 showed 3868 players and LOWER rates for every trophy but the 100% one.
- A later refresh (~14:00) showed 14028 players; rates as of that refresh (identical at
  14:16, 14:24 and again fetched 01:50 on 11-20 — the page had NOT refreshed for 11.5+ h):
  see `tools/fixtures/exophase_live_1120.json` (55 trophies, in id order, same order as the
  case's trophy ids) and `tools/fixtures/snapshots.ndjson` (7 reads).
- So Exophase refreshes the game's counts on its own slow schedule (observed gaps: a few
  hours at launch, then >11 h), the page can be STALE for many hours at the answer read,
  and between refreshes the players-tracked pool can jump ~3-4x (tracker expansion), which
  DILUTES rates when the pool grows faster than earned counts.
- The FINAL case f00 (history cut 2026-11-21T11:30Z, target 2026-11-26T00:00Z, horizon
  6510 min) will contain reads every 10 minutes with maybe 2-5 refresh points spread over
  ~2.5 days. The answer read (first valid read in the hour from the target) shows rates from
  Exophase's LAST REFRESH BEFORE the target — expected staleness at target: hours (could be
  ~0 to ~24+). Budget for a typical staleness (e.g., subtract roughly the mean observed
  refresh gap / 2 to the full gap when extrapolating), not for a fresh read.

## What to build (rewrite model.py; keep main.py's I/O contract unchanged)
1. Parse the history into refresh points: a read whose players_tracked or any rate differs
   from the previous read is a refresh; identical reads are stale (Exophase page unchanged).
2. Per-trophy earned counts c_i = rate/100 * players at each refresh point (monotone
   non-decreasing; respect rate rounding to 2 decimals).
3. Median forecast: accrue earned counts from the last refresh to the estimated answer time
   using a decaying accrual rate per trophy (power-law decay in hours-since-launch, exponent
   shared across trophies, estimated from observed refresh-interval accruals when there are
   >= 2 intervals), then divide by the extrapolated players-tracked pool at the answer time
   (pool growth fit in log-space over refresh points; a refreshed pool can jump, so blend a
   power-law fit with the last observed per-interval growth rate).
4. Structure priors: classify each trophy (platinum / story-chapter chain / grind /
   collectible / stunt / online-multiplayer / other) from name+description+tier+secret like
   the current model does, and use the class to temper the accrual decay (online and grind
   trophies accrue slower and keep rising later; story trophies plateau fast).
5. Quantile spread: per-trophy sigma in logit space, driven by (a) relative accrual
   uncertainty across observed intervals, (b) Poisson count noise ~ 1/sqrt(c), (c) staleness
   uncertainty, (d) pool-expansion dilution uncertainty (the 638->3868->14028 jumps show
   dilution risk), all grown with remaining time; widen enough for honest 90% coverage —
   the house baseline's 0.5*sqrt(days) random walk is too narrow in places, too wide in
   others. 100% trophies stay 100.
6. Blend/ensemble where the data is thin: shrink the extrapolated median toward a
   structure-driven prior (base-rate family medians) when a trophy has few refresh points
   or a tiny count; never let a trophy's median move DOWN unless its last refresh dropped it.
7. Fallbacks: flat history (like practice p00/p01) -> a sensible launch-refresh prior (do
   NOT just carry the rate forward — the house baseline does that and it was the weak spot);
   sparse history, single refresh, missing rates -> degrade gracefully to the curve/flat
   fallbacks. Deterministic: no randomness, no dict-iteration order dependence.
8. Constraints: Python 3.13 stdlib only, < 8 s / 1 CPU / 1 GB per case, output rows exactly
   {"id", "q": [q05,q25,q50,q75,q95]} finite non-decreasing in [0,100], file <= 65536 bytes,
   rounding to 4 decimals like the existing main.py round4.

## Local verification (no remote grader for this competition)
- `python3 tools/validate.py` style checks: run main.py over the kit practice cases
  (tools/fixtures/p00.in and p01.in) and the synthetic
  final-like case tools/fixtures/f00.in, then run
  tools/fixtures/verify.py <input_dir> <output_dir> for format.
  A previous backtest harness is at tools/fixtures/backtest.py — reuse or rewrite it.
- A synthetic sanity test: build a case with known ground truth (simulate counts with a
  logistic accrual + pool jumps, 2-3 refreshes, 4.8-day horizon) and check your forecast's
  pinball loss on the logit scale against (a) the house baseline formula
  q_tau = 100/(1+exp(-(logit(r)+z*0.5*sqrt(days)))) and (b) a logit-linear trend fit —
  your model should beat both on that simulation. Write the harness as
  tools/backtest_local.py and iterate until it wins there. Keep the simulation honest
  (decay, dilution, staleness), but do not overfit to one seed; try 3-5 seeds.
- Also run on p00/p01 (flat practice) and confirm valid output and reasonable wide-ish
  quantiles (their answers are the refreshed page, which no flat-history model can nail;
  practice pays nothing, so don't tune on it — just don't crash or freeze).

Do not touch README badges/metadata. Commit nothing; leave changes in the working tree.
Report at the end: what the model does now, backtest losses vs the two baselines, and the
exact commands you ran.
