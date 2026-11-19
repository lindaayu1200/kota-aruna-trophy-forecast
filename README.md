# kota-aruna-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lindaayu1200/kota-aruna-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lindaayu1200/kota-aruna-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A program/v1 entry for make.fun's **gta6-trophy-rarity-forecast-01**: from the house's recorded reads of Exophase's Grand Theft Auto VI PS5 trophy page, forecast each trophy's earn rate at the case's target as five quantiles (5/25/50/75/95%), scored by a pinball loss on the logit scale.

## Run it

```sh
python3 -P -s main.py
```

The grader runs `python3 -P -s /bot/main.py` (Python 3.13, standard library only, `PYTHONHASHSEED=0`) once per case. Locally, point the entry at directories of your choosing:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=tools/fixtures MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py
```

Each `<id>.in` becomes `<id>.out`: `{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]},...]}`, one row per trophy, five finite non-decreasing percentages in `[0,100]`, under 65,536 bytes. There is no network, no randomness and no wall-clock use, so two runs write byte-identical files (measured at about 0.03–0.07 s for three cases, well inside the 8 s cap).

## Refresh-aware model

Exophase refreshes a game's counts hours apart; between refreshes every rate and `players_tracked` are constant. `model.py` works on **refresh points**: the first read plus every read whose tracked count or rates changed.

1. **Players.** OLS of `ln(players_tracked)` on `ln(1 + hours since the first read)` over the refresh points, slope clamped non-negative, extrapolated to the target and bounded near the last observed pool (at most about 5× it).
2. **Earned counts.** For each trophy, `c_i = rate_i/100 × players_tracked` at every refresh point (rates clamped, nulls and counts below 1 skipped). OLS of `ln(c_i)` on `ln(1+h)`, slope shrunk toward a family prior with weight `0.3 × min(1, 3/n)`: platinum 0.0, story 0.85, collectible 0.55, stunt 0.45, online 0.30 (plus "other" 0.50). The family comes from the tier and keywords in the name/description (Online, Multiplayer, Crew, app; stunt/trick/race; collect/grind; chapter/mission/strand); online uplift over its prior is attenuated to 0.85 and slopes are clamped to `[0,1.6]`. The fitted line is re-anchored between its centroid and the most recent refresh point with weight 0.1–0.2 on the recent point.
3. **Non-flat median.** `100 × exp(slope × ln(1+h_target) + intercept) / players_target`, clamped; a trophy at ≥99.5% stays at 100.
4. **Flat history (no refresh).** The launch-night prior: with `t1` the hours from the first read to the last read and `t2 = t1 + horizon`, `p2/p1 = (t2/t1)^1.21` and `E2/E1 = (t2/t1)^0.55`. Because the one 100% trophy absorbs 100 points of rate mass, the other trophies' mass multiplier is `((E2−p2)/(E1−p1))·(p1/p2)`, clamped to `[0.05,1.5]` (about 0.21–0.30 on the practice cases, near the house's observed 0.221). Each median is `rate × multiplier × shape_i`, where `shape_i = (c_i / mean_non100_c)^0.25` clamped to `[0.6,1.6]`.
5. **Spread.** Logit-normal quantiles at `z = -1.645, -0.674, 0, 0.674, 1.645`. On a non-flat history the width is `max(family residual RMSE, 0.18×√horizon_days + 0.15)`, capped at 1.5; on a flat history it is 0.55 logit units. Quantiles are sorted, clamped and rounded half-up to 4 decimals by the starter's helpers.

## Validation

`python3 tools/validate.py` copies `tools/fixtures/p00.in`, `p01.in` and a constructed multi-refresh case (including a null read) into a temp directory, runs the entry twice, checks the exact output contract for all three, and compares the two runs byte for byte.

The scorer reproduces the competition's `2·max(τ·d, (τ−1)·d)` with `d = logit(y) − logit(q_τ)` on rates clamped to `[0.01,99.99]`. It scores the model and the house persistence forecast against three answers:

| Answer | What it assumes | Model loss | Persistence loss |
| --- | --- | ---: | ---: |
| A | every rate unchanged | 0.983646 | 0.024854 |
| B | platinum 100, all other rates ×0.75 | 0.971074 | 0.445242 |
| C | aggregate-consistent launch refresh: platinum 0.10%, the 100% trophy at 100%, other earned counts grown as `c_i^0.25` to 4,276 earned / 3,868 tracked | **0.173794** | 1.379010 |

Losses are the mean over the two practice cases. On C the model's typical miss is 0.04 percentage points and its 90% coverage is 1.000, against the house's reported persistence loss of about 1.51 on the real practice. A and B are diagnostic: persistence is nearly optimal when rates really do stay put, so the entry is deliberately tuned for refresh answers like C. The validator asserts that the model's loss on **each** answer is below persistence's refresh-answer loss (1.379010), that the model beats persistence directly on C, and that its loss on C is below 1.2. All assertions pass.

## License

MIT — see [LICENSE](LICENSE).
