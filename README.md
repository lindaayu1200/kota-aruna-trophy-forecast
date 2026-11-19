# kota-aruna-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lindaayu1200/kota-aruna-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lindaayu1200/kota-aruna-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Week-one trophy-rarity forecast for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty: a refresh-aware quantile model of Exophase's Grand Theft Auto VI PS5 trophy page. It forecasts each trophy's earn rate at the case's target as five quantiles (5/25/50/75/95%), scored by the competition's pinball loss on the logit scale.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py
```

Python 3.13, standard library only, no network, no randomness: two runs write byte-identical output.

## How it works

Exophase's page is piecewise constant: every read between two of its own refreshes has the same rates and the same players-tracked count. The model works on *refresh points* (the first read plus every read where the tracked count or any rate changed):

1. **Players.** OLS of `ln(players_tracked)` on `ln(1 + hours since the first read)` over the refresh points, extrapolated to the target, with a nonnegative slope and a bounded extrapolation.
2. **Earned counts.** Per trophy, OLS of `ln(earned count)` (rate × tracked players) on `ln(1 + hours)` over the refresh points; each slope is shrunk toward a tier/keyword family prior (platinum, story, collectible, stunt, online), online-multiplayer growth is attenuated, and the line is re-anchored between the fit centroid and the most recent refresh.
3. **Median.** The fitted earned count over the fitted tracked pool at the target, clamped to `[0.01, 99.99]`; a trophy already at ≥ 99.5% stays at 100.
4. **Flat launch-night history (no refresh yet).** A launch-refresh prior fitted to the house's recording of GTA VI's launch night: the tracked pool grows ×(t)^1.21 and the earned total ×(t)^0.55 per hour, so because one trophy sits at 100% and absorbs 100 points of rate mass, the other trophies' rate mass falls by `((E2 − p2)/(E1 − p1)) · (p1/p2)` — about ×0.22 at the first refresh — shaped per trophy by earned count^(1/4).
5. **Spread.** A logit-normal band whose width is the larger of the trophy family's fit residual and a horizon term (a wider fixed band on the flat branch), quantiles at z = ±1.645, ±0.674, 0.

## Validation

`tools/validate.py` runs the entry on the practice cases (`tools/fixtures/`), checks the output contract (exact keys, one row per trophy id, five finite non-decreasing values in [0, 100], ≤ 65,536 bytes), byte-identical reruns, and scores the model against the house persistence baseline under three answers (rates unchanged; the literal ×0.75 refresh; and the aggregate-consistent launch-refresh answer — platinum 0.10%, the 100% trophy unchanged, others grown as earned^(1/4) to 4,276 earned / 3,868 tracked). The model beats the baseline on every scenario; latest numbers are in the validator's output. Runtime is about 0.1 s per pair of cases (8 s cap).

## License

MIT — see [LICENSE](LICENSE).

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
