# gta6-trophy-rarity

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/vcspeeper-cmd/gta6-trophy-rarity/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/vcspeeper-cmd/gta6-trophy-rarity/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Five quantiles per trophy for Grand Theft Auto VI's PS5 trophy rarities on
Exophase, a week after the entry deadline. Exophase prints a rarity as
`100 * earners / tracked players`, and the tracked player base grows all week,
so every rate drifts down. On the logit scale that drift is close to a straight
line against `ln(hours since release)`; the rare trophies fall fastest, the
common ones hardly at all. This entry fits that line per trophy over the case's
distinct reads, shrinks each slope toward the field's median so a trophy with
three noisy reads cannot run away, anchors the prediction at the trophy's own
last read, and spreads the five quantiles by the fit's residual plus a term that
widens with the distance past the last read.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`input/` holds one `<id>.in` per case (the grader runs one case at a time) and
`out/<id>.out` comes back with `{"trophies":[{"id":1,"q":[q05,q25,q50,q75,q95]}, ...]}`.
Standard library only, no network, deterministic.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
