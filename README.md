# gta6-trophy-rarity

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/vcspeeper-cmd/gta6-trophy-rarity/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/vcspeeper-cmd/gta6-trophy-rarity/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Five quantiles per trophy for Grand Theft Auto VI's PS5 trophy rarities on
Exophase, a week after the entry deadline.

Exophase prints a rarity as `100 * earners / tracked players`, so the printed
number is a ratio of two counts and it drifts down for two reasons: the tracked
player base grows with arrivals who have earned almost nothing yet, and the
earners of each trophy grow as players progress. This entry models the two
counts instead of the ratio.

- `players_tracked` is extrapolated from the case's own refresh steps by three
  growth forms — a power law in elapsed time, a saturating `Pmax * (1 - e^-t/tau)`
  and a straight line — and the median is the point forecast. The spread between
  the three, plus a penalty for how far past the last read the target sits, is
  the denominator's uncertainty.
- Each trophy's minority count (the earners for a rare trophy, the non-earners
  for a common one) grows as a power of the denominator. The exponent is fitted
  from the case's own refreshes and shrunk to a curve measured on the launch
  ramp: between 638 and 14,028 tracked players the share of arrivals who earn a
  trophy is `gamma(f) * f`, with `gamma` rising from about 0.06 for the rarest
  trophies to 1 for the ones nearly everyone has. The curve is re-levelled to
  the slope the case's own median logit rate shows, so its shape stays measured
  while its level follows the data.
- The five quantiles come from a fixed-seed Monte Carlo over both counts, so a
  trophy with four earners carries the integer noise its band should, and a
  trophy at 99.99% stays pinned instead of sprouting a spurious spread.

Standard library only, no network, deterministic, about 0.3 s for 55 trophies.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`input/` holds one `<id>.in` per case (the grader runs one case at a time) and
`out/<id>.out` comes back with `{"trophies":[{"id":1,"q":[q05,q25,q50,q75,q95]}, ...]}`.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
