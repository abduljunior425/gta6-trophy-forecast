[![gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/abduljunior425/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/abduljunior425/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# Dilution-Aware Logit, Step-Aware Blend

Forecast for every GTA VI PS5 trophy its Exophase earn rate at
2026-11-26T00:00:00Z, as five quantiles a trophy, for
[make.fun's gta6-trophy-rarity-forecast-01 bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## The idea

Exophase does not refresh its counts continuously. On launch night the page
sat unchanged — 638 players tracked, every rate identical — from 23:48Z to
04:00Z, then jumped in one refresh to 3,868 players and 4,276 trophies
earned, and every rate but the one at 100% fell. A rate is a ratio of two
moving counts, so this entry:

1. **Extrapolates log-earners and log-players separately, then divides** —
   new players dilute rates even as earners grow. It never extrapolates
   the ratio itself.
2. **Collapses flat runs of identical reads** to one observation: a stale
   page is not zero growth, and flat reads must not drag the fit to zero.
3. **Fits a recency-weighted line in logit space** over the last 24 h of
   reads, **shrunk toward a difficulty-class prior**: story trophies are
   earned fast in week one, activity/collectible grinds slowly, the
   platinum (capped at a 2% median) slowest of all.
4. **Mixes two branches** — no refresh before the target (rate stays put)
   vs a refresh (rate jumps to the extrapolated value) — with the refresh
   branch more likely the longer the page has been stale.
5. **Shrinks the median toward persistence**, up to 60% at the final's
   6,510-minute horizon, and widens the spread with the fit residual, a
   random-walk term in sqrt(horizon), and prior uncertainty.

Growth priors are calibrated on the one refresh the rules record
(638 → 3,868 players, 943 → 4,276 trophies earned across the 04:30Z
refresh): the player curve integrates to ×6.2 over that window, the
story-earner curve to ×1.8, the grind-earner curve to ×1.3.

## Run it

Python 3.13, standard library only, no network, deterministic
(byte-identical outputs across runs):

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` case in the input directory and writes one
`<id>.out` per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75,
q95]}]}`.

## Backtest

On the practice cases, scored against the 04:30Z refresh reconstructed
from the rules' own aggregates (players 3,868, trophies earned 4,276,
the 100% trophy held by all players, the remaining earners split
pro-rata — the reconstruction reproduces the house baseline's published
practice loss of ~1.51), this entry scores 0.24–0.35 against
persistence's 1.40–1.69 across three reconstruction variants (pro-rata,
earners frozen, extra earners to story trophies only), with 98% 90%
coverage. On a growth scenario (earners ×1.4, players → 8,000) it scores
0.50–0.58 vs persistence's 2.07, and on a synthetic final case (two days
of moving history with 5-hour refresh steps, 6,510-minute horizon) 0.58
vs persistence's 1.00.

## License

MIT — see [LICENSE](LICENSE).
