"""Dilution-Aware Logit, Step-Aware Blend — gta6-trophy-rarity-forecast-01.

Forecasts every GTA VI PS5 trophy's Exophase earn rate at a case's target,
as five quantiles (5/25/50/75/95%), from the house's reads of the trophy page.

Why this shape, in one paragraph: Exophase does not refresh its counts
continuously — on launch night the page sat unchanged (638 players tracked,
every rate identical) from 23:48Z to 04:00Z, then jumped in one refresh to
3,868 players and 4,276 trophies earned, and every rate but the one at 100%
fell. So a rate is a ratio of two moving counts (earners over players
tracked), both stale between refreshes, and a forecaster that extrapolates
the ratio itself misses the dilution: new players arrive faster than new
earners. This program therefore extrapolates log(earners) and log(players
tracked) separately and divides at the target; it collapses flat runs of
identical reads to a single observation (a stale page is not zero growth);
it shrinks each fitted slope toward a difficulty-class prior (story trophies
are earned fast in week one, grind trophies slowly); it treats the forecast
as a two-branch mixture — no refresh before the target (rate stays put) vs
a refresh (rate jumps to the extrapolated value) — with the refresh branch
more likely the longer the page has been stale; and it shrinks the median
toward persistence, up to 60% at the final's 6,510-minute horizon.

Growth priors (log-growth per hour, decaying with hours since release):
they are calibrated on the only refresh the rules record — players
638 -> 3,868 and trophies earned 943 -> 4,276 across the 04:30Z refresh,
about 4.7 h after the first read — which implies true player growth near
0.38 log/h on launch night and earner growth near 0.06–0.12 log/h. The
player curve below integrates to x6.2 over that window; the story-earner
curve to x1.8 and the grind-earner curve to x1.3. By the final's deadline
the priors have decayed to a few percent per hour and the fits (taken over
the last 24 h of reads, where the page actually moves) dominate them.

Standard library only. Deterministic: no randomness, no wall clock, no
network. Reads every *.in in $MAKEFUN_INPUT_DIR (else /input), writes
/output/<id>.out (else $MAKEFUN_OUTPUT_DIR).

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)

# Trophy difficulty classes, by Exophase list position (id). Story trophies
# are earned by playing the main missions (fast in week one); activity and
# collectible trophies are grinds (slow); the platinum needs all 54 others.
STORY_IDS = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22, 23, 24, 25, 26, 27, 29}
GRIND_IDS = {14, 15, 16, 17, 18, 19, 20, 21, 28, 30, 31, 32, 33, 34, 35, 36,
             37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
             53, 54, 55}

# Prior instantaneous log-growth rates per hour, as (a, b, c) in
# g(t) = a * exp(-t / b) + c, t = hours since release. Players surge at
# launch and decay; story earners rise fast then slow; grind earners crawl.
PRIOR_CURVES = {
    "players": (0.45, 10.0, 0.015),
    "story": (0.12, 24.0, 0.008),
    "grind": (0.05, 24.0, 0.002),
    "platinum": (0.01, 24.0, 0.001),
}

REFRESH_INTERVAL_H = 4.5   # Exophase's counts refresh every few hours
HALF_LIFE_H = 6.0          # recency half-life of the weighted fits
PLATINUM_CAP = 2.0         # week-one platinum median never above 2%


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    return (moment - RELEASE).total_seconds() / 3600.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (as the starter does)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def avg_prior(kind, t0, t1):
    """Mean of the prior growth curve over [t0, t1] (hours since release)."""
    a, b, c = PRIOR_CURVES[kind]
    if t1 <= t0:
        return a * math.exp(-t0 / b) + c
    integral = a * b * (math.exp(-t0 / b) - math.exp(-t1 / b)) + c * (t1 - t0)
    return integral / (t1 - t0)


def wls_slope(times, values, weights):
    """Weighted least-squares slope (per hour) and intercept value."""
    sw = sum(weights)
    if sw <= 0:
        return 0.0, (values[-1] if values else 0.0)
    mt = sum(w * t for w, t in zip(weights, times)) / sw
    mv = sum(w * v for w, v in zip(weights, values)) / sw
    den = sum(w * (t - mt) ** 2 for w, t in zip(weights, times))
    if den <= 1e-12:
        return 0.0, mv
    slope = sum(w * (t - mt) * (v - mv)
                for w, t, v in zip(weights, times, values)) / den
    return slope, mv


def trophy_kind(trophy):
    tid = trophy["id"]
    if trophy.get("tier") == "platinum" or tid == 1:
        return "platinum"
    if tid in STORY_IDS:
        return "story"
    if tid in GRIND_IDS:
        return "grind"
    if "Chapter" in trophy.get("description", ""):
        return "story"
    return "grind"


def collapse_flat(reads):
    """Collapse runs of identical consecutive reads to their first read.

    A run of identical reads is one stale Exophase page, not many
    observations of zero growth: keeping them would drag every fit to a
    slope of zero and fake a zero residual.
    """
    out = []
    for read in reads:
        key = (read["players_tracked"], tuple(read["rates"]))
        if out and out[-1][0] == key:
            continue
        out.append((key, read))
    return [read for _, read in out]


def refresh_probability(stale_age_h, horizon_h):
    """Chance a refresh lands before the target.

    Refreshes come every few hours, so a page already stale for most of an
    interval is due: once stale_age + horizon covers the interval the
    refresh branch is the likely one. Over a horizon of many intervals a
    refresh is near-certain.
    """
    if horizon_h >= 3.0 * REFRESH_INTERVAL_H:
        return 0.99
    if stale_age_h + horizon_h >= REFRESH_INTERVAL_H:
        return 0.85
    return 1.0 - math.exp(-horizon_h / REFRESH_INTERVAL_H)


def mixture_quantile(taus, last_lg, center_b, sd_a, sd_b, p_refresh):
    """Quantiles of the two-branch mixture on the logit scale.

    Branch A (weight 1-p): no refresh, rate stays at the last read.
    Branch B (weight p): a refresh lands, rate is the extrapolated value.
    Solved by bisection on the mixture CDF; fully deterministic.
    """
    lo = min(last_lg, center_b) - 6.0 * max(sd_a, sd_b, 0.1) - 1.0
    hi = max(last_lg, center_b) + 6.0 * max(sd_a, sd_b, 0.1) + 1.0

    def cdf(x):
        return ((1.0 - p_refresh) * phi((x - last_lg) / sd_a)
                + p_refresh * phi((x - center_b) / sd_b))

    out = []
    for tau in taus:
        a, b = lo, hi
        for _ in range(80):
            mid = 0.5 * (a + b)
            if cdf(mid) < tau:
                a = mid
            else:
                b = mid
        out.append(0.5 * (a + b))
    return out


def forecast_case(case, name):
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    taus = case.get("levels", LEVELS)
    checkpoint = parse_time(case["at"])
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_days = case["horizon_minutes"] / 1440.0
    t_target = hours_since_release(checkpoint) + horizon_h

    # Last 24 h of reads, flat runs collapsed to one observation each.
    cutoff = checkpoint.timestamp() - 24.0 * 3600.0
    recent = [r for r in history if parse_time(r["at"]).timestamp() >= cutoff]
    if len(recent) < 2:
        recent = history[-2:] if len(history) >= 2 else history
    obs = collapse_flat(recent)
    times = [hours_since_release(parse_time(r["at"])) for r in obs]
    t_anchor = times[-1]
    weights = [math.exp(-(t_anchor - t) / HALF_LIFE_H) for t in times]
    last = obs[-1]
    last_players = last["players_tracked"]
    last_rates = last["rates"]

    # Terminal flat run: how long the displayed page has already been stale.
    # Measured on the raw recent reads: collapse_flat has already removed
    # the duplicates, so the collapsed observations can never show a flat
    # run — measuring on them always reports the whole 24 h window as
    # stale and overstates the refresh probability whenever the page
    # actually moved recently.
    flat_start_t = hours_since_release(parse_time(recent[-1]["at"]))
    for j in range(len(recent) - 1, 0, -1):
        a, b = recent[j], recent[j - 1]
        if (a["players_tracked"], tuple(a["rates"])) == \
           (b["players_tracked"], tuple(b["rates"])):
            flat_start_t = hours_since_release(parse_time(b["at"]))
        else:
            break
    stale_age_h = max(0.0, hours_since_release(checkpoint) - flat_start_t)
    p_refresh = refresh_probability(stale_age_h, horizon_h)

    # --- players tracked: fit log(P), shrunk to the player prior ---
    log_p = [math.log(max(1, r["players_tracked"])) for r in obs]
    slope_p_fit, _ = wls_slope(times, log_p, weights)
    n_p = len({round(v, 6) for v in log_p})
    # One distinct value carries no slope information at all (a flat page);
    # with several distinct values spanning the window the fit is trusted
    # and the prior is only a weak regulariser.
    w_p = (n_p - 1) / ((n_p - 1) + 1.0) if n_p >= 2 else 0.0
    slope_p = w_p * slope_p_fit + (1.0 - w_p) * avg_prior(
        "players", t_anchor, t_target)
    log_p_target = math.log(max(1, last_players)) + slope_p * (t_target - t_anchor)

    # Horizon shrink toward persistence (grows with horizon, 60% at the
    # final's 6,510 minutes, almost none at 90 minutes).
    shrink = 0.6 * (1.0 - math.exp(-horizon_days / 1.5))
    spread_rw = 0.45 * math.sqrt(horizon_days)

    rows = []
    for i, trophy in enumerate(trophies):
        kind = trophy_kind(trophy)
        last_rate = last_rates[i]
        if last_rate is None:
            # Fall back to the most recent read that lists this trophy.
            for read in reversed(obs):
                if read["rates"][i] is not None:
                    last_rate = read["rates"][i]
                    break
            if last_rate is None:
                last_rate = 1.0
        last_lg = logit(last_rate)

        # A trophy every tracked player holds (100%) cannot be diluted:
        # its earners are the players themselves, so it stays put. Give it
        # the narrow random-walk spread of the persistence baseline — on
        # the logit scale a wide spread near 100% would be very costly.
        if last_rate >= 99.99:
            spread = 0.5 * math.sqrt(horizon_days)
            lg_q = mixture_quantile(taus, last_lg, last_lg, 0.15, spread,
                                    0.5)
            q = [min(99.9995, max(0.0005, inv_logit(v))) for v in lg_q]
            q = [round4(v) for v in q]
            rows.append({"id": trophy["id"], "q": q})
            continue

        # --- earners: fit log(E + 0.5), shrunk to the class prior ---
        series = []
        for r, t in zip(obs, times):
            rate = r["rates"][i]
            if rate is None:
                continue
            earners = rate / 100.0 * r["players_tracked"]
            series.append((t, math.log(earners + 0.5)))
        if len(series) >= 2:
            st, sv = zip(*series)
            sw = [math.exp(-(t_anchor - t) / HALF_LIFE_H) for t in st]
            slope_e_fit, _ = wls_slope(list(st), list(sv), sw)
            n_e = len({round(v, 6) for v in sv})
        else:
            slope_e_fit, n_e = 0.0, 1
        w_e = (n_e - 1) / ((n_e - 1) + 1.0) if n_e >= 2 else 0.0
        slope_e = w_e * slope_e_fit + (1.0 - w_e) * avg_prior(
            kind, t_anchor, t_target)
        earners_last = last_rate / 100.0 * last_players
        log_e_target = (math.log(earners_last + 0.5)
                        + slope_e * (t_target - t_anchor))

        # --- dilution-aware median: earners over players at the target ---
        model_rate = 100.0 * math.exp(log_e_target - log_p_target)
        if not (0.0 < model_rate < 10000.0):
            model_rate = last_rate
        model_lg = logit(model_rate)
        center_b = (1.0 - shrink) * model_lg + shrink * last_lg

        # --- spread: fit residual + random walk + prior uncertainty ---
        if len(series) >= 3:
            resid = []
            for r, t in zip(obs, times):
                rate = r["rates"][i]
                if rate is None:
                    continue
                earners = rate / 100.0 * r["players_tracked"]
                resid.append(math.log(earners + 0.5)
                             - math.log(max(1, r["players_tracked"])))
            r_mean = sum(resid) / len(resid)
            s_fit = math.sqrt(sum((v - r_mean) ** 2 for v in resid)
                              / max(1, len(resid) - 1))
            s_fit = min(1.0, max(0.15, s_fit))
        else:
            s_fit = 0.30
        drift = abs(center_b - last_lg)
        s_prior = min(1.0, max(0.25, 0.5 * drift))
        sd_b = math.sqrt(s_fit ** 2 + spread_rw ** 2 + s_prior ** 2)

        lg_q = mixture_quantile(taus, last_lg, center_b, 0.15, sd_b,
                                p_refresh)
        q = [inv_logit(v) for v in lg_q]

        # Platinum can only pop once the other 54 are earned: cap its median.
        if kind == "platinum" and len(q) >= 3 and q[2] > PLATINUM_CAP:
            q[2] = PLATINUM_CAP

        q = [min(99.9995, max(0.0005, v)) for v in q]
        mono = []
        running = -1.0
        for v in q:
            running = max(running, v)
            mono.append(running)
        mono = [round4(v) for v in mono]
        fixed = []
        running = -1.0
        for v in mono:
            running = max(running, v)
            fixed.append(running)
        rows.append({"id": trophy["id"], "q": fixed})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception as exc:  # never crash a case: fall back to last rates
            print(f"warning: {path.name} failed ({exc}); using last rates",
                  file=sys.stderr)
            last = case["history"][-1]
            rows = []
            for i, trophy in enumerate(case["trophies"]):
                rate = last["rates"][i]
                if rate is None:
                    rate = 1.0
                rows.append({"id": trophy["id"],
                             "q": [round4(min(99.9995, max(0.0005, rate)))] * 5})
            forecast = {"trophies": rows}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
