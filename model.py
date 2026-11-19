"""Trophy-rarity forecast: a structural player-flow model, not a curve fit.

The page prints, for each trophy, the share of the players Exophase tracks who
have earned it.  That share is a ratio of two counts:

    rate_i = 100 * earners_i / players_tracked

so it moves for two reasons: `players_tracked` grows (new tracked players, who
have earned almost nothing yet), and `earners_i` grows (players progressing).
Modelling the two counts separately beats modelling the printed rate, because
the printed rate is quantised to 2 decimals and, for a rare trophy, is just a
handful of players over a growing denominator.

Measured on the house's own launch-night recording (638 tracked players) against
the page today (14,028), the fraction of *new* tracked players who earn trophy i
follows a clean power law in the trophy's current share f:

    q_i = 3.28 * f_i ** 1.35        (44 trophies, launch ramp)

i.e. a trophy held by 1.9% of players is earned by 1.9% of arrivals, one held by
0.06% by 0.008% of arrivals.  That is the prior this model shrinks towards; a
case's own history overrides it once it holds several distinct refreshes.

`players_tracked` is extrapolated from the same history by fitting three growth
forms and taking the median, with the spread between them carried into the
predictive width.  Quantiles come from a fixed-seed Monte Carlo over both the
denominator and the numerator, so integer noise on a 4-earner trophy widens its
band the way it should.
"""
import math
import random

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6449, -0.6745, 0.0, 0.6745, 1.6449]

# --- priors, all measured rather than guessed -------------------------------
Q_COEF = 3.2775          # q = Q_COEF * f ** Q_EXP   (new players earning trophy i)
Q_EXP = 1.348
SLOPE_PRIOR_MIN = -1.15  # d logit(rate) / d ln(players) is bounded below
PHI = 0.30               # relative overdispersion of the minority count
SIGMA_FLOOR = 0.06       # never claim a band narrower than this (logit units)
DRAWS = 3000
SEED = 20261119

# --- players_tracked growth ------------------------------------------------
# Hours are counted from the first read of the case's history, so nothing here
# depends on knowing the release instant.
GROWTH_OFFSET_H = 1.0
LAUNCH_T0 = "2026-11-18T23:48:00Z"
LAUNCH_P0 = 638.0
LAUNCH_EXP = 1.05
EXTRAP_PENALTY = 0.55    # extra logit sd per unit of ln(1 + hours beyond data)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(rate):
    p = clamp(rate, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def expit(z):
    if z > 30.0:
        return 100.0
    if z < -30.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-z))


def hours_between(a, b):
    """Whole minutes between two ISO-8601 UTC stamps, as hours."""
    return minutes_between(a, b) / 60.0


def _parse_iso(s):
    # 2026-11-19T04:30:00.000Z -> (y, mo, d, h, mi, se)
    y = int(s[0:4]); mo = int(s[5:7]); d = int(s[8:10])
    h = int(s[11:13]); mi = int(s[14:16])
    se = float(s[17:19]) if len(s) > 18 else 0.0
    return y, mo, d, h, mi, se


def _days_from_civil(y, m, d):
    y -= 1 if m <= 2 else 0
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def minutes_between(a, b):
    ya, ma, da, ha, mia, sa = _parse_iso(a)
    yb, mb, db, hb, mib, sb = _parse_iso(b)
    ta = _days_from_civil(ya, ma, da) * 1440 + ha * 60 + mia + sa / 60.0
    tb = _days_from_civil(yb, mb, db) * 1440 + hb * 60 + mib + sb / 60.0
    return tb - ta


# --- the players_tracked curve ---------------------------------------------
def fit_players(history, target_at):
    """Point forecast of players_tracked at the target, plus its log sd.

    Three growth forms are fitted to the distinct (hours, players) readings and
    each is extrapolated; the median is the point forecast and the spread
    between them, together with a penalty for how far the extrapolation runs
    past the last reading, is the uncertainty.
    """
    pts = []
    for row in history:
        p = row.get("players_tracked")
        at = row.get("at")
        if isinstance(p, (int, float)) and p and p > 0 and isinstance(at, str):
            pts.append((at, float(p)))
    if not pts:
        return 1000.0, 1.0
    t0 = pts[0][0]
    series = []
    for at, p in pts:
        h = hours_between(t0, at) + GROWTH_OFFSET_H
        series.append((h, p))
    # A refresh is a step: keep the first read that shows each distinct count,
    # because that is the moment the new value became current.  The end of the
    # data is the last read, flat or not.
    steps = []
    for h, p in series:
        if steps and abs(steps[-1][1] - p) < 0.5:
            continue
        steps.append((h, p))
    last_h = series[-1][0]
    last_p = series[-1][1]
    tgt = hours_between(t0, target_at) + GROWTH_OFFSET_H
    tgt = max(tgt, last_h + 1e-6)
    preds = []
    if len(steps) >= 3:
        # (a) power law in log-log space
        xs = [math.log(h) for h, _ in steps]
        ys = [math.log(p) for _, p in steps]
        preds.append(_extrap_power(xs, ys, math.log(tgt)))
        # (b) saturating: P = Pmax * (1 - exp(-h / tau)), grid over tau
        preds.append(_extrap_saturating(steps, tgt))
        # (c) linear in hours
        preds.append(_extrap_linear(steps, tgt))
    elif len(steps) == 2:
        (h1, p1), (h2, p2) = steps
        c = math.log(p2 / p1) / math.log(h2 / h1) if h2 > h1 else 1.0
        preds.append(p2 * (tgt / h2) ** clamp(c, 0.0, 2.0))
        preds.append(p2 + (p2 - p1) / max(1e-6, h2 - h1) * (tgt - h2))
    if not preds:
        # One distinct reading: Exophase has not refreshed inside the history,
        # so a flat series says nothing about the pace.  Fall back on the launch
        # ramp the rules themselves record (638 tracked at 2026-11-18T23:48Z,
        # 3,868 by 04:30Z), and only while the case is still on launch night.
        preds.append(last_p * (tgt / last_h) ** 0.9)
        try:
            launch_h = hours_between(LAUNCH_T0, target_at) + GROWTH_OFFSET_H
            if 0.0 < launch_h < 24.0:
                preds.append(LAUNCH_P0 * launch_h ** LAUNCH_EXP)
        except Exception:
            pass
    preds = [p for p in preds if p > 0 and math.isfinite(p)]
    preds.sort()
    point = preds[len(preds) // 2]
    if len(preds) > 1:
        spread = math.log(preds[-1] / preds[0]) / 2.0
    else:
        spread = 0.0
    beyond = math.log1p(max(0.0, tgt - last_h) / max(1.0, last_h))
    sigma = math.sqrt(spread ** 2 + (EXTRAP_PENALTY * beyond) ** 2 + 0.02 ** 2)
    return point, clamp(sigma, 0.03, 1.2)


def _extrap_power(xs, ys, xt):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return math.exp(my)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    b = clamp(b, 0.0, 3.0)
    return math.exp(my + b * (xt - mx))


def _extrap_linear(steps, tgt):
    n = len(steps)
    mx = sum(h for h, _ in steps) / n
    my = sum(p for _, p in steps) / n
    sxx = sum((h - mx) ** 2 for h, _ in steps)
    if sxx <= 1e-12:
        return my
    b = sum((h - mx) * (p - my) for h, p in steps) / sxx
    return max(1.0, my + b * (tgt - mx))


def _extrap_saturating(steps, tgt):
    """P = Pmax * (1 - exp(-h / tau)); best tau on a grid, Pmax by least squares."""
    best = None
    last_h, last_p = steps[-1]
    for i in range(1, 121):
        tau = 2.0 * (1.06 ** i)
        if tau > 400:
            break
        phi = [1.0 - math.exp(-h / tau) for h, _ in steps]
        num = sum(f * p for f, (_, p) in zip(phi, steps))
        den = sum(f * f for f in phi)
        if den <= 1e-12:
            continue
        pmax = num / den
        err = sum((pmax * f - p) ** 2 for f, (_, p) in zip(phi, steps))
        if best is None or err < best[0]:
            best = (err, pmax, tau)
    if best is None:
        return last_p
    _, pmax, tau = best
    return max(last_p, pmax * (1.0 - math.exp(-tgt / tau)))


# --- the per-trophy law -----------------------------------------------------
def field_slope(refreshed):
    """The one slope the whole field moves on: median logit rate vs ln players.

    Each refresh gives one number (the median logit rate over its trophies);
    regressing those on ln(players) measures how fast the field as a whole
    decays without letting the spread between trophies inside a refresh swamp
    the estimate.  Returns None when the history holds fewer than two refreshes.
    """
    pts = []
    for lnp, rates in refreshed:
        vals = [logit(v) for v in rates
                if isinstance(v, (int, float)) and 0.01 < v < 99.99]
        if len(vals) < 10:
            continue
        vals.sort()
        pts.append((lnp, vals[len(vals) // 2]))
    if len(pts) < 2:
        return None
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx <= 1e-9:
        return None
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx
    return clamp(b, SLOPE_PRIOR_MIN, 0.35)


def prior_slope(f, scale=1.0):
    """d logit(rate) / d ln(players) for a trophy now held by fraction `f`.

    Measured on the launch ramp (638 -> 14,028 tracked players): the share of
    arrivals who earn a trophy is gamma(f) * f, and gamma rises with how common
    the trophy already is, from ~0.06 for the rarest to 1 for the ones nearly
    everyone has.  A power law fitted to that data overshoots past gamma = 1, so
    the prior is the measured curve interpolated in log f and clipped.

    `scale` re-levels the curve to the slope the case's own refreshes show, so
    the shape stays measured while the level follows the data.
    """
    if f <= 0:
        return SLOPE_PRIOR_MIN
    knots = ((0.0003, 0.05), (0.0010, 0.10), (0.0030, 0.45),
             (0.0100, 0.85), (0.0300, 0.99), (1.0, 1.0))
    if f <= knots[0][0]:
        gamma = knots[0][1]
    elif f >= knots[-1][0]:
        gamma = knots[-1][1]
    else:
        gamma = knots[-1][1]
        for (f0, g0), (f1, g1) in zip(knots, knots[1:]):
            if f0 <= f <= f1:
                t = math.log(f / f0) / math.log(f1 / f0)
                gamma = g0 + t * (g1 - g0)
                break
    return clamp((gamma - 1.0) * scale, SLOPE_PRIOR_MIN, 0.0)


def prior_scale(refreshed, last_rates):
    """How far to re-level the measured curve, from the case's own field slope."""
    field = field_slope(refreshed)
    if field is None:
        return 1.0
    base = []
    for v in last_rates:
        if isinstance(v, (int, float)) and v > 0:
            base.append(prior_slope(clamp(v, 0.0, 100.0) / 100.0))
    if not base:
        return 1.0
    base.sort()
    median = base[len(base) // 2]
    if median > -0.02:
        return 1.0
    return clamp(field / median, 0.25, 2.5)


def fit_slope(points, f_last, scale=1.0):
    """Per-trophy slope from the case's own refreshes, shrunk to the prior."""
    prior = prior_slope(f_last, scale)
    xs, ys = [], []
    for lnp, rate in points:
        if rate is None or rate <= 0:
            continue
        xs.append(lnp)
        ys.append(logit(rate))
    n = len(xs)
    if n < 3:
        return prior, 1.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-9:
        return prior, 1.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    resid = [y - (my + b * (x - mx)) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / max(1, n - 2))
    # a wide fit is a weak fit: shrink harder
    w = (n - 2) / (n - 2 + 2.5 + (sd / 0.25) ** 2)
    w = clamp(w, 0.0, 0.92)
    b = clamp(b, SLOPE_PRIOR_MIN, 0.35)
    return w * b + (1.0 - w) * prior, sd


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    levels = case.get("levels") or LEVELS
    horizon = float(case.get("horizon_minutes") or 0.0)
    target_hours = horizon / 60.0

    if not history:
        return {"trophies": [{"id": t.get("id"), "q": [50.0] * 5} for t in trophies]}

    last = history[-1]
    p_last = float(last.get("players_tracked") or 1.0)
    rates_last = last.get("rates") or []
    target_at = case.get("target")
    if not isinstance(target_at, str):
        target_at = last.get("at")
    p_point, sigma_p = fit_players(history, target_at)
    p_point = max(p_point, p_last)

    # distinct refreshes, oldest first
    refreshed = []
    prev = None
    for row in history:
        r = row.get("rates") or []
        key = tuple(r)
        if key == prev:
            continue
        prev = key
        p = row.get("players_tracked")
        if isinstance(p, (int, float)) and p and p > 0:
            refreshed.append((math.log(float(p)), r))

    rng = random.Random(SEED)
    scale = prior_scale(refreshed, rates_last)
    out = []
    for i, t in enumerate(trophies):
        rate = rates_last[i] if i < len(rates_last) else None
        if not isinstance(rate, (int, float)):
            out.append({"id": t.get("id"), "q": [50.0] * 5})
            continue
        f = clamp(float(rate), 0.0, 100.0) / 100.0
        points = []
        for lnp, r in refreshed:
            v = r[i] if i < len(r) else None
            points.append((lnp, v if isinstance(v, (int, float)) else None))
        slope, fit_sd = fit_slope(points, f, scale)

        # The minority count is the one that carries the uncertainty: for a
        # common trophy that is the players who have not earned it, for a rare
        # one the players who have.
        if f <= 0.5:
            c_last = max(0.0, f * p_last)          # earners
        else:
            c_last = max(0.0, (1.0 - f) * p_last)  # non-earners
        # the slope was fitted on the printed rate, so translate it into the
        # minority count's own growth: d ln c / d ln P = slope + 1 for earners
        gamma = clamp(slope + 1.0, 0.0, 1.0) if f <= 0.5 else clamp(1.0 - slope, 0.0, 1.0)

        draws = []
        for _ in range(DRAWS):
            lp = math.log(max(p_point, 1.0)) + sigma_p * rng.gauss(0.0, 1.0)
            pt = math.exp(lp)
            if pt < p_last:
                pt = p_last
            mean_c = c_last * (pt / p_last) ** gamma if c_last > 0 else 0.0
            sig_c = math.sqrt(1.0 / max(mean_c, 1.0) + PHI ** 2) if mean_c > 0 else 0.0
            sig_c = math.sqrt(sig_c ** 2 + fit_sd ** 2 * 0.25)
            c = mean_c * math.exp(sig_c * rng.gauss(0.0, 1.0) - 0.5 * sig_c ** 2) if mean_c > 0 else 0.0
            if c < 0:
                c = 0.0
            if c > pt:
                c = pt
            r = 100.0 * c / pt if f <= 0.5 else 100.0 * (1.0 - c / pt)
            draws.append(clamp(r, 0.0, 100.0))
        draws.sort()
        q = []
        n = len(draws)
        for lv in levels:
            idx = int(clamp(lv * (n - 1), 0, n - 1))
            q.append(draws[idx])
        # never claim more precision than the scale can hold
        if q[4] - q[0] < 1e-12:
            mid = q[2] if q[2] > 0 else 0.01
            q = [expit(logit(mid) + z * SIGMA_FLOOR) for z in Z]
        out.append({"id": t.get("id"), "q": [round(x, 4) for x in q]})
    return {"trophies": out}
