"""The forecast.

Exophase prints a trophy's rarity as 100 * earners / tracked players, and the
tracked player base grows through the week, so every rate drifts. On the logit
scale that drift is close to a straight line against ln(hours since release):
the rare trophies fall fastest, the common ones hardly at all. This module fits
that line per trophy over the case's distinct reads, shrinks each slope toward
the field's median so a trophy with three noisy reads cannot run away, anchors
the prediction at the trophy's own last read, and then spreads five quantiles
around it: the fit's residual plus a widening term for the distance past the
last read, because the further the target, the less the line is worth.
"""

import math

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722)
FLOOR = 0.01
CEIL = 99.99
MIN_HOURS = 0.25
# How many reads' worth of weight the field's median slope carries against a
# trophy's own fit, and how far the shrunk slope may stray from it.
SHRINK = 2.0
SLOPE_CAP = 3.0
# The spread never leaves this band, in logit units.
SPREAD_MIN = 0.15
SPREAD_MAX = 2.0
# How much the band widens per e-fold of distance past the last read.
STRETCH_RATE = 0.28
# The prior spread when a trophy has too few reads to fit anything.
PRIOR_SPREAD = 0.55

_MONTH_DAYS = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)


def clamp_rate(x: float) -> float:
    return min(CEIL, max(FLOOR, x))


def logit(rate: float) -> float:
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(z: float) -> float:
    if z >= 0:
        return 100.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def iso_hours(stamp: str) -> float:
    """Hours on a fixed epoch for a YYYY-MM-DDTHH:MM:SS(.sss)Z stamp.

    Parsed by hand: the entry ships nothing but the standard library, and a
    hand parser cannot trip over a platform's datetime quirks.
    """
    text = stamp.strip()
    year = int(text[0:4])
    month = int(text[5:7])
    day = int(text[8:10])
    hour = int(text[11:13])
    minute = int(text[14:16])
    second = int(text[17:19])
    frac = 0.0
    if len(text) > 20 and text[19] == '.':
        end = text.index('Z') if 'Z' in text else len(text)
        frac = float('0.' + text[20:end])
    days = 365 * (year - 1970) + (year - 1969) // 4 + _MONTH_DAYS[month - 1] + day - 1
    if month > 2 and year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        days += 1
    return days * 24.0 + hour + minute / 60.0 + (second + frac) / 3600.0


def distinct_reads(history):
    """The reads whose rate vector differs from the one before it.

    Exophase refreshes a game's counts hours apart, so the recorder's cadence
    repeats the same numbers many times; a repeated read adds no slope.
    """
    out = []
    previous = None
    for read in history:
        key = tuple(read.get("rates") or ())
        if key and key != previous:
            out.append(read)
            previous = key
    return out


def fit_line(points):
    """Least squares of logit(rate) on ln(hours): (slope, residual sd, n)."""
    xs = [math.log(t) for t, _ in points]
    ys = [logit(r) for _, r in points]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0, 0.0, n
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / max(1, n - 2))
    return slope, sd, n


def collect(case):
    """Per trophy: its (hours, rate) points over the case's distinct reads."""
    release = iso_hours(case["game"]["release_date"] + "T00:00:00Z")
    reads = distinct_reads(case["history"]) or case["history"]
    stamps = [max(MIN_HOURS, iso_hours(read["slot"]) - release) for read in reads]
    per_trophy = []
    for index in range(len(case["trophies"])):
        points = []
        for stamp, read in zip(stamps, reads):
            rates = read.get("rates") or []
            if index >= len(rates):
                continue
            rate = rates[index]
            if rate is None or rate <= 0.0 or rate >= 100.0:
                continue
            points.append((stamp, float(rate)))
        per_trophy.append(points)
    target_hours = max(MIN_HOURS, iso_hours(case["target"]) - release)
    return per_trophy, target_hours


def field_slope(per_trophy):
    """The median of the per-trophy slopes, weighted by how many reads each had."""
    fitted = []
    for points in per_trophy:
        if len(points) >= 3:
            slope, _, n = fit_line(points)
            fitted.append((slope, n))
    if not fitted:
        return 0.0
    fitted.sort()
    total = sum(n for _, n in fitted)
    seen = 0
    for slope, n in fitted:
        seen += n
        if seen * 2 >= total:
            return slope
    return fitted[-1][0]


def forecast_case(case: dict) -> dict:
    per_trophy, target_hours = collect(case)
    pooled = field_slope(per_trophy)
    rows = []
    for index, trophy in enumerate(case["trophies"]):
        points = per_trophy[index]
        if not points:
            centre = logit(1.0)
            spread = PRIOR_SPREAD + STRETCH_RATE * 2.0
            t_last = MIN_HOURS
        else:
            t_last, r_last = points[-1]
            if len(points) >= 3:
                slope, residual, n = fit_line(points)
                slope = (n * slope + SHRINK * pooled) / (n + SHRINK)
                if pooled < 0.0:
                    slope = max(slope, SLOPE_CAP * pooled)
                spread = math.sqrt(residual * residual + PRIOR_SPREAD * PRIOR_SPREAD * 0.5)
            else:
                slope = pooled
                spread = PRIOR_SPREAD
            stretch = math.log(max(1.0, target_hours / t_last))
            centre = logit(r_last) + slope * stretch
            spread = math.sqrt(spread * spread + (STRETCH_RATE * stretch) ** 2)
        spread = min(SPREAD_MAX, max(SPREAD_MIN, spread))
        quantiles = [round4(inv_logit(centre + z * spread)) for z in Z]
        for i in range(1, 5):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}
