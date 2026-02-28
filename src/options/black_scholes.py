from __future__ import annotations

import math


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(spot: float, strike: float, t_years: float, rate: float, vol: float, is_call: bool) -> float:
    """
    Black-Scholes European option price.
    """
    spot = max(float(spot), 1e-9)
    strike = max(float(strike), 1e-9)
    t = max(float(t_years), 1e-9)
    vol = max(float(vol), 1e-6)
    rate = float(rate)

    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)

    if is_call:
        return spot * norm_cdf(d1) - strike * math.exp(-rate * t) * norm_cdf(d2)
    return strike * math.exp(-rate * t) * norm_cdf(-d2) - spot * norm_cdf(-d1)

