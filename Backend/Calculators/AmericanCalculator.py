import scipy.stats as si
import numpy as np
from datetime import datetime, time

# ---------------------------------------------------------
# Filename: AmericanCalculator.py
# Author:
# Created: 2025-10-30
# Description: AmericanOptionCalculator
# ---------------------------------------------------------


def _parse_time_flexible(tstr):
    """Handles 'HH:MM', 'HH:MM:SS', or 'AM'/'PM'."""
    if not tstr:
        return time(0, 0)
    tstr = str(tstr).strip().upper()
    if tstr in {"AM"}:
        return time(0, 0)
    if tstr in {"PM"}:
        return time(12, 0)
    for fmt in ["%H:%M", "%H:%M:%S", "%I:%M %p", "%I %p"]:
        try:
            return datetime.strptime(tstr, fmt).time()
        except ValueError:
            continue
    return time(0, 0)


def _yearfrac(start_dt, end_dt):
    """Actual/365 year fraction."""
    return max((end_dt - start_dt).total_seconds() / (365 * 24 * 3600), 0.0)


def _normalize_rate(x):
    """Treat 1.5 as 1.5%."""
    x = float(x)
    return x / 100.0 if x > 1 else x


def _pv_dividends(div_list, start_dt, exp_dt, r):
    """PV of discrete dividends before expiry."""
    if not isinstance(div_list, list):
        return 0.0
    pv = 0.0
    for d in div_list:
        pay_date = datetime.strptime(d["date"], "%Y-%m-%d")
        if start_dt < pay_date < exp_dt:
            T = _yearfrac(start_dt, pay_date)
            pv += float(d["amount"]) * np.exp(-r * T)
    return pv


def _american_crr_price(S, K, r, q, sigma, T, steps, is_call=True):
    import math
    S = max(S, 1e-12)
    if T <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    dt = T / steps
    if sigma <= 0 or dt <= 0:
        # No volatility: price equals intrinsic at start (no time value)
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    if abs(u - d) < 1e-14:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    disc = math.exp(-r * dt)
    drift = math.exp((r - q) * dt)
    p = (drift - d) / (u - d)
    # clamp to [0,1] to avoid arbitrage issues from numerics
    p = max(0.0, min(1.0, p))

    # Terminal values
    vals = [0.0] * (steps + 1)
    S_ud = S * (d ** steps)
    for j in range(steps + 1):
        S_T = S_ud * (u / d) ** j
        intrinsic = max(S_T - K, 0.0) if is_call else max(K - S_T, 0.0)
        vals[j] = intrinsic

    # Backward induction with early exercise
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            cont = disc * (p * vals[j + 1] + (1 - p) * vals[j])
            S_ij = S * (u ** j) * (d ** (i - j))
            exercise = max(S_ij - K, 0.0) if is_call else max(K - S_ij, 0.0)
            vals[j] = max(cont, exercise)
    return vals[0]


def calculate_option_value(data):
    # Inputs
    opt_type = data["type"].lower()
    is_call = opt_type == "call"
    S0 = float(data["stock_price"])
    K = float(data["strike"])
    sigma = float(data["volatility"])
    r = _normalize_rate(data["interest_rate"])

    # Dates and T
    start_dt = datetime.combine(
        datetime.strptime(data["start_date"], "%Y-%m-%d").date(),
        _parse_time_flexible(data.get("start_time")),
    )
    exp_dt = datetime.combine(
        datetime.strptime(data["expiration_date"], "%Y-%m-%d").date(),
        _parse_time_flexible(data.get("expiration_time")),
    )
    T = _yearfrac(start_dt, exp_dt)

    # Steps
    steps = int(data.get("steps", 1000))
    steps = max(1, steps)

    # Dividends handling
    div = data.get("dividends", data.get("dividens", 0.0))
    if isinstance(div, list):
        q = 0.0
        S = max(S0 - _pv_dividends(div, start_dt, exp_dt, r), 1e-12)
    else:
        q = float(div)
        S = S0

    # Edge case T<=0
    if T <= 0:
        price = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        return {
            "theoretical_price": round(price, 3),
            "delta": 0.0,
            "gamma": 0.0,
            "rho": 0.0,
            "theta": 0.0,
            "vega": 0.0,
        }

    # Pricing function for bumps
    def price_fn(Sv, sigmav, rv, Tv):
        steps_local = max(1, steps)
        return _american_crr_price(Sv, K, rv, q, sigmav, max(Tv, 0.0), steps_local, is_call=is_call)

    # Base price
    base_price = price_fn(S, sigma, r, T)

    # Step sizes for Greeks
    h = max(0.01 * S, 1e-4)
    ds = max(0.01 * sigma, 1e-4)
    dr = 1e-4

    # Delta and Gamma (central differences)
    p_up = price_fn(S + h, sigma, r, T)
    p_dn = price_fn(max(S - h, 1e-12), sigma, r, T)
    delta = (p_up - p_dn) / (2 * h)
    gamma = (p_up - 2 * base_price + p_dn) / (h * h)

    # Vega (central diff) scaled per +1 vol point
    p_vs_up = price_fn(S, sigma + ds, r, T)
    p_vs_dn = price_fn(S, max(sigma - ds, 1e-12), r, T)
    vega = (p_vs_up - p_vs_dn) / (2 * ds)
    vega_pct = vega / 100.0  # per +1% vol

    # Rho (central diff) scaled per +1% rate
    p_r_up = price_fn(S, sigma, r + dr, T)
    p_r_dn = price_fn(S, sigma, max(r - dr, -1.0), T)
    rho = (p_r_up - p_r_dn) / (2 * dr)
    rho_pct = rho / 100.0

    # Theta per day: V(T_minus) - V(T)
    one_day = 1.0 / 365.0
    T_minus = max(T - one_day, 0.0)
    p_T_minus = price_fn(S, sigma, r, T_minus)
    theta_per_day = p_T_minus - base_price

    return {
        "theoretical_price": round(base_price, 3),
        "delta": round(delta, 3),
        "gamma": round(gamma, 3),
        "rho": round(rho_pct, 3),
        "theta": round(theta_per_day, 3),
        "vega": round(vega_pct, 3),
    }
