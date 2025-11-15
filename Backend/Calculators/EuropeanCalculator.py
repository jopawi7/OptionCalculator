import scipy.stats as si
import numpy as np
from datetime import datetime, time

# ---------------------------------------------------------
# Filename: EuropeanCalculator.py
# Author:
# Created: 2025-10-30
# Description: European option pricer (Black–Scholes) with automatic unit handling
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


def calculate_option_value(data):
    # Extract and normalize
    opt_type = data["type"].lower()
    S0 = float(data["stock_price"])
    K = float(data["strike"])
    sigma = float(data["volatility"])
    r = _normalize_rate(data["interest_rate"])

    start_dt = datetime.combine(
        datetime.strptime(data["start_date"], "%Y-%m-%d").date(),
        _parse_time_flexible(data.get("start_time")),
    )
    exp_dt = datetime.combine(
        datetime.strptime(data["expiration_date"], "%Y-%m-%d").date(),
        _parse_time_flexible(data.get("expiration_time")),
    )
    T = _yearfrac(start_dt, exp_dt)

    # Handle dividends (continuous or discrete)
    div = data.get("dividends", data.get("dividens", 0.0))
    if isinstance(div, list):
        q = 0.0
        S = S0 - _pv_dividends(div, start_dt, exp_dt, r)
    else:
        q = float(div)
        S = S0

    # Black–Scholes core
    if T <= 0 or sigma <= 0:
        return {"theoretical_price": 0, "delta": 0, "gamma": 0, "rho": 0, "theta": 0, "vega": 0}

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    Nd1, Nd2 = si.norm.cdf(d1), si.norm.cdf(d2)
    nd1 = si.norm.pdf(d1)

    df_r, df_q = np.exp(-r * T), np.exp(-q * T)

    if opt_type == "call":
        price = S * df_q * Nd1 - K * df_r * Nd2
        delta = df_q * Nd1
        theta = (
            -S * df_q * nd1 * sigma / (2 * np.sqrt(T))
            - r * K * df_r * Nd2
            + q * S * df_q * Nd1
        )
        rho = K * T * df_r * Nd2
    else:
        price = K * df_r * si.norm.cdf(-d2) - S * df_q * si.norm.cdf(-d1)
        delta = df_q * (Nd1 - 1)
        theta = (
            -S * df_q * nd1 * sigma / (2 * np.sqrt(T))
            + r * K * df_r * si.norm.cdf(-d2)
            - q * S * df_q * si.norm.cdf(-d1)
        )
        rho = -K * T * df_r * si.norm.cdf(-d2)

    gamma = df_q * nd1 / (S * sigma * np.sqrt(T))
    vega = S * df_q * nd1 * np.sqrt(T)

    # Convert to standard "per 1% change" outputs (as in most option calculators)
    vega_pct = vega / 100.0       # per +1% vol
    rho_pct = rho / 100.0         # per +1% rate
    theta_per_day = theta / 365.0 # per day

    return {
        "theoretical_price": round(price, 3),
        "delta": round(delta, 3),
        "gamma": round(gamma, 3),
        "rho": round(rho_pct, 3),
        "theta": round(theta_per_day, 3),
        "vega": round(vega_pct, 3),
    }
