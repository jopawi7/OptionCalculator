from datetime import datetime, time
import numpy as np
from math import exp, sqrt, log, erf, pi

# ---------------------------------------------------------
# Filename: Utils.py
# LastUpdated: 2025-11-15
# Description: Some util functions that are needed for several calculations
# ---------------------------------------------------------

def parse_time_string(time_string: str):
    """
    Converts time string into Datetime format.
    Handles: (HH:MM), (HH:MM:SS), (AM/PM)

    Used in:
        - AmericanCalculator
        - EuropeanCalculator
    Args:
        time_string (str): time string to be parsed
    Returns:
        datetime.time object representing the parsed time
    """

    #If there is no value return midnight
    if not time_string:
        return time(0, 0)

    tstr = str(time_string).strip().upper()

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


def yearfrac(start_dt, end_dt):
    """Actual/365 year fraction."""
    return max((end_dt - start_dt).total_seconds() / (365 * 24 * 3600), 0.0)


def normalize_rate(x):
    """Treat 1.5 as 1.5%."""
    x = float(x)
    return x / 100.0 if x > 1 else x


def pv_dividends(div_list, start_dt, exp_dt, r):
    """PV of discrete dividends before expiry."""
    if not isinstance(div_list, list):
        return 0.0
    pv = 0.0
    for d in div_list:
        pay_date = datetime.strptime(d["date"], "%Y-%m-%d")
        if start_dt < pay_date < exp_dt:
            T = yearfrac(start_dt, pay_date)
            pv += float(d["amount"]) * np.exp(-r * T)
    return pv
