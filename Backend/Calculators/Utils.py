from datetime import datetime, time
import numpy as np

# ---------------------------------------------------------
# Filename: Utils.py
# LastUpdated: 2025-11-16
# Description: Utility functions for calculations
# ---------------------------------------------------------

def calculate_year_fraction(start_dt, end_dt):
    """
    Calculates the fraction of a year between two dates using the Actual/365 convention.
    """
    return max((end_dt - start_dt).total_seconds() / (365 * 24 * 3600), 0.0)

def normalize_interest_rate(x):
    """
    1.5 -> 0.015 ; -1.5 -> -0.015
    0.05 stays 0.05 ; -0.005 stays -0.005
    """
    x = float(x)
    return x / 100.0 if abs(x) > 1 else x


def parse_time_string(time_string: str):
    if not time_string:
        return time(0, 0, 0)
    tstr = str(time_string).strip().lower()
    if tstr == "am":
        return time(9, 30, 0)   # or whatever mapping you intend
    if tstr == "pm":
        return time(15, 30, 0)  # ditto

    try:
        return datetime.strptime(tstr, "%H:%M:%S").time()
    except ValueError as e:
        raise ValueError(
            f"Invalid time string: {time_string!r}. Expected 'HH:MM:SS' or 'am'/'pm'."
        ) from e


def calculate_present_value_dividends(dividend_list, start_date, expiry_date, risk_free_rate):
    """
    Calculates the present value of dividends between start_date and expiry_date.
    """

    # Convert start_date and expiry_date to datetime if they are strings
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(expiry_date, str):
        expiry_date = datetime.strptime(expiry_date, "%Y-%m-%d")

    if not isinstance(dividend_list, list):
        return 0.0

    present_value = 0.0
    for single_dividend in dividend_list:
        pay_date = datetime.strptime(single_dividend["date"], "%Y-%m-%d")
        if start_date < pay_date < expiry_date:
            T = calculate_year_fraction(start_date, pay_date)
            present_value += float(single_dividend["amount"]) * np.exp(-risk_free_rate * T)
    return present_value

def calc_continuous_dividend_yield(stock_price, pv_dividends, time_to_maturity):
    # Falls keine Dividenden oder Laufzeit sehr kurz, ist die Rendite null
    if pv_dividends < 1e-12 or time_to_maturity < 1e-12:
        return 0.0
    try:
        q = - (1.0 / time_to_maturity) * np.log((stock_price - pv_dividends) / stock_price)
    except Exception:
        q = 0.0
    return q


def calculate_time_to_maturity(start_date, start_time, expire_date, expire_time):
    """
    Calculate time to maturity in years based on start_date, start_time, expire_date, expire_time.
    Throws ValueError if expiration datetime is not after start datetime.
    """
    start_dt = datetime.combine(
        datetime.strptime(start_date, "%Y-%m-%d").date(),
        parse_time_string(start_time)
    )
    expire_dt = datetime.combine(
        datetime.strptime(expire_date, "%Y-%m-%d").date(),
        parse_time_string(expire_time)
    )
    if expire_dt <= start_dt:
        raise ValueError("Expiration datetime must be after start datetime.")
    return calculate_year_fraction(start_dt, expire_dt)


