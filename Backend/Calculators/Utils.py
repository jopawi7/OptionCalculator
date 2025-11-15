from datetime import datetime, time
import numpy as np
from math import exp, sqrt, log, erf, pi

# ---------------------------------------------------------
# Filename: Utils.py
# LastUpdated: 2025-11-15
# Description: Some util functions that are needed for several calculations
# ---------------------------------------------------------

def calculate_year_fraction(start_dt, end_dt):
    """
    Calculates the fraction of a year between two dates using the Actual/365 convention.
    """
    return max((end_dt - start_dt).total_seconds() / (365 * 24 * 3600), 0.0)

def normalize_interest_rate(x):
    """
    Treats an input value like 1.5 as 1.5%, converting it to a decimal (0.015) if above 1; leaves values ≤ 1 unchanged.
    """
    x = float(x)
    return x / 100.0 if x > 1 else x


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

def calculate_present_value_dividends(dividend_list, start_date, expiry_date, risk_free_rate):
    """
    Calculates the present value of dividends between start_date and expiry_date.

    Used in:
        - AmericanCalculator
        - EuropeanCalculator
    Args:
        dividend_list (list): list of dividends to be calculated
        start_date (datetime): start date of the option
        expiry_date (datetime): expiry date of the option
        risk_free_rate (float): risk free rate of the underlying asset
    Returns:
        present_value (float): present value of dividends between start_date and expiry_date
    """
    #If dividendList is invalid
    if not isinstance(dividend_list, list):
        return 0.0

    present_value = 0.0

    for single_dividend in dividend_list:
        #extract date of single dividend
        pay_date = datetime.strptime(single_dividend["date"], "%Y-%m-%d")

        #Sum over present values
        if start_date < pay_date < expiry_date:
            T = calculate_year_fraction(start_date, pay_date)
            present_value += float(single_dividend["amount"]) * np.exp(-risk_free_rate * T)

    return present_value





def calculate_time_to_maturity(start_date, start_time, expire_date, expire_time):
    """Calculate time to maturity in years based on start_date, start_time, expire_date, expire_time."""
    start_date = datetime.combine(
        datetime.strptime(start_date, "%Y-%m-%d").date(),
        parse_time_string(start_time)
    )

    expire_date = datetime.combine(
        datetime.strptime(expire_date, "%Y-%m-%d").date(),
        parse_time_string(expire_time)
    )

    return calculate_year_fraction(start_date, expire_date)




