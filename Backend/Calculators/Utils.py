from datetime import datetime, time
import numpy as np
from typing import Dict, Any, Tuple
from scipy.stats import norm

# ---------------------------------------------------------
# Filename: Utils.py
# LastUpdated: 2025-11-22
# Description: Utility functions for calculations
# ---------------------------------------------------------

#calculates the fraction of a year depending on start date and end date
def calculate_year_fraction(start_daytime, end_daytime):
    """
    Calculates the fraction of a year between two dates using the Actual/365 convention.
    """
    return max((end_daytime - start_daytime).total_seconds() / (365 * 24 * 3600), 0.0)

#normalizes interest rate
def normalize_interest_rate_or_volatility(x):
    """
    1.5 -> 0.015 ; -1.5 -> -0.015
    0.05 stays 0.05 ; -0.005 stays -0.005
    """
    x = float(x)
    return x / 100.0 if abs(x) > 1 else x

#Parse am/pm to daytime
def parse_time_string(time_string: str):
    if not time_string:
        return time(0, 0, 0)
    tstr = str(time_string).strip().lower()
    if tstr == "am":
        return time(9, 30, 0)
    if tstr == "pm":
        return time(16, 00, 0)  # ditto

    try:
        return datetime.strptime(tstr, "%H:%M:%S").time()
    except ValueError as e:
        raise ValueError(
            f"Invalid time string: {time_string!r}. Expected 'HH:MM:SS' or 'am'/'pm'."
        ) from e


#Calculates the present value of all discrete dividend payments
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


#Calculates continous dividend yield q
def calc_continuous_dividend_yield(stock_price, pv_dividends, time_to_maturity):
# If there are effectively no dividends or time to maturity is very short, set the yield to zero
    if pv_dividends < 1e-12 or time_to_maturity < 1e-12:
        return 0.0
    try:
        q = - (1.0 / time_to_maturity) * np.log((stock_price - pv_dividends) / stock_price)
    except Exception:
        q = 0.0
    return q


#Calculates time to maturity
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


#Function to calculate greeks wihtout dividend payments
def calculate_greeks_without_dividend_payments(option_type: str,S: float, strike: float, interest_rate: float,volatility: float,T: float) -> Tuple[float, float, float, float, float]:
    d1 = (np.log(S / strike) + (interest_rate + 0.5 * volatility ** 2) * T) / (volatility * np.sqrt(T))
    d2 = d1 - volatility * np.sqrt(T)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (- (S * volatility * norm.pdf(d1)) / (2 * np.sqrt(T))
                 - interest_rate * strike * np.exp(-interest_rate * T) * norm.cdf(d2))
        rho = strike * T * np.exp(-interest_rate * T) * norm.cdf(d2)
    else:
        delta = -norm.cdf(-d1)
        theta = (- (S * volatility * norm.pdf(d1)) / (2 * np.sqrt(T))
                 + interest_rate * strike * np.exp(-interest_rate * T) * norm.cdf(-d2))
        rho = -strike * T * np.exp(-interest_rate * T) * norm.cdf(-d2)

    gamma = norm.pdf(d1) / (S * volatility * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return float(delta), float(gamma), float(theta), float(vega), float(rho)