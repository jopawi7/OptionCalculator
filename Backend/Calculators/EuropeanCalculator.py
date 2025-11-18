from datetime import datetime
import numpy as np
from scipy.stats import norm
from dateutil.parser import parse

# ---------------------------------------------------------
# Filename: EuropeanCalculator.py
# LastUpdated: 2025-11-16
# Our Model assumes constant interest rates and no forward interest rates compared to Cboe
# Description: Calculate the value of European options based on Black-Scholes
# ---------------------------------------------------------


def year_fraction_with_time(start_date, start_time, expiration_date, expiration_time):
    start_datetime_str = start_date + ' ' + start_time
    exp_time_hour = 12 if expiration_time.lower() == 'pm' else 0
    expiration_datetime_str = expiration_date + f' {exp_time_hour}:00:00'

    start_dt = parse(start_datetime_str)
    expiration_dt = parse(expiration_datetime_str)

    delta = expiration_dt - start_dt
    return delta.total_seconds() / (365.25 * 24 * 3600)  # berücksichtigt durchschnittliche Schaltjahre

def calculate_option_value(data):
    option_type = data["type"]
    start_date = data["start_date"]
    start_time = data["start_time"]
    expiration_date = data["expiration_date"]
    expiration_time = data["expiration_time"]
    strike = data["strike"]
    stock_price = data["stock_price"]
    volatility = data["volatility"]
    interest_rate = data["interest_rate"] / 100.0

    T = year_fraction_with_time(start_date, start_time, expiration_date, expiration_time)

    S = stock_price  # Keine Dividendenanpassung

    d1 = (np.log(S / strike) + (interest_rate + 0.5 * volatility ** 2) * T) / (volatility * np.sqrt(T))
    d2 = d1 - volatility * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - strike * np.exp(-interest_rate * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (- (S * volatility * norm.pdf(d1)) / (2 * np.sqrt(T))
                 - interest_rate * strike * np.exp(-interest_rate * T) * norm.cdf(d2))
    else:
        price = strike * np.exp(-interest_rate * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
        theta = (- (S * volatility * norm.pdf(d1)) / (2 * np.sqrt(T))
                 + interest_rate * strike * np.exp(-interest_rate * T) * norm.cdf(-d2))

    gamma = norm.pdf(d1) / (S * volatility * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = strike * T * np.exp(-interest_rate * T) * norm.cdf(d2 if option_type == "call" else -d2)

    return {
        "theoretical_price": round(price, 3),
        "delta": round(delta, 3),
        "gamma": round(gamma, 3),
        "rho": round(rho, 3),
        "theta": round(theta, 3),
        "vega": round(vega, 3),
    }