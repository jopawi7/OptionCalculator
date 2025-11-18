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


def year_fraction_with_exact_days(start_date, start_time, expiration_date, expiration_time):
    start_str = f"{start_date} {start_time or '00:00:00'}"

    # Setze Stunden und Minuten getrennt als Strings bzw. ints
    if expiration_time and expiration_time.lower() == 'pm':
        exp_time_str = '16:00:00'
    else:
        exp_time_str = '09:30:00'

    expiration_str = f"{expiration_date} {exp_time_str}"

    start_dt = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
    exp_dt = datetime.strptime(expiration_str, '%Y-%m-%d %H:%M:%S')

    delta_days = (exp_dt - start_dt).days + (exp_dt - start_dt).seconds / (24 * 3600)
    year_basis = 365  # oder 365.25 für Schaltjahre

    return delta_days / year_basis


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

    T = year_fraction_with_exact_days(start_date, start_time, expiration_date, expiration_time)

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