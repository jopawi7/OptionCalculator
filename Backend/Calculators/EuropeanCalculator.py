import numpy as np
import scipy.stats as si
from datetime import datetime
from Utils import *

# ---------------------------------------------------------
# Filename: EuropeanCalculator.py
# LastUpdated: 2025-11-15
# Description: Calculate the value of European options based on Black-Scholes
# ---------------------------------------------------------

def calculate_option_value(data):
    """
    Calculates the option value for a European option with fixed maturity based on Black-Scholes. For better performance we use floats instead of precise decimal numbers and accept that this might lead to small deviations.

    Used in:
        - EuropeanCalculator

    Args:
        data: JSON-Object with all necessary data for the calculation.

    Returns:
        data: JSON-Object with the calculated values for the option.
    """

    #Parse Input to useable variables
    stock_price = float(data["stock_price"])
    strike_price  = float(data["strike"])
    risk_free_rate = normalize_rate(data["interest_rate"])
    sigma = float(data["volatility"])
    time_to_maturity = calculate_time_to_maturity(data["start_date"], data.get("start_time"), data["expiration_date"], data.get("expiration_time"))



    # Dividenden
    div = data.get("dividends", data.get("dividens", 0.0))
    if isinstance(div, list):
        S = max(stock_price - pv_dividends(div, start_dt, exp_dt, risk_free_rate), 1e-12)
        q = 0.0
    else:
        S = stock_price
        q = float(div)

    # Grenzfälle
    if time_to_maturity <= 0.0 or sigma <= 0.0 or S <= 0.0 or strike_price <= 0.0:
        df_r = np.exp(-risk_free_rate * time_to_maturity)
        df_q = np.exp(-q * time_to_maturity)
        if data["type"] == "call":
            price = max(S * df_q - strike_price * df_r, 0.0)
        elif data["type"] == "put":
            price = max(strike_price * df_r - S * df_q, 0.0)
        else:
            raise ValueError("type must be 'call' or 'put'")
        return {
            "theoretical_price": round(float(price), 3),
            "delta": 0.0,
            "gamma": 0.0,
            "rho": 0.0,
            "theta": 0.0,
            "vega": 0.0
        }






    # Black-Scholes Berechnung
    sqrtT = np.sqrt(time_to_maturity)
    d1 = (np.log(S / strike_price) + (risk_free_rate - q + 0.5 * sigma * sigma) * time_to_maturity) / (sigma * sqrtT)
    d2 = d1 - sigma*sqrtT
    Nd1, Nd2 = si.norm.cdf(d1), si.norm.cdf(d2)
    nd1 = si.norm.pdf(d1)
    df_r = np.exp(-risk_free_rate * time_to_maturity)
    df_q = np.exp(-q * time_to_maturity)

    if data["type"] == "call":
        price = S * df_q * Nd1 - strike_price * df_r * Nd2
        delta = df_q*Nd1
        theta = -(S*df_q*nd1*sigma) / (2*sqrtT) - risk_free_rate * strike_price * df_r * Nd2 + q * S * df_q * Nd1
        rho   = strike_price * time_to_maturity * df_r * Nd2
    elif data["type"] == "put":
        Nmd1, Nmd2 = si.norm.cdf(-d1), si.norm.cdf(-d2)
        price = strike_price * df_r * Nmd2 - S * df_q * Nmd1
        delta = df_q*(Nd1 - 1.0)
        theta = -(S*df_q*nd1*sigma) / (2*sqrtT) + risk_free_rate * strike_price * df_r * Nmd2 - q * S * df_q * Nmd1
        rho   = -strike_price * time_to_maturity * df_r * Nmd2
    else:
        raise ValueError("type must be 'call' or 'put'")

    gamma = (df_q*nd1)/(S*sigma*sqrtT)
    vega  = S*df_q*nd1*sqrtT

    # Greeks OHNE Skalierung - raw values
    out = {
        "theoretical_price": round(float(price), 3),
        "delta": round(float(delta), 3),
        "gamma": round(float(gamma), 3),
        "rho": round(float(rho), 3),
        "theta": round(float(theta), 3),
        "vega": round(float(vega), 3),
    }
    return out