from datetime import datetime
import numpy as np
from scipy.stats import norm
from datetime import datetime
from Utils import *




def calculate_option_value(data):
    """
    Calculate the price and Greeks of a European option (call or put)
    using the Black-Scholes model with dividend yield adjustment.

     Required keys in `data`:
      - type: "call" or "put"
      - exercise_style: should be "european"
      - start_date: "YYYY-MM-DD"
      - start_time: "HH:MM:SS" or "am"/"pm"
      - expiration_date: "YYYY-MM-DD"
      - expiration_time: "HH:MM:SS" or "am"/"pm"
      - strike: float
      - stock_price: float
      - volatility: float
      - interest_rate: float
      - dividends: list of {"date": "YYYY-MM-DD", "amount": float}
    """

    # =====================================================
    # 1) EXTRACT AND CALCULATE PARAMETERS
    # =====================================================
    option_type = data["type"]
    start_date = data["start_date"]
    start_time = data["start_time"]
    expiration_date = data["expiration_date"]
    expiration_time = data["expiration_time"]
    K = data["strike"]
    S0 = data["stock_price"]
    volatility = data["volatility"]
    interest_rate = data["interest_rate"]
    dividends_list = data.get("dividends", [])

    # Normalize volatility and interest rate (e.g. 20 → 0.20)
    sigma = normalize_interest_rate_or_volatility(volatility)
    r = normalize_interest_rate_or_volatility(interest_rate)

    # Time to maturity in years (Actual/365) using Utils
    T = calculate_time_to_maturity( start_date,start_time, expiration_date,expiration_time,
)

    # If we have less than 5 dividends, we dont´t consider them. WHY? Because the other calculator also doesnt consider them
    if len(dividends_list) < 5:
        q = 0.0
    else:
        pv_div = calculate_present_value_dividends(
            dividends_list,
            start_date,
            expiration_date,
            r,
        )
        q = calc_continuous_dividend_yield(
            stock_price=S0,
            pv_dividends=pv_div,
            time_to_maturity=T,
        )

    # =====================================================
    # 2) BLACK-SCHOLES CORE COMPUTATION (with dividend yield)
    # =====================================================

    sqrt_T = np.sqrt(T)

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == "call":
        price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (
                - (S0 * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * sqrt_T)
                + q * S0 * np.exp(-q * T) * norm.cdf(d1)
                - r * K * np.exp(-r * T) * norm.cdf(d2)
        )
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)

    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (
                - (S0 * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * sqrt_T)
                - q * S0 * np.exp(-q * T) * norm.cdf(-d1)
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
        )
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("Option type must be 'call' or 'put'.")

    gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S0 * sigma * sqrt_T)
    vega = S0 * np.exp(-q * T) * norm.pdf(d1) * sqrt_T


    # =====================================================
    # 3) RETURN RESULTS (Main.py handles printing)
    # =====================================================

    return {
        "option_price": float(round(price, 3)),
        "delta": float(round(delta, 3)),
        "gamma": float(round(gamma, 3)),
        "theta": float(round(theta, 3)),
        "vega": float(round(vega, 3)),
        "rho": float(round(rho, 3)),
    }