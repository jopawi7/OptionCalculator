from datetime import datetime
import numpy as np
from scipy.stats import norm
from datetime import datetime


#Relative Import needed for Frontend/Absolute for MainCalculator.py
try:
    from .Utils import *
except ImportError:
    from Utils import *

# ---------------------------------------------------------
# Filename: EuropeanCalculator.py
# Created: 2025-11-22
# Description: Calculates the value of European options based on Black-Scholes
# We deduct the present value of dividends from the present value of the underlying
# ---------------------------------------------------------


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
    strike = data["strike"]
    S = data["stock_price"]
    volatility = normalize_interest_rate_or_volatility(data['volatility'])
    interest_rate = normalize_interest_rate_or_volatility(data['interest_rate'])
    dividends_list = data.get("dividends", [])

    T = calculate_time_to_maturity(start_date, start_time, expiration_date, expiration_time)

    #The online calculator does only consider dividends if there are five or more dividends scheduled. Don´t ask why!
    #They also do not adapt the greeks to dividend payments so we use there the price from before deducting the PV of all dividens if number of dividends >=5.
    if len(dividends_list) < 5:
        stock_present_value = S
    else:
        stock_present_value = S - calculate_present_value_dividends(data["dividends"], start_date, expiration_date,interest_rate)

    # =====================================================
    # 2) BLACK-SCHOLES CORE COMPUTATION
    # =====================================================

    d1 = (np.log(S / strike) + (interest_rate + 0.5 * volatility ** 2) * T) / (volatility * np.sqrt(T))
    d2 = d1 - volatility * np.sqrt(T)

    if option_type == "call":
        price = stock_present_value * norm.cdf(d1) - strike * np.exp(-interest_rate * T) * norm.cdf(d2)
    else:
        price = strike * np.exp(-interest_rate * T) * norm.cdf(-d2) - stock_present_value * norm.cdf(-d1)

    delta, gamma, theta, vega, rho = calculate_greeks_without_dividend_payments(option_type, S, strike, interest_rate,volatility, T)

    # =====================================================
    # 3) RETURN RESULTS (MainCalculator.py handles printing)
    # =====================================================

    #Price can become negative because of discrete dividend treatment deduction
    return {
        "theoretical_price": round(0 if price <= 0 else price, 3),
        "delta": round(delta, 3),
        "gamma": round(gamma, 3),
        "rho": round(rho, 3),
        "theta": round(theta, 3),
        "vega": round(vega, 3),
}