from __future__ import annotations

from typing import Any, Dict
import numpy as np
from scipy.stats import norm

from Utils import (
    calculate_time_to_maturity,
    normalize_interest_rate_or_volatility,
    calculate_present_value_dividends,
    calc_continuous_dividend_yield,
)


def calculate_option_value(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate the price and Greeks of a European option (call or put)
    using the Black-Scholes model with dividend yield adjustment.
    """

    # =====================================================
    # 1) DISPLAY INPUTS
    # =====================================================

    # Extract raw input values
    option_type = data["type"].lower()
    exercise_style = data["exercise_style"]
    start_date = data["start_date"]
    start_time = data["start_time"]
    expiration_date = data["expiration_date"]
    expiration_time = data["expiration_time"]
    stock_price_raw = float(data["stock_price"])
    strike_price_raw = float(data["strike"])
    volatility_raw = float(data["volatility"])
    interest_rate_raw = float(data["interest_rate"])
    dividends_list = data.get("dividends", [])

    # Normalize volatility and interest rate (e.g. 20 → 0.20)
    sigma = normalize_interest_rate_or_volatility(volatility_raw)
    r = normalize_interest_rate_or_volatility(interest_rate_raw)

    print("\n=== EUROPEAN OPTION INPUTS ===")
    input_fields = {
        "Option type": option_type,
        "Exercise style": exercise_style,
        "Start date": start_date,
        "Start time": start_time,
        "Expiration date": expiration_date,
        "Expiration time": expiration_time,
        "Stock price (S0)": stock_price_raw,
        "Strike price (K)": strike_price_raw,
        "Volatility (raw input)": volatility_raw,
        "Volatility (normalized)": sigma,
        "Interest rate (raw)": interest_rate_raw,
        "Interest rate (normalized)": r,
        "Dividends": dividends_list,
    }
    for key, value in input_fields.items():
        print(f"{key:25s}: {value}")

    # =====================================================
    # 2) EXTRACT AND CALCULATE PARAMETERS
    # =====================================================

    S0 = stock_price_raw
    K = strike_price_raw

    # Time to maturity in years (Actual/365) using Utils
    T = calculate_time_to_maturity(
        start_date,
        start_time,
        expiration_date,
        expiration_time,
    )

    # If we have less than 5 dividends, we don't count them
    if len(dividends_list) < 5:
        pv_div = 0.0
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
    # 3) BLACK-SCHOLES CORE COMPUTATION (with dividend yield)
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
    # 4) RETURN RESULTS (Main.py handles printing)
    # =====================================================

    return {
        "option_price": float(round(price, 6)),
        "delta": float(round(delta, 6)),
        "gamma": float(round(gamma, 6)),
        "theta": float(round(theta, 6)),
        "vega": float(round(vega, 6)),
        "rho": float(round(rho, 6)),
    }
