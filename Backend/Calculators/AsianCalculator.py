# ---------------------------------------------------------
# Description
#   Pricing and Greeks for an ASIAN option.
#   The function `calculate_option_value` expects a `data`
#   dictionary similar to the CBOE option calculator inputs,
#   plus a few extra fields for Asian options and Monte Carlo.
#
# Notes
#   - This module assumes `data` has already been loaded and
#     validated elsewhere (for example, from a file).
#   - The goal is to keep the pricing logic explicit and
#     readable, not to hide steps behind abstractions.
# ---------------------------------------------------------
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
from Backend.Calculators.Utils import *


def calculate_option_value(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute the theoretical price and Greeks of an Asian option.

    The input `data` should contain:
    - type: "call" or "put"
    - exercise_style: "asian" (currently)
    - start_date: "YYYY-MM-DD"
    - start_time: "HH:MM:SS" or "AM"/"PM"
    - expiration_date: "YYYY-MM-DD"
    - expiration_time: "HH:MM:SS" or "AM"/"PM"
    - strike: float
    - stock_price: float
    - volatility: float (0.20 for 20%)
    - interest_rate: float (1.5 for 1.5%)
    - average_type: "arithmetic" or "geometric" (default: "arithmetic")
    - number_of_steps: int (MC time steps)
    - number_of_simulations: int (MC paths)
    - dividends: list of {date, amount} (optional)

    Returns:
    Dictionary with theoretical_price, delta, gamma, rho, theta, vega
    """

    # ========== 1. Parse dates and times using Utils.py ==========
    time_to_maturity = calculate_time_to_maturity(
        data["start_date"],
        data.get("start_time", ""),
        data["expiration_date"],
        data.get("expiration_time", "")
    )

    if time_to_maturity <= 0:
        # Option expired or invalid
        intrinsic = max(
            data["stock_price"] - data["strike"], 0.0
        ) if data["type"].lower() == "call" else max(
            data["strike"] - data["stock_price"], 0.0
        )
        return {
            "theoretical_price": round(intrinsic, 3),
            "delta": 0.0,
            "gamma": 0.0,
            "rho": 0.0,
            "theta": 0.0,
            "vega": 0.0,
        }

    # ========== 2. Normalize parameters using Utils.py ==========
    risk_free_rate = normalize_interest_rate(data["interest_rate"])
    sigma = float(data["volatility"])  # Already in decimal form (0.20)
    strike_price = float(data["strike"])

    # Adjust stock price for PV of dividends
    pv_div = calculate_present_value_dividends(
        data.get("dividends", []),
        data["start_date"],
        data["expiration_date"],
        risk_free_rate
    )
    stock_price = max(float(data["stock_price"]) - pv_div, 1e-12)

    # ========== 3. Asian option parameters ==========
    is_call = data["type"].lower() == "call"
    average_type = data.get("average_type", "arithmetic").lower()
    number_of_steps = int(data.get("number_of_steps", 100))
    number_of_simulations = int(data.get("number_of_simulations", 10000))

    # ========== 4. Price the option ==========
    if average_type == "geometric":
        base_price = _asian_geometric_price(
            stock_price, strike_price, risk_free_rate, sigma,
            time_to_maturity, number_of_steps, number_of_simulations,
            is_call
        )
    else:  # arithmetic (default)
        base_price = _asian_arithmetic_monte_carlo_price(
            stock_price, strike_price, risk_free_rate, sigma,
            time_to_maturity, number_of_steps, number_of_simulations,
            is_call
        )

    # ========== 5. Calculate Greeks using finite differences ==========
    h = max(0.01 * stock_price, 1e-4)
    ds = max(0.01 * sigma, 1e-4)
    dr = 1e-4
    dt_bump = 1.0 / 365.0

    # Delta and Gamma
    def price_fn(S, sig, r, T):
        if average_type == "geometric":
            return _asian_geometric_price(
                S, strike_price, r, sig, T, number_of_steps,
                number_of_simulations, is_call
            )
        else:
            return _asian_arithmetic_monte_carlo_price(
                S, strike_price, r, sig, T, number_of_steps,
                number_of_simulations, is_call
            )

    p_up = price_fn(stock_price + h, sigma, risk_free_rate, time_to_maturity)
    p_dn = price_fn(max(stock_price - h, 1e-12), sigma, risk_free_rate, time_to_maturity)
    delta = (p_up - p_dn) / (2 * h)
    gamma = (p_up - 2 * base_price + p_dn) / (h * h)

    # Vega
    p_vs_up = price_fn(stock_price, sigma + ds, risk_free_rate, time_to_maturity)
    p_vs_dn = price_fn(stock_price, max(sigma - ds, 1e-12), risk_free_rate, time_to_maturity)
    vega = (p_vs_up - p_vs_dn) / (2 * ds) / 100.0  # per 1% vol

    # Rho
    p_r_up = price_fn(stock_price, sigma, risk_free_rate + dr, time_to_maturity)
    p_r_dn = price_fn(stock_price, sigma, max(risk_free_rate - dr, -1.0), time_to_maturity)
    rho = (p_r_up - p_r_dn) / (2 * dr) / 100.0  # per 1% rate

    # Theta (per day)
    T_minus = max(time_to_maturity - dt_bump, 0.0)
    p_T_minus = price_fn(stock_price, sigma, risk_free_rate, T_minus)
    theta = p_T_minus - base_price

    # ========== 6. Return results ==========
    return {
        "theoretical_price": round(base_price, 3),
        "delta": round(delta, 3),
        "gamma": round(gamma, 3),
        "rho": round(rho, 3),
        "theta": round(theta, 3),
        "vega": round(vega, 3),
    }


def _asian_arithmetic_monte_carlo_price(S0, K, r, sigma, T, steps, N, is_call):
    """
    Price arithmetic average Asian option using Monte Carlo.
    """
    np.random.seed(42)

    dt = T / steps
    discount = np.exp(-r * T)

    # Generate stock price paths
    S = np.zeros((N, steps + 1))
    S[:, 0] = S0

    for t in range(1, steps + 1):
        Z = np.random.standard_normal(N)
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    # Calculate arithmetic average for each path
    average_prices = np.mean(S, axis=1)

    # Calculate payoff at maturity
    if is_call:
        payoff = np.maximum(average_prices - K, 0.0)
    else:
        payoff = np.maximum(K - average_prices, 0.0)

    # Expected value (discounted)
    option_price = discount * np.mean(payoff)

    return option_price


def _asian_geometric_price(S0, K, r, sigma, T, steps, N, is_call):
    """
    Price geometric average Asian option.
    Uses Kemna-Vorst closed-form approximation.
    """
    # Adjusted parameters for geometric average
    sigma_g = sigma / np.sqrt(3)
    mu_g = (r - 0.5 * sigma ** 2) / 2 + sigma_g ** 2 / 6

    # Use Black-Scholes with adjusted parameters
    d1 = (np.log(S0 / K) + (mu_g + 0.5 * sigma_g ** 2) * T) / (sigma_g * np.sqrt(T))
    d2 = d1 - sigma_g * np.sqrt(T)

    from scipy.stats import norm

    if is_call:
        price = (
                S0 * np.exp(mu_g * T) * norm.cdf(d1) -
                K * np.exp(-r * T) * norm.cdf(d2)
        )
    else:
        price = (
                K * np.exp(-r * T) * norm.cdf(-d2) -
                S0 * np.exp(mu_g * T) * norm.cdf(-d1)
        )

    return max(price, 0.0)


if __name__ == "__main__":
    # Test with example data
    test_data = {
        "type": "call",
        "exercise_style": "asian",
        "start_date": "2025-11-16",
        "start_time": "17:22:34",
        "expiration_date": "2026-04-30",
        "expiration_time": "AM",
        "strike": 100,
        "stock_price": 299,
        "volatility": 0.20,
        "interest_rate": 1.5,
        "average_type": "arithmetic",
        "number_of_steps": 100,
        "number_of_simulations": 10000,
        "dividends": [
            {"date": "2025-11-20", "amount": 1.0},
            {"date": "2025-11-21", "amount": 2.0}
        ]
    }

    result = calculate_option_value(test_data)
    print("Asian Option Pricing Results:")
    print(f"Theoretical Price: {result['theoretical_price']}")
    print(f"Delta: {result['delta']}")
    print(f"Gamma: {result['gamma']}")
    print(f"Vega: {result['vega']}")
    print(f"Theta: {result['theta']}")
    print(f"Rho: {result['rho']}")
