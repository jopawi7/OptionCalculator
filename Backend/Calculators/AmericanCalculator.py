from Backend.Calculators.Utils import *
import numpy as np
from datetime import datetime, time


# ---------------------------------------------------------
# Filename: AmericanCalculator.py
# LastUpdated: 2025-11-15
# Description: AmericanOptionCalculator
# ---------------------------------------------------------



def monte_carlo_american_option(S0, K, r, sigma, T, steps, N, is_call):
    """
    Price American option using Monte Carlo simulation with Longstaff-Schwartz algorithm.

    Args:
        S0: Initial stock price
        K: Strike price
        r: Risk-free rate (decimal)
        sigma: Volatility
        T: Time to maturity (years)
        steps: Number of time steps
        N: Number of simulation paths
        is_call: True for call, False for put

    Returns:
        Option price
    """
    if T <= 0:
        if is_call:
            return max(S0 - K, 0.0)
        else:
            return max(K - S0, 0.0)

    np.random.seed(42)  # For reproducibility

    dt = T / steps
    discount = np.exp(-r * dt)

    # Simulate price paths
    S = np.zeros((N, steps + 1))
    S[:, 0] = S0

    for t in range(1, steps + 1):
        Z = np.random.standard_normal(N)
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    # Calculate payoffs at maturity
    if is_call:
        payoff = np.maximum(S[:, -1] - K, 0)
    else:
        payoff = np.maximum(K - S[:, -1], 0)

    # Longstaff-Schwartz algorithm for early exercise
    # Work backwards through time
    for t in range(steps - 1, 0, -1):
        # Intrinsic value at time t
        if is_call:
            intrinsic = np.maximum(S[:, t] - K, 0)
        else:
            intrinsic = np.maximum(K - S[:, t], 0)

        # Find paths where early exercise is valuable (in the money)
        itm = intrinsic > 0

        if np.sum(itm) > 0:
            # Regression to estimate continuation value
            # Use polynomial basis functions: 1, S, S^2
            X = S[itm, t]
            Y = payoff[itm] * discount

            # Fit polynomial regression
            A = np.vstack([np.ones(len(X)), X, X ** 2]).T
            try:
                coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
                continuation_value = A @ coeffs

                # Exercise if intrinsic value > continuation value
                exercise = intrinsic[itm] > continuation_value

                # Update payoff for exercised paths
                payoff[itm] = np.where(exercise, intrinsic[itm], payoff[itm] * discount)
                # Update payoff for out-of-money paths
                payoff[~itm] *= discount
            except:
                # If regression fails, just use intrinsic vs discounted continuation
                payoff = np.maximum(intrinsic, payoff * discount)
        else:
            # No ITM paths, just discount
            payoff *= discount

    # Discount to present and average
    option_price = np.mean(payoff) * discount

    return option_price


def calculate_option_value(data):
    """
    Calculates the option value for an American option using Monte Carlo simulation.

    Args:
        data: JSON-Object with all necessary data for the calculation.
              Expected keys:
              - type: "call" or "put"
              - strike: Strike price
              - stock_price: Current stock price
              - interest_rate: Risk-free rate (can be in % or decimal)
              - volatility: Volatility
              - start_date: Start date (YYYY-MM-DD)
              - start_time: Start time (HH:MM, HH:MM:SS, AM, PM, etc.)
              - expiration_date: Expiration date (YYYY-MM-DD)
              - expiration_time: Expiration time
              - number_of_steps: Number of time steps (default 100)
              - number_of_simulations: Number of MC simulations (default 10000)
              - dividends: List of dividends (optional)

    Returns:
        dict: JSON-Object with the calculated values for the option.
    """

    # Parse Input to useable variables
    strike_price = float(data["strike"])
    risk_free_rate = normalize_interest_rate(data["interest_rate"])
    sigma = float(data["volatility"])

    # Calculate time to maturity using the utils function
    time_to_maturity = calculate_time_to_maturity(
        data["start_date"],
        data.get("start_time", ""),
        data["expiration_date"],
        data.get("expiration_time", "")
    )

    # Check if option type is call or put
    is_call = data["type"].lower() == "call"

    # Number of steps and simulations for Monte Carlo
    steps = int(data.get("number_of_steps", 100))
    steps = max(1, steps)

    # Number of simulation paths
    simulations = int(data.get("number_of_simulations", 10000))
    simulations = max(1000, simulations)

    # Dividends handling using the utils function
    pv_dividends = calculate_present_value_dividends(
        data.get("dividends", []),
        data["start_date"],
        data["expiration_date"],
        risk_free_rate
    )

    stock_price = max(
        float(data["stock_price"]) - pv_dividends,
        1e-12
    )

    # Edge case T<=0
    if time_to_maturity <= 0:
        price = max(stock_price - strike_price, 0.0) if is_call else max(strike_price - stock_price, 0.0)
        return {
            "theoretical_price": round(price, 3),
            "delta": 0.0,
            "gamma": 0.0,
            "rho": 0.0,
            "theta": 0.0,
            "vega": 0.0,
        }

    # Pricing function for bumps
    def price_fn(Sv, sigmav, rv, Tv):
        return monte_carlo_american_option(
            Sv, strike_price, rv, sigmav, max(Tv, 0.0), steps, simulations, is_call
        )

    # Base price
    base_price = price_fn(stock_price, sigma, risk_free_rate, time_to_maturity)

    # Step sizes for Greeks
    h = max(0.01 * stock_price, 1e-4)
    ds = max(0.01 * sigma, 1e-4)
    dr = 1e-4

    # Delta and Gamma (central differences)
    p_up = price_fn(stock_price + h, sigma, risk_free_rate, time_to_maturity)
    p_dn = price_fn(max(stock_price - h, 1e-12), sigma, risk_free_rate, time_to_maturity)
    delta = (p_up - p_dn) / (2 * h)
    gamma = (p_up - 2 * base_price + p_dn) / (h * h)

    # Vega (central diff) scaled per +1 vol point
    p_vs_up = price_fn(stock_price, sigma + ds, risk_free_rate, time_to_maturity)
    p_vs_dn = price_fn(stock_price, max(sigma - ds, 1e-12), risk_free_rate, time_to_maturity)
    vega = (p_vs_up - p_vs_dn) / (2 * ds)
    vega_pct = vega / 100.0  # per +1% vol

    # Rho (central diff) scaled per +1% rate
    p_r_up = price_fn(stock_price, sigma, risk_free_rate + dr, time_to_maturity)
    p_r_dn = price_fn(stock_price, sigma, max(risk_free_rate - dr, -1.0), time_to_maturity)
    rho = (p_r_up - p_r_dn) / (2 * dr)
    rho_pct = rho / 100.0

    # Theta per day: V(T_minus) - V(T)
    one_day = 1.0 / 365.0
    T_minus = max(time_to_maturity - one_day, 0.0)
    p_T_minus = price_fn(stock_price, sigma, risk_free_rate, T_minus)
    theta_per_day = p_T_minus - base_price

    return {
        "theoretical_price": round(base_price, 3),
        "delta": round(delta, 3),
        "gamma": round(gamma, 3),
        "rho": round(rho_pct, 3),
        "theta": round(theta_per_day, 3),
        "vega": round(vega_pct, 3),
    }


# Example usage
if __name__ == "__main__":
    # Example data matching the new format
    data = {
        "type": "call",
        "exercise_style": "american",
        "start_date": "2025-11-16",
        "start_time": "17:22:34",
        "expiration_date": "2026-04-30",
        "expiration_time": "AM",
        "strike": 100,
        "stock_price": 299,
        "volatility": 0.20,
        "interest_rate": 1.5,
        "number_of_steps": 100,
        "number_of_simulations": 10000,
        "dividends": [
            {"date": "2025-11-20", "amount": 1.0},
            {"date": "2025-11-21", "amount": 2.0}
        ]
    }

    result = calculate_option_value(data)
    print("Option Pricing Results:")
    print(f"Theoretical Price: {result['theoretical_price']}")
    print(f"Delta: {result['delta']}")
    print(f"Gamma: {result['gamma']}")
    print(f"Vega: {result['vega']}")
    print(f"Theta: {result['theta']}")
    print(f"Rho: {result['rho']}")



"""
Code for AmericanCalculator with binomial Model. Unfortunately we should use the monte carlo simulation.

def _american_binomial_calculation(stock_price, strike_price, risk_free_rate, dividend_yield, sigma, time_to_maturity, steps, is_call=True):
    stock_price = max(stock_price, 1e-12)
    if time_to_maturity <= 0:
        return max(stock_price - strike_price, 0.0) if is_call else max(strike_price - stock_price, 0.0)
    dt = time_to_maturity / steps
    if sigma <= 0 or dt <= 0:
        # No volatility: price equals intrinsic at start (no time value)
        return max(stock_price - strike_price, 0.0) if is_call else max(strike_price - stock_price, 0.0)
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    if abs(u - d) < 1e-14:
        return max(stock_price - strike_price, 0.0) if is_call else max(strike_price - stock_price, 0.0)
    disc = math.exp(-risk_free_rate * dt)
    drift = math.exp((risk_free_rate - dividend_yield) * dt)
    p = (drift - d) / (u - d)
    # clamp to [0,1] to avoid arbitrage issues from numerics
    p = max(0.0, min(1.0, p))

    # Terminal values
    vals = [0.0] * (steps + 1)
    S_ud = stock_price * (d ** steps)
    for j in range(steps + 1):
        S_T = S_ud * (u / d) ** j
        intrinsic = max(S_T - strike_price, 0.0) if is_call else max(strike_price - S_T, 0.0)
        vals[j] = intrinsic

    # Backward induction with early exercise
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            cont = disc * (p * vals[j + 1] + (1 - p) * vals[j])
            S_ij = stock_price * (u ** j) * (d ** (i - j))
            exercise = max(S_ij - strike_price, 0.0) if is_call else max(strike_price - S_ij, 0.0)
            vals[j] = max(cont, exercise)
    return vals[0]
"""