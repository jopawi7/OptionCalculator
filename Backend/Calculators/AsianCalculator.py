from __future__ import annotations
from typing import Dict, Any, Tuple
import math
import numpy as np
from .Utils import *


# ---------------------------------------------------------
# Filename: AsianCalculator.py
# Description:
#   Pricing and Greeks for an ASIAN option with dividends.
#   The main entry point is `calculate_option_value(data)`,
#   using the same data structure as the other calculators.
# ---------------------------------------------------------


def calculate_option_value(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute the theoretical price of an Asian option with dividends and greeks.

    Required keys in `data`:
      - type: "call" or "put"
      - exercise_style: should be "asian"
      - start_date: "YYYY-MM-DD"
      - start_time: "HH:MM:SS" or "AM"/"PM"
      - expiration_date: "YYYY-MM-DD"
      - expiration_time: "HH:MM:SS" or "AM"/"PM"
      - strike: float
      - stock_price: float
      - volatility: float
      - interest_rate: float
      - average_type: "arithmetic" or "geometric"
        - number_of_steps / n_fixings: int
      - number_of_simulations / mc_sims: int
      - dividends: list of {"date": "YYYY-MM-DD", "amount": float}
    """

    # =====================================================
    # 1) EXTRACT AND CALCULATE PARAMETERS
    # =====================================================
    option_type = data["type"]
    start_date = data["start_date"]
    start_time = data.get("start_time", "")
    expiration_date = data["expiration_date"]
    expiration_time = data.get("expiration_time", "")
    initial_stock_price = data["stock_price"]
    strike_price = data["strike"]
    volatility_raw = data["volatility"]
    interest_rate_raw = data["interest_rate"]
    average_price_type = data.get("average_type", "arithmetic")
    dividends_list = data.get("dividends", [])
    number_of_fixings = data.get("number_of_steps", 100)
    number_of_simulations = data.get("number_of_simulations", 10000)
    random_seed = 42

    # Time to maturity in years (ACT/365)
    time_to_maturity = calculate_time_to_maturity(start_date,start_time,expiration_date,expiration_time)


    # Normalize volatility and interest rate (percent or decimal)
    volatility = normalize_interest_rate_or_volatility(volatility_raw)
    risk_free_rate = normalize_interest_rate_or_volatility(interest_rate_raw)


    # Dividends: discrete list → PV → continuous yield q (always used for Asian)
    present_value_dividends = calculate_present_value_dividends( dividends_list, start_date, expiration_date,risk_free_rate)

    continuous_dividend_yield = calc_continuous_dividend_yield(initial_stock_price,present_value_dividends,time_to_maturity)

    # MC time step (if not provided, use uniform grid over T)
    mc_dt = time_to_maturity / max(number_of_fixings, 1)
    is_call_option = (option_type == "call")

    # =====================================================
    # 2) PRICING (GEOMETRIC CLOSED FORM OR ARITHMETIC MC)
    # =====================================================

    if average_price_type == "geometric":
        option_price = _asian_geometric_closed_form_price(
            initial_stock_price,
            strike_price,
            risk_free_rate,
            continuous_dividend_yield,
            volatility,
            time_to_maturity,
            number_of_fixings,
            is_call=is_call_option,
        )

        delta, gamma, theta, vega, rho = calculate_greeks_without_dividend_payments(option_type, initial_stock_price,strike_price, risk_free_rate, volatility, time_to_maturity)
    else:
        # Default: arithmetic average via Monte Carlo
        option_price = _asian_arithmetic_monte_carlo_price(
            initial_stock_price,
            strike_price,
            risk_free_rate,
            continuous_dividend_yield,
            volatility,
            time_to_maturity,
            number_of_fixings,
            number_of_simulations,
            mc_dt,
            random_seed,
            is_call=is_call_option,
            use_antithetic_variates=True,
            use_control_variate_technique=True,
        )

        delta, gamma, theta, vega, rho = calculate_greeks_without_dividend_payments(option_type, initial_stock_price, strike_price, risk_free_rate,volatility, time_to_maturity)

    # =====================================================
    # 3) RETURN RESULTS (Main.py handles printing)
    # =====================================================

    return {
        "theoretical_price": float(round(option_price, 3)),
        "delta": float(round(delta, 3)),
        "gamma": float(round(gamma, 3)),
        "theta": float(round(theta, 3)),
        "vega": float(round(vega, 3)),
        "rho": float(round(rho, 3)),
    }


# =========================================================
# Geometric closed-form pricing
# =========================================================

def _asian_geometric_closed_form_price(
    initial_stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    continuous_dividend_yield: float,
    volatility: float,
    time_to_maturity: float,
    number_of_fixings: int,
    is_call: bool = True,
) -> float:
    """
    Closed-form price for a geometric Asian option with equally spaced fixings
    under Black–Scholes dynamics with continuous dividend yield q.
    """
    try:
        n = max(number_of_fixings, 1)
        mu_g = (
            (risk_free_rate - continuous_dividend_yield)
            - 0.5 * volatility**2
        ) * (n + 1) / (2.0 * n)

        sigma_g = volatility * math.sqrt(
            (n + 1) * (2 * n + 1) / (6.0 * n**2)
        )

        S_g0 = initial_stock_price * math.exp(mu_g * time_to_maturity)
        sigma_eff = sigma_g * math.sqrt(time_to_maturity)

        if sigma_eff <= 0:
            intrinsic = max(S_g0 - strike_price, 0.0) if is_call else max(strike_price - S_g0, 0.0)
            return intrinsic

        d1 = (
            math.log(S_g0 / strike_price)
            + (risk_free_rate - continuous_dividend_yield + 0.5 * sigma_eff**2)
            * time_to_maturity
        ) / sigma_eff
        d2 = d1 - sigma_eff

        Nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
        Nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
        Nmd1 = 1.0 - Nd1
        Nmd2 = 1.0 - Nd2

        disc = math.exp(-risk_free_rate * time_to_maturity)
        div_disc = math.exp(-continuous_dividend_yield * time_to_maturity)

        if is_call:
            return disc * (S_g0 * div_disc / disc * Nd1 - strike_price * Nd2)
        else:
            return disc * (strike_price * Nmd2 - S_g0 * div_disc / disc * Nmd1)
    except Exception:
        return 0.0


# =========================================================
# Arithmetic Monte Carlo pricing
# =========================================================

def _asian_arithmetic_monte_carlo_price(
    initial_stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    continuous_dividend_yield: float,
    volatility: float,
    time_to_maturity: float,
    number_of_fixings: int,
    number_of_simulations: int,
    mc_dt: float,
    random_seed: int,
    is_call: bool,
    use_antithetic_variates: bool = True,
    use_control_variate_technique: bool = True,
) -> float:
    """
    Monte Carlo pricer for an arithmetic Asian option
    with continuous dividend yield q (no discrete jumps).
    """

    rng = np.random.default_rng(random_seed)

    dt = float(mc_dt)
    if dt <= 0.0:
        dt = time_to_maturity / max(number_of_fixings, 1)

    n_steps = max(1, int(math.ceil(time_to_maturity / dt)))
    dt = time_to_maturity / n_steps

    if number_of_fixings > 0:
        fixing_times = np.linspace(dt, time_to_maturity, num=number_of_fixings)
    else:
        fixing_times = np.array([time_to_maturity])

    if use_antithetic_variates:
        n_batches = max(number_of_simulations // 2, 1)
    else:
        n_batches = max(number_of_simulations, 1)

    disc = math.exp(-risk_free_rate * time_to_maturity)
    drift_dt = (risk_free_rate - continuous_dividend_yield - 0.5 * volatility**2) * dt
    vol_sqrt_dt = volatility * math.sqrt(dt)

    payoffs = []

    for _ in range(n_batches):
        z = rng.standard_normal(size=(1, n_steps))
        z_sets = [z]
        if use_antithetic_variates:
            z_sets.append(-z)

        for z_path in z_sets:

            S = initial_stock_price
            t = 0.0
            fix_idx = 0
            running_sum = 0.0

            for step_idx in range(n_steps):
                S *= math.exp(drift_dt + vol_sqrt_dt * float(z_path[0, step_idx]))
                t += dt

                while fix_idx < len(fixing_times) and t + 1e-12 >= fixing_times[fix_idx]:
                    running_sum += S
                    fix_idx += 1

            while fix_idx < len(fixing_times):
                running_sum += S
                fix_idx += 1

            avg_S = running_sum / max(number_of_fixings, 1)

            payoff = max(avg_S - strike_price, 0.0) if is_call else max(strike_price - avg_S, 0.0)
            payoffs.append(payoff)

    crude_estimate = disc * float(np.mean(payoffs))

    if use_control_variate_technique:
        # Hook for a proper control variate using geometric Asian
        geo_price = _asian_geometric_closed_form_price(
            initial_stock_price,
            strike_price,
            risk_free_rate,
            continuous_dividend_yield,
            volatility,
            time_to_maturity,
            number_of_fixings,
            is_call=is_call,
        )
        # Currently not adjusting with control variate (could be implemented later)
        _ = geo_price

    return crude_estimate
