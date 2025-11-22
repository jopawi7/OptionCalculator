from __future__ import annotations

from typing import Any, Dict, List, Tuple
from datetime import datetime

import numpy as np

from Utils import (
    calculate_time_to_maturity,
    normalize_interest_rate_or_volatility,
    calculate_year_fraction,
)


DATE_FMT = "%Y-%m-%d"


def calculate_option_value(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Price an American option (call or put) using Monte Carlo simulation
    with Longstaff–Schwartz for early exercise and **discrete dividend jumps**.

    Expected keys in `data` (same style as other calculators):
      - type: "call" or "put"
      - exercise_style: "american"
      - start_date, start_time
      - expiration_date, expiration_time
      - strike, stock_price
      - volatility        (can be % or decimal)
      - interest_rate     (can be % or decimal)
      - number_of_steps
      - number_of_simulations
      - dividends: list of {"date": "YYYY-MM-DD", "amount": float}
    """

    # =====================================================
    # 1) DISPLAY INPUTS
    # =====================================================

    option_type = data["type"].lower()
    exercise_style = data["exercise_style"]
    start_date = data["start_date"]
    start_time = data.get("start_time", "")
    expiration_date = data["expiration_date"]
    expiration_time = data.get("expiration_time", "")
    stock_price_raw = float(data["stock_price"])
    strike_price_raw = float(data["strike"])
    volatility_raw = float(data["volatility"])
    interest_rate_raw = float(data["interest_rate"])
    dividends_list = data.get("dividends", [])

    # Normalize volatility and interest rate (e.g. 25 → 0.25)
    sigma = normalize_interest_rate_or_volatility(volatility_raw)
    r = normalize_interest_rate_or_volatility(interest_rate_raw)

    number_of_steps = int(data.get("number_of_steps", 50))
    number_of_simulations = int(data.get("number_of_simulations", 50_000))

    print("\n=== AMERICAN OPTION INPUTS ===")
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
        "Number of steps": number_of_steps,
        "Number of simulations": number_of_simulations,
        "Dividends": dividends_list,
    }
    for key, value in input_fields.items():
        print(f"{key:25s}: {value}")

    # =====================================================
    # 2) EXTRACT AND CALCULATE PARAMETERS
    # =====================================================

    S0 = stock_price_raw
    K = strike_price_raw

    # Time to maturity in years (Actual/365)
    T = calculate_time_to_maturity(
        start_date,
        start_time,
        expiration_date,
        expiration_time,
    )

    if T <= 0.0:
        # Already at or past expiration → intrinsic only
        if option_type == "call":
            price_now = max(S0 - K, 0.0)
        elif option_type == "put":
            price_now = max(K - S0, 0.0)
        else:
            raise ValueError("Option type must be 'call' or 'put'.")
        return {
            "option_price": float(round(price_now, 6)),
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0,
        }

    if sigma <= 0.0:
        raise ValueError("Volatility must be > 0 for American Monte Carlo.")

    number_of_steps = max(1, int(number_of_steps))
    number_of_simulations = max(1_000, int(number_of_simulations))

    is_call = (option_type == "call")

    # --- Build dividend schedule as (tau_in_years, amount) --- (different dividend treatment compared to other calculators)
    start_dt = datetime.strptime(start_date, DATE_FMT)
    expiry_dt = datetime.strptime(expiration_date, DATE_FMT)

    dividend_events: List[Tuple[float, float]] = []
    for div in dividends_list:
        try:
            pay_dt = datetime.strptime(div["date"], DATE_FMT)
            amount = float(div["amount"])
        except Exception:
            continue

        if amount <= 0.0:
            continue
        # only dividends strictly between start and expiry
        if start_dt < pay_dt <= expiry_dt:
            tau = calculate_year_fraction(start_dt, pay_dt)
            if 0.0 < tau <= T:
                dividend_events.append((tau, amount))

    # Sort by time to be safe
    dividend_events.sort(key=lambda x: x[0])

    # Pre-generate random normals for common random numbers
    rng = np.random.default_rng(42)
    base_normals = rng.standard_normal(size=(number_of_simulations, number_of_steps))

    # =====================================================
    # 3) CORE MONTE CARLO + LSM WITH DISCRETE DIVIDENDS
    # =====================================================

    def _build_dividends_per_step(total_T: float, steps: int) -> np.ndarray:
        """
        Map each dividend (tau, amount) to a time step index 0..steps-1.
        At the end of step t we subtract div_per_step[t] from the price.
        """
        div_per_step = np.zeros(steps, dtype=float)
        if not dividend_events or total_T <= 0.0:
            return div_per_step

        dt_local = total_T / steps
        for tau, amount in dividend_events:
            # find step index whose end time (t+1)*dt is >= tau
            idx = int(tau / dt_local)
            if idx >= steps:
                idx = steps - 1
            div_per_step[idx] += amount
        return div_per_step

    def _simulate_paths_with_dividends(
        S_initial: float,
        sigma_local: float,
        r_local: float,
        T_local: float,
        normals: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate GBM paths with risk-free drift r_local and volatility sigma_local,
        and apply discrete dividend jumps at the approximate ex-dates.
        """
        steps = normals.shape[1]
        dt_local = T_local / steps
        drift = (r_local - 0.5 * sigma_local * sigma_local) * dt_local
        vol_sqrt_dt = sigma_local * np.sqrt(dt_local)

        div_per_step = _build_dividends_per_step(T_local, steps)

        paths = np.empty((normals.shape[0], steps + 1), dtype=float)
        paths[:, 0] = S_initial

        for t_idx in range(steps):
            z = normals[:, t_idx]
            paths[:, t_idx + 1] = paths[:, t_idx] * np.exp(drift + vol_sqrt_dt * z)

            # apply dividend jump after evolving to this step
            div_amt = div_per_step[t_idx]
            if div_amt != 0.0:
                paths[:, t_idx + 1] = np.maximum(paths[:, t_idx + 1] - div_amt, 0.0)

        return paths

    def _intrinsic_value(spot: np.ndarray, is_call_flag: bool) -> np.ndarray:
        if is_call_flag:
            return np.maximum(spot - K, 0.0)
        else:
            return np.maximum(K - spot, 0.0)

    def _longstaff_schwartz_price(
        S_initial: float,
        sigma_local: float,
        r_local: float,
        T_local: float,
        normals: np.ndarray,
    ) -> float:
        """
        Longstaff–Schwartz for American option with **discrete dividends**.
        """
        steps = normals.shape[1]
        dt_local = T_local / steps
        discount_local = np.exp(-r_local * dt_local)

        paths = _simulate_paths_with_dividends(
            S_initial,
            sigma_local,
            r_local,
            T_local,
            normals,
        )

        # Payoff at maturity
        cashflows = _intrinsic_value(paths[:, -1], is_call)

        # Backward induction
        for t in range(steps - 1, 0, -1):
            spot_t = paths[:, t]
            intrinsic_t = _intrinsic_value(spot_t, is_call)

            in_the_money = intrinsic_t > 0.0
            if not np.any(in_the_money):
                # No ITM paths → discount all cashflows
                cashflows *= discount_local
                continue

            X = spot_t[in_the_money]
            Y = cashflows[in_the_money] * discount_local  # discounted to time t

            # Basis functions: 1, S, S^2
            A = np.vstack([np.ones_like(X), X, X * X]).T
            coeffs, *_ = np.linalg.lstsq(A, Y, rcond=None)
            continuation = A @ coeffs

            # Exercise rule on ITM set
            intrinsic_itm = intrinsic_t[in_the_money]
            exercise_mask = intrinsic_itm > continuation

            exercised_indices = np.where(in_the_money)[0][exercise_mask]
            continued_indices = np.where(in_the_money)[0][~exercise_mask]

            cashflows[exercised_indices] = intrinsic_t[exercised_indices]
            cashflows[continued_indices] *= discount_local

            # OTM paths just discount
            otm_indices = np.where(~in_the_money)[0]
            cashflows[otm_indices] *= discount_local

        # Discount from first step back to time 0
        price_estimate = np.mean(cashflows) * discount_local
        return float(price_estimate)

    def _price_given(S_local: float,
                     sigma_local: float,
                     r_local: float,
                     T_local: float) -> float:
        """
        Wrapper for pricing with given parameters, using the same random
        numbers (common random numbers) for Greeks.
        """
        if T_local <= 0.0:
            if is_call:
                return max(S_local - K, 0.0)
            else:
                return max(K - S_local, 0.0)
        return _longstaff_schwartz_price(
            S_initial=S_local,
            sigma_local=sigma_local,
            r_local=r_local,
            T_local=T_local,
            normals=base_normals,
        )

    # --- Base price ---
    base_price = _price_given(S0, sigma, r, T)

    # --- Greeks via bump-and-reprice ---
    h_S = max(0.01 * S0, 1e-4)
    h_sigma = max(0.01 * sigma, 1e-4)
    h_r = 1e-4
    h_T = 1.0 / 365.0  # one day in years

    # Delta & Gamma
    price_up_S = _price_given(S0 + h_S, sigma, r, T)
    price_dn_S = _price_given(max(S0 - h_S, 1e-8), sigma, r, T)
    delta = (price_up_S - price_dn_S) / (2.0 * h_S)
    gamma = (price_up_S - 2.0 * base_price + price_dn_S) / (h_S * h_S)

    # Vega
    price_up_sigma = _price_given(S0, sigma + h_sigma, r, T)
    price_dn_sigma = _price_given(S0, max(sigma - h_sigma, 1e-8), r, T)
    vega = (price_up_sigma - price_dn_sigma) / (2.0 * h_sigma)

    # Rho
    price_up_r = _price_given(S0, sigma, r + h_r, T)
    price_dn_r = _price_given(S0, sigma, r - h_r, T)
    rho = (price_up_r - price_dn_r) / (2.0 * h_r)

    # Theta (bump maturity)
    T_short = max(T - h_T, 1e-8)
    T_long = T + h_T
    price_short_T = _price_given(S0, sigma, r, T_short)
    price_long_T = _price_given(S0, sigma, r, T_long)
    theta = (price_short_T - price_long_T) / (2.0 * h_T)

    # =====================================================
    # 4) RETURN RESULTS (MainCalculator.py prints them)
    # =====================================================

    return {
        "option_price": float(round(base_price, 6)),
        "delta": float(round(delta, 6)),
        "gamma": float(round(gamma, 6)),
        "theta": float(round(theta, 6)),
        "vega": float(round(vega, 6)),
        "rho": float(round(rho, 6)),
    }
