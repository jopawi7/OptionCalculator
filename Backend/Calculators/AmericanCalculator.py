from typing import Any, Dict, List, Tuple
from datetime import datetime
import numpy as np
from typing import Dict, Any, Tuple

#Relative Import needed for Frontend/Absolute for MainCalculator.py
try:
    from .Utils import *
except ImportError:
    from Utils import *

# ---------------------------------------------------------
# Filename: AmericanCalculator.py
# Created: 2025-11-22
# Price an American option (call or put) using Monte Carlo simulation
# with Longstaff–Schwartz for early exercise and discrete dividends.
# ---------------------------------------------------------

DATE_FMT = "%Y-%m-%d"

def calculate_option_value(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Price an American option (call or put) using Monte Carlo simulation
    with Longstaff–Schwartz for early exercise and discrete dividend jumps.

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
    # 1) EXTRACT AND CALCULATE PARAMETERS
    # =====================================================
    option_type = data["type"]
    start_date = data["start_date"]
    start_time = data.get("start_time", "")
    expiration_date = data["expiration_date"]
    expiration_time = data.get("expiration_time", "")
    S0 = data["stock_price"]
    K = data["strike"]
    volatility_raw = data["volatility"]
    interest_rate_raw = data["interest_rate"]
    dividends_list = data.get("dividends", [])

    # Normalize volatility and interest rate (e.g. 25 → 0.25)
    sigma = normalize_interest_rate_or_volatility(volatility_raw)
    r = normalize_interest_rate_or_volatility(interest_rate_raw)

    number_of_steps = int(data.get("number_of_steps", 100))
    number_of_simulations = int(data.get("number_of_simulations", 10000))


    # Time to maturity in years (Actual/365)
    T = calculate_time_to_maturity(start_date, start_time,expiration_date,expiration_time)


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
    # 2) CORE MONTE CARLO + LSM WITH DISCRETE DIVIDENDS
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

    delta, gamma, theta, vega, rho = calculate_greeks_without_dividend_payments(option_type, S0,K, r, sigma, T)


    # =====================================================
    # 3) RETURN RESULTS (MainCalculator.py prints them)
    # =====================================================
    return {
        "theoretical_price": round(0 if base_price <= 0 else base_price, 3),
        "delta": round(0 if base_price <= 0 else delta, 3),
        "gamma": round(0 if base_price <= 0 else gamma, 3),
        "rho": round(0 if base_price <= 0 else rho, 3),
        "theta": round(0 if base_price <= 0 else theta, 3),
        "vega": round(0 if base_price <= 0 else vega, 3),
    }

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