from Utils import *
import math

# ---------------------------------------------------------
# Filename: AmericanCalculator.py
# LastUpdated: 2025-11-15
# Description: AmericanOptionCalculator
# ---------------------------------------------------------


def _american_binomial_calculation(stock_price, strike_price, risk_free_rate, dividend_yield, sigma, time_to_maturity, steps, is_call=True):
    """
    Prices an American option using a binomial tree.

    Used in:
        - AmericanCalculator

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Strike price.
        risk_free_rate (float): Risk-free interest rate.
        dividend_yield (float): Dividend yield.
        sigma (float): Volatility.
        time_to_maturity (float): Time to maturity in years.
        steps (int): Number of time steps in the tree.
        is_call (bool): True if call option, False if put.

    Returns:
        float: Theoretical price of the American option.
    """

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


def calculate_option_value(data):
    """
    Calculates the option value for an American using a binomial tree.
      Used in:
          - AmericanCalcluator

      Args:
          data: JSON-Object with all necessary data for the calculation.

      Returns:
          data: JSON-Object with the calculated values for the option.
      """

    # Parse Input to useable variables
    strike_price = float(data["strike"])
    risk_free_rate = normalize_interest_rate(data["interest_rate"])
    sigma = float(data["volatility"])
    time_to_maturity = calculate_time_to_maturity(data["start_date"], data.get("start_time"), data["expiration_date"],
                                                  data.get("expiration_time"))

    # Check if option type is call or put
    is_call = data["type"] == "call"

    #Number of steps for tree <- Must be set here; eventually dynamic approach to program
    # Steps
    steps = int(data.get("steps", 1000))
    steps = max(1, steps)

    # Dividends handling
    stock_price = max(
        float(data["stock_price"]) - calculate_present_value_dividends(
            data.get("dividends", []),
            data["start_date"],
            data["expiration_date"],
            risk_free_rate
        ),
        1e-12
    )
    dividend_yield = 0

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
        steps_local = max(1, steps)
        return _american_binomial_calculation(Sv, strike_price, rv, dividend_yield, sigmav, max(Tv, 0.0), steps_local, is_call=is_call)

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
