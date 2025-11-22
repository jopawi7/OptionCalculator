from math import log, sqrt, exp
from scipy.stats import norm
from Utils import *
from typing import Any, Dict


def calculate_option_value(data: Dict[str, Any]) -> Dict[str, float]:

    """
    Compute price and Greeks for a binary (cash-or-nothing/asset-or-nothing) options
    using Black-Scholes model.

      Required keys in `data`:
      - type: "call" or "put"
      - exercise_style: should be "binary"
      - start_date: "YYYY-MM-DD"
      - start_time: "HH:MM:SS" or "am"/"pm"
      - expiration_date: "YYYY-MM-DD"
      - expiration_time: "HH:MM:SS" or "am"/"pm"
      - strike: float
      - stock_price: float
      - volatility: float
      - interest_rate: float
      - binary_payout: float
      - dividends: list of {"date": "YYYY-MM-DD", "amount": float}


         Payoff structures supported:
      - cash:   pays 1 at expiry if in the money
      - custom: pays binary_payout at expiry if in the money
      - asset:  pays S_T at expiry if in the money
    """

    # 1) Extract and calculate parameters
    option_type = data["type"]
    stock_price = data["stock_price"]
    strike_price = data["strike"]
    sigma = normalize_interest_rate_or_volatility(data['volatility'])
    risk_free_rate = normalize_interest_rate_or_volatility(data['interest_rate'])
    time_to_maturity = calculate_time_to_maturity(data["start_date"], data["start_time"], data["expiration_date"], data["expiration_time"])
    payout_at_expiry = data.get("binary_payout", 1.0)

    # Normalize dividends
    pv_dividends = calculate_present_value_dividends(
        data.get("dividends", []),
        data["start_date"],
        data["expiration_date"],
        risk_free_rate,
    )

    continuous_dividend_yield = calc_continuous_dividend_yield(
        stock_price,
        pv_dividends,
        time_to_maturity,
    )

    payoff_type = data.get("binary_payoff_structure", "cash")
    binary_payout = data.get("binary_payout", 1.0)

    is_asset_or_nothing = payoff_type == "asset"
    if payoff_type == "cash":
        payout_at_expiry = 1.0
    elif payoff_type == "custom":
        payout_at_expiry = binary_payout
    else:
        payout_at_expiry = None # not used for asset-or-nothing


    #2) Black–Scholes core and greeks
    sqrt_T = np.sqrt(time_to_maturity)
    d1 = (np.log(stock_price / strike_price) + (risk_free_rate - continuous_dividend_yield + 0.5 * sigma * sigma) * time_to_maturity) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if is_asset_or_nothing:
        if option_type == "call":
            base_price = stock_price * np.exp(-continuous_dividend_yield * time_to_maturity) * norm.cdf(d1)
        else:
            base_price = stock_price * np.exp(-continuous_dividend_yield * time_to_maturity) * norm.cdf(-d1)
    else:
        if option_type == "call":
            base_price = payout_at_expiry * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2)
        else:
            base_price = payout_at_expiry * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2)

    # Greeks via finite differences
    def price_shift(s_shift, r_shift, sigma_shift, t_shift):
        sqrt_Ts = np.sqrt(t_shift)
        d1s = (np.log(s_shift / strike_price) + (r_shift - continuous_dividend_yield + 0.5 * sigma_shift * sigma_shift) * t_shift) / (
                    sigma_shift * sqrt_Ts)
        d2s = d1s - sigma_shift * sqrt_Ts
        if is_asset_or_nothing:
            return (
                s_shift * np.exp(-continuous_dividend_yield * t_shift) * norm.cdf(d1s)
                if option_type == "call"
                else s_shift * np.exp(-continuous_dividend_yield * t_shift) * norm.cdf(-d1s)
            )
        else:
            disc = np.exp(-r_shift * t_shift)
            coeff = payout_at_expiry
            return (
                coeff * disc * norm.cdf(d2s)
                if option_type == "call"
                else coeff * disc * norm.cdf(-d2s)
            )

    dS = max(1e-4 * stock_price, 1e-4)
    price_up_S = price_shift(stock_price + dS, risk_free_rate, sigma, time_to_maturity)
    price_down_S = price_shift(stock_price - dS, risk_free_rate, sigma, time_to_maturity)
    delta = (price_up_S - price_down_S) / (2 * dS)
    gamma = (price_up_S - 2 * base_price + price_down_S) / (dS ** 2)

    dsigma = max(1e-4 * sigma, 1e-4)
    price_up_sigma = price_shift(stock_price, risk_free_rate, sigma + dsigma, time_to_maturity)
    price_down_sigma = price_shift(stock_price, risk_free_rate, sigma - dsigma, time_to_maturity)
    vega = (price_up_sigma - price_down_sigma) / (2 * dsigma)

    dr = max(1e-4 * max(abs(risk_free_rate), 1.0), 1e-4)
    price_up_r = price_shift(stock_price, risk_free_rate + dr, sigma, time_to_maturity)
    price_down_r = price_shift(stock_price, risk_free_rate - dr, sigma, time_to_maturity)
    rho = (price_up_r - price_down_r) / (2 * dr)

    dT = min(0.01, 0.25 * time_to_maturity) if time_to_maturity > 0.01 else time_to_maturity * 0.5
    price_up_T = price_shift(stock_price, risk_free_rate, sigma, time_to_maturity + dT)
    price_down_T = price_shift(stock_price, risk_free_rate, sigma, time_to_maturity - dT)
    theta = -((price_up_T - price_down_T) / (2 * dT))

    return {
        "theoretical_price": round(base_price, 3),
        "delta": round(delta, 3),
        "gamma": round(gamma, 3),
        "rho": round(rho, 3),
        "theta": round(theta, 3),
        "vega": round(vega, 3),
    }