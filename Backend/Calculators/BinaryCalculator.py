from __future__ import annotations

from typing import Any, Dict
import numpy as np
from scipy.stats import norm
from Backend.Calculators.Utils import *


# ---------------------------------------------------------
# Filename: BinaryCalculatr.py
# Created: 2025-11-22
# Calculate the price and Greeks of a European binary option (call or put)
# under the Black–Scholes model with continuous dividend yield.
# ---------------------------------------------------------


def calculate_option_value(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate the price and Greeks of a European binary option (call or put)
    under the Black–Scholes model with continuous dividend yield.

    Payoff structures supported:
      - cash:   pays 1 at expiry if in the money
      - custom: pays binary_payout at expiry if in the money
      - asset:  pays S_T at expiry if in the money
    """

    # =====================================================
    # 1) EXTRACT AND CALCULATE PARAMETERS
    # =====================================================
    option_type = data["type"]
    S0 = data["stock_price"]
    K = data["strike"]
    sigma = normalize_interest_rate_or_volatility(data["volatility"])
    r = normalize_interest_rate_or_volatility(data["interest_rate"])
    dividends_list = data.get("dividends", [])

    T = calculate_time_to_maturity(data["start_date"],data["start_time"],data["expiration_date"], data["expiration_time"], )

    pv_div = calculate_present_value_dividends(
        dividends_list,data["start_date"],data["expiration_date"],r)

    q = calc_continuous_dividend_yield(S0,pv_div,T)


    payoff_type = data.get("binary_payoff_structure", "cash")
    binary_payout = data['binary_payout']

    is_asset_or_nothing = payoff_type == "asset"

    if payoff_type == "cash":
        payout_at_expiry = 1.0
    elif payoff_type == "custom":
        payout_at_expiry = binary_payout
    else:
        payout_at_expiry = None  # not used for asset-or-nothing

    # =====================================================
    # 3) BLACK–SCHOLES CORE + GREEKS
    # =====================================================
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if is_asset_or_nothing:
        if option_type == "call":
            base_price = S0 * np.exp(-q * T) * norm.cdf(d1)
        else:
            base_price = S0 * np.exp(-q * T) * norm.cdf(-d1)
    else:
        if option_type == "call":
            base_price = payout_at_expiry * np.exp(-r * T) * norm.cdf(d2)
        else:
            base_price = payout_at_expiry * np.exp(-r * T) * norm.cdf(-d2)

    # Greeks via finite differences
    def price_shift(S_shift, r_shift, sigma_shift, T_shift):
        sqrt_Ts = np.sqrt(T_shift)
        d1s = (np.log(S_shift / K) + (r_shift - q + 0.5 * sigma_shift * sigma_shift) * T_shift) / (sigma_shift * sqrt_Ts)
        d2s = d1s - sigma_shift * sqrt_Ts
        if is_asset_or_nothing:
            return (
                S_shift * np.exp(-q * T_shift) * norm.cdf(d1s)
                if option_type == "call"
                else S_shift * np.exp(-q * T_shift) * norm.cdf(-d1s)
            )
        else:
            disc = np.exp(-r_shift * T_shift)
            coeff = payout_at_expiry
            return (
                coeff * disc * norm.cdf(d2s)
                if option_type == "call"
                else coeff * disc * norm.cdf(-d2s)
            )

    dS = max(1e-4 * S0, 1e-4)
    price_up_S = price_shift(S0 + dS, r, sigma, T)
    price_down_S = price_shift(S0 - dS, r, sigma, T)
    delta = (price_up_S - price_down_S) / (2 * dS)
    gamma = (price_up_S - 2 * base_price + price_down_S) / (dS ** 2)

    dsigma = max(1e-4 * sigma, 1e-4)
    price_up_sigma = price_shift(S0, r, sigma + dsigma, T)
    price_down_sigma = price_shift(S0, r, sigma - dsigma, T)
    vega = (price_up_sigma - price_down_sigma) / (2 * dsigma)

    dr = max(1e-4 * max(abs(r), 1.0), 1e-4)
    price_up_r = price_shift(S0, r + dr, sigma, T)
    price_down_r = price_shift(S0, r - dr, sigma, T)
    rho = (price_up_r - price_down_r) / (2 * dr)

    dT = min(0.01, 0.25 * T) if T > 0.01 else T * 0.5
    price_up_T = price_shift(S0, r, sigma, T + dT)
    price_down_T = price_shift(S0, r, sigma, T - dT)
    theta = -((price_up_T - price_down_T) / (2 * dT))

    # =====================================================
    # 4) RETURN RESULTS
    # =====================================================
    return {
        "theoretical_price": float(round(base_price, 3)),
        "delta": float(round(delta, 3)),
        "gamma": float(round(gamma, 3)),
        "theta": float(round(theta, 3)),
        "vega": float(round(vega, 3)),
        "rho": float(round(rho, 3)),
    }