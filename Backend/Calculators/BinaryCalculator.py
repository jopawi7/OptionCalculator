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
    Calculate the price and Greeks of a European binary option (call or put)
    under the Black–Scholes model with continuous dividend yield.

    Payoff structures supported:
      - cash:   pays 1 at expiry if in the money
      - custom: pays binary_payout at expiry if in the money
      - asset:  pays S_T at expiry if in the money
    """

    # =====================================================
    # 1) DISPLAY INPUTS
    # =====================================================
    print("\n=== BINARY OPTION INPUTS ===")
    input_fields = {
        "Option type": data["type"],
        "Exercise style": data["exercise_style"],
        "Start date": data["start_date"],
        "Start time": data["start_time"],
        "Expiration date": data["expiration_date"],
        "Expiration time": data["expiration_time"],
        "Stock price (S0)": data["stock_price"],
        "Strike price (K)": data["strike"],
        "Volatility (raw)": data["volatility"],
        "Interest rate (raw)": data["interest_rate"],
        "Dividends": data.get("dividends", []),
        "Binary payoff type": data.get("binary_payoff_type", "cash"),
        "Binary payout": data.get("binary_payout", 1.0),
    }
    for key, value in input_fields.items():
        print(f"{key:25s}: {value}")

    # =====================================================
    # 2) EXTRACT AND CALCULATE PARAMETERS
    # =====================================================
    option_type = data["type"].lower()
    S0 = float(data["stock_price"])
    K = float(data["strike"])
    sigma = normalize_interest_rate_or_volatility(float(data["volatility"]))
    r = normalize_interest_rate_or_volatility(float(data["interest_rate"]))
    dividends_list = data.get("dividends", [])

    T = calculate_time_to_maturity(
        data["start_date"],
        data["start_time"],
        data["expiration_date"],
        data["expiration_time"],
    )

    pv_div = calculate_present_value_dividends(
        dividends_list,
        data["start_date"],
        data["expiration_date"],
        r,
    )
    q = calc_continuous_dividend_yield(
        stock_price=S0,
        pv_dividends=pv_div,
        time_to_maturity=T,
    )

    payoff_type = str(data.get("binary_payoff_type", "cash")).lower()
    binary_payout = float(data.get("binary_payout", 1.0))

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
        "theoretical_price": round(base_price, 6),
        "delta": round(delta, 6),
        "gamma": round(gamma, 6),
        "theta": round(theta, 6),
        "vega": round(vega, 6),
        "rho": round(rho, 6),
    }
