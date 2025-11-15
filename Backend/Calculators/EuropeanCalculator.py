import numpy as np
import scipy.stats as si
from datetime import datetime
from Utils import *

# ---------------------------------------------------------
# Filename: EuropeanCalculator.py
# LastUpdated: 2025-11-15
# Description: Calculate the value of European otpions based on Black-Scholes
# ---------------------------------------------------------

def calculate_option_value(data):
    # Inputs
    opt_type = str(data["type"]).strip().lower()
    S0 = float(data["stock_price"])
    K  = float(data["strike"])
    sig = float(data["volatility"])
    r   = _normalize_rate(data["interest_rate"])

    start_dt = datetime.combine(
        datetime.strptime(data["start_date"], "%Y-%m-%d").date(),
        parse_time_string(data.get("start_time"))
    )
    exp_dt = datetime.combine(
        datetime.strptime(data["expiration_date"], "%Y-%m-%d").date(),
        parse_time_string(data.get("expiration_time"))
    )
    T = _yearfrac(start_dt, exp_dt)

    # Dividenden
    div = data.get("dividends", data.get("dividens", 0.0))
    if isinstance(div, list):
        S = max(S0 - _pv_dividends(div, start_dt, exp_dt, r), 1e-12)
        q = 0.0
    else:
        S = S0
        q = float(div)

    # Grenzfälle
    if T <= 0.0 or sig <= 0.0 or S <= 0.0 or K <= 0.0:
        df_r = np.exp(-r*T)
        df_q = np.exp(-q*T)
        if opt_type == "call":
            price = max(S*df_q - K*df_r, 0.0)
        elif opt_type == "put":
            price = max(K*df_r - S*df_q, 0.0)
        else:
            raise ValueError("type must be 'call' or 'put'")
        return {
            "theoretical_price": round(float(price), 3),
            "delta": 0.0,
            "gamma": 0.0,
            "rho": 0.0,
            "theta": 0.0,
            "vega": 0.0
        }

    # Black-Scholes Berechnung
    sqrtT = np.sqrt(T)
    d1 = (np.log(S/K) + (r - q + 0.5*sig*sig)*T) / (sig*sqrtT)
    d2 = d1 - sig*sqrtT
    Nd1, Nd2 = si.norm.cdf(d1), si.norm.cdf(d2)
    nd1 = si.norm.pdf(d1)
    df_r = np.exp(-r*T)
    df_q = np.exp(-q*T)

    if opt_type == "call":
        price = S*df_q*Nd1 - K*df_r*Nd2
        delta = df_q*Nd1
        theta = -(S*df_q*nd1*sig)/(2*sqrtT) - r*K*df_r*Nd2 + q*S*df_q*Nd1
        rho   = K*T*df_r*Nd2
    elif opt_type == "put":
        Nmd1, Nmd2 = si.norm.cdf(-d1), si.norm.cdf(-d2)
        price = K*df_r*Nmd2 - S*df_q*Nmd1
        delta = df_q*(Nd1 - 1.0)
        theta = -(S*df_q*nd1*sig)/(2*sqrtT) + r*K*df_r*Nmd2 - q*S*df_q*Nmd1
        rho   = -K*T*df_r*Nmd2
    else:
        raise ValueError("type must be 'call' or 'put'")

    gamma = (df_q*nd1)/(S*sig*sqrtT)
    vega  = S*df_q*nd1*sqrtT

    # Greeks OHNE Skalierung - raw values
    out = {
        "theoretical_price": round(float(price), 3),
        "delta": round(float(delta), 3),
        "gamma": round(float(gamma), 3),
        "rho": round(float(rho), 3),
        "theta": round(float(theta), 3),
        "vega": round(float(vega), 3),
    }
    return out