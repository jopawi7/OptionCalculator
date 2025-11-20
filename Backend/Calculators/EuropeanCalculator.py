from datetime import datetime
import numpy as np
from scipy.stats import norm
from datetime import datetime

# ---------------------------------------------------------
# Filename: EuropeanCalculator.py
# LastUpdated: 2025-11-16
# Our Model assumes constant interest rates and no forward interest rates compared to Cboe
# Description: Calculate the value of European options based on Black-Scholes without considering dividends
# Possibility to inclued Prepaid-Forward
# ---------------------------------------------------------


def parse_time_str(time_str):
    """
    Parst ein Zeitformat gemäß Schema:
    - HH:MM:SS (24h) z.B. "15:30:00"
    - oder "am"/"pm" (ohne Uhrzeit, interpretiert als feste Zeit)
    Liefert Zeitstring 'HH:MM:SS' für datetime parsing zurück.
    """
    if not time_str:
        return "00:00:00"
    s = time_str.lower()
    if s == 'am':
        return "09:30:00"  # Marktöffnung
    elif s == 'pm':
        return "16:00:00"  # Marktschluss
    else:
        # Annahme: HH:MM:SS vorliegend
        return time_str

def year_fraction_with_exact_days(start_date, start_time, expiration_date, expiration_time):
    start_str = f"{start_date} {start_time}"
    expiration_str = f"{expiration_date} {expiration_time}"

    start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
    exp_dt = datetime.strptime(expiration_str, "%Y-%m-%d %H:%M:%S")

    delta_days = (exp_dt - start_dt).total_seconds() / (24 * 3600)
    return delta_days / 365.0


def calculate_present_value_dividends(dividend_list, start_date, expiry_date, risk_free_rate):
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(expiry_date, str):
        expiry_date = datetime.strptime(expiry_date, "%Y-%m-%d")

    if not isinstance(dividend_list, list):
        return 0.0

    present_value = 0.0
    for div in dividend_list:
        pay_date = datetime.strptime(div["date"], "%Y-%m-%d")
        if start_date < pay_date < expiry_date:
            # übergib gültige Strings & Zeiten:
            T = year_fraction_with_exact_days(
                start_date.strftime("%Y-%m-%d"), "00:00:00",
                pay_date.strftime("%Y-%m-%d"), "00:00:00"
            )
            present_value += float(div["amount"]) * np.exp(-risk_free_rate * T)
    return present_value



def calculate_option_value(data):
    option_type = data["type"]
    start_date = data["start_date"]
    start_time = data["start_time"]
    expiration_date = data["expiration_date"]
    expiration_time = data["expiration_time"]
    strike = data["strike"]
    stock_price = data["stock_price"]
    volatility = data["volatility"]
    interest_rate = data["interest_rate"] / 100.0


    T = year_fraction_with_exact_days(start_date, start_time, expiration_date, expiration_time)
    S = stock_price - calculate_present_value_dividends(data["dividends"], start_date, expiration_date, interest_rate)

    print(S)

    d1 = (np.log(S / strike) + (interest_rate + 0.5 * volatility ** 2) * T) / (volatility * np.sqrt(T))
    d2 = d1 - volatility * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - strike * np.exp(-interest_rate * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (- (S * volatility * norm.pdf(d1)) / (2 * np.sqrt(T))
                 - interest_rate * strike * np.exp(-interest_rate * T) * norm.cdf(d2))
        rho = strike * T * np.exp(-interest_rate * T) * norm.cdf(d2)
    else:
        price = strike * np.exp(-interest_rate * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
        theta = (- (S * volatility * norm.pdf(d1)) / (2 * np.sqrt(T))
                 + interest_rate * strike * np.exp(-interest_rate * T) * norm.cdf(-d2))
        rho = -strike * T * np.exp(-interest_rate * T) * norm.cdf(-d2)

    gamma = norm.pdf(d1) / (S * volatility * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)


    return {
        "theoretical_price": round(price, 3),
        "delta": round(delta, 3),
        "gamma": round(gamma, 3),
        "rho": round(rho, 3),
        "theta": round(theta, 3),
        "vega": round(vega, 3),
    }