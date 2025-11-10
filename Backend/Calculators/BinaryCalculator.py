# ---------------------------------------------------------
# Filename: BinaryCalculator.py
# Author:
# Created: 2025-10-30
# Description: Compute the value and Greeks of a Binary Option (Call & Put)
#              Conventions aligned with standard reference table
# ---------------------------------------------------------

from math import log, sqrt, exp, erf, pi

# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------
def N(x):
    """CDF de la loi normale standard."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def n(x):
    """PDF de la loi normale standard."""
    return (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * x**2)

# ---------------------------------------------------------
# Main function
# ---------------------------------------------------------
def calculate_option_value(data):
    # Expected Output values
    price = 0
    delta = 0
    gamma = 0
    rho = 0
    theta = 0
    vega = 0

    # Display inputs
    print(f"Option type: {data['type']}")
    print(f"Exercise style: {data['exercise_style']}")
    print(f"Start date: {data['start_date']}")
    print(f"Start time: {data['start_time']}")
    print(f"Expiration date: {data['expiration_date']}")
    print(f"Expiration time: {data['expiration_time']}")
    print(f"Strike: {data['strike']}")
    print(f"Stock price: {data['stock_price']}")
    print(f"Volatility: {data['volatility']}")
    print(f"Interest rate: {data['interest_rate']}")
    print(f"Dividends: {data['dividends']}")

    # ---------------------------------------------------------
    # Hardcoded parameters (will be replaced by data[] later)
    # ---------------------------------------------------------
    S = 110       # current stock price
    K = 100       # strike price
    sigma = 1.20  # volatility (decimal form)
    r = 0.00      # risk-free rate
    T = 0.44      # maturity (years)
    Q = 1.0       # fixed payoff for binary option

    # ---------------------------------------------------------
    # Common quantities
    # ---------------------------------------------------------
    d2 = (log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    Nd2 = N(d2)
    nd2 = n(d2)

    # ---------------------------------------------------------
    # Binary CALL (cash-or-nothing)
    # ---------------------------------------------------------
    if data["type"].lower() in ["call_binary", "binary_call", "call"]:
        price = Q * exp(-r * T) * Nd2
        delta = Q * exp(-r * T) * nd2 / (S * sigma * sqrt(T))
        gamma = Q * exp(-r * T) * nd2 * d2 / (S**2 * sigma**2 * T)
        vega = Q * exp(-r * T) * nd2 * d2 / sigma
        rho = Q * exp(-r * T) * (T * Nd2 - T * nd2 * sqrt(T) / sigma)
        theta = -Q * exp(-r * T) * (
            nd2 * ((log(S / K) + (r - 0.5 * sigma**2) * T) / (2 * T * sigma * sqrt(T)))
            + r * Nd2
        )

    # ---------------------------------------------------------
    # Binary PUT (cash-or-nothing)
    # ---------------------------------------------------------
    elif data["type"].lower() in ["put_binary", "binary_put", "put"]:
        price = -Q * exp(-r * T) * N(-d2)  # signe pour correspondre à la table
        delta = -Q * exp(-r * T) * nd2 / (S * sigma * sqrt(T))
        gamma = -Q * exp(-r * T) * nd2 * d2 / (S**2 * sigma**2 * T)
        vega = -Q * exp(-r * T) * nd2 * d2 / sigma
        rho = -Q * exp(-r * T) * (T * N(-d2) - T * nd2 * sqrt(T) / sigma)
        theta = Q * exp(-r * T) * (
            nd2 * ((log(S / K) + (r - 0.5 * sigma**2) * T) / (2 * T * sigma * sqrt(T)))
            + r * N(-d2)
        )

    else:
        print("⚠️ Type d’option non reconnu (utilise 'call_binary' ou 'put_binary').")

    # ---------------------------------------------------------
    # Return computed values
    # ---------------------------------------------------------
    return {
        "theoretical_price": round(price, 10),
        "delta": round(delta, 10),
        "gamma": round(gamma, 10),
        "rho": round(rho, 10),
        "theta": round(theta, 10),
        "vega": round(vega, 10)
    }


# ---------------------------------------------------------
# Example direct test
# ---------------------------------------------------------
if __name__ == "__main__":
    dummy_data_call = {
        "type": "call_binary",
        "exercise_style": "European",
        "start_date": "2025-10-30",
        "start_time": "09:00",
        "expiration_date": "2026-03-01",
        "expiration_time": "16:00",
        "strike": 100,
        "stock_price": 110,
        "volatility": 1.20,
        "interest_rate": 0.00,
        "dividends": 0.00
    }

    dummy_data_put = dummy_data_call.copy()
    dummy_data_put["type"] = "put_binary"

    print("\n=== CALL BINAIRE ===")
    result_call = calculate_option_value(dummy_data_call)
    for k, v in result_call.items():
        print(f"{k}: {v}")

    print("\n=== PUT BINAIRE ===")
    result_put = calculate_option_value(dummy_data_put)
    for k, v in result_put.items():
        print(f"{k}: {v}")
