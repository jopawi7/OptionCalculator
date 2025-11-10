from math import log, sqrt, exp, erf, pi
from datetime import datetime
import json
import os


def N(x):
    """CDF of the standard normal distribution."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def n(x):
    """PDF of the standard normal distribution."""
    return (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * x**2)


def read_inputs_from_file(filename=None):
    """
    Read and normalize input parameters from a JSON file.
    If no filename is provided, use ../Input/input.json relative to this script.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if filename is None:
        filename = os.path.join(base_dir, "..", "Input", "input.json")

    filename = os.path.normpath(filename)

    if not os.path.exists(filename):
        raise FileNotFoundError(f"JSON file not found: {filename}")

    with open(filename, "r") as f:
        data = json.load(f)

    if isinstance(data.get("dividends"), list):
        total_div = sum(d.get("amount", 0.0) for d in data["dividends"])
        data["dividends"] = total_div / len(data["dividends"]) if data["dividends"] else 0.0
    elif not isinstance(data.get("dividends"), (int, float)):
        data["dividends"] = 0.0

    data["type"] = data.get("type", "").lower()

    return data


def calculate_option_value(data):
    """
    Compute price and Greeks for a binary (cash-or-nothing) option 
    using Black-Scholes model.
    """
    opt_type = data.get("type", "").lower()
    if opt_type not in ["call", "put"]:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    S = data["stock_price"]
    K = data["strike"]
    sigma = data["volatility"]
    r = data["interest_rate"]
    
    # Normalize dividends
    dividends = data.get("dividends", 0.0)
    if isinstance(dividends, list):
        q = sum(d.get("amount", 0.0) for d in dividends) / len(dividends) if dividends else 0.0
    else:
        q = float(dividends) if dividends else 0.0
    
    Q = 1.0

    fmt = "%Y-%m-%d"
    T = (datetime.strptime(data["expiration_date"], fmt) - 
         datetime.strptime(data["start_date"], fmt)).days / 365.0
    
    if T <= 0:
        raise ValueError("Expiration date must be after start date.")

    d2 = (log(S / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    Nd2 = N(d2)
    nd2 = n(d2)

    sqrt_T = sqrt(T)
    exp_rT = exp(-r * T)
    S_sigma_sqrtT = S * sigma * sqrt_T

    if opt_type == "call":
        price = Q * exp_rT * Nd2
        delta = Q * exp_rT * nd2 / S_sigma_sqrtT
        gamma = -Q * exp_rT * nd2 * d2 / (S**2 * sigma**2 * T)
        vega = -Q * exp_rT * nd2 * d2 / sigma
        rho = Q * T * exp_rT * Nd2
        theta = -Q * exp_rT * (
            r * Nd2 + nd2 / (2 * T * sigma * sqrt_T) * 
            (log(S / K) + (r - q + 0.5 * sigma**2) * T)
        )
    else:
        Nmd2 = N(-d2)
        price = Q * exp_rT * Nmd2
        delta = -Q * exp_rT * nd2 / S_sigma_sqrtT
        gamma = Q * exp_rT * nd2 * d2 / (S**2 * sigma**2 * T)
        vega = Q * exp_rT * nd2 * d2 / sigma
        rho = -Q * T * exp_rT * Nmd2
        theta = -Q * exp_rT * (
            -r * Nmd2 + nd2 / (2 * T * sigma * sqrt_T) * 
            (log(S / K) + (r - q + 0.5 * sigma**2) * T)
        )

    return {
        "theoretical_price": round(price, 10),
        "delta": round(delta, 10),
        "gamma": round(gamma, 10),
        "rho": round(rho, 10),
        "theta": round(theta, 10),
        "vega": round(vega, 10),
    }


if __name__ == "__main__":
    print("\nBinary Option Calculator")
    choice = input("Use JSON file input? (y/n): ").strip().lower()

    if choice == "y":
        try:
            data = read_inputs_from_file()
            print("Loaded input from ../Input/input.json")
        except Exception as e:
            print(f"Error loading JSON: {e}")
            exit(1)
    else:
        print("Enter parameters manually:\n")
        data = {
            "type": input("Option type (call / put): ").strip().lower(),
            "start_date": input("Start date (YYYY-MM-DD): ").strip(),
            "expiration_date": input("Expiration date (YYYY-MM-DD): ").strip(),
            "strike": float(input("Strike: ")),
            "stock_price": float(input("Stock price: ")),
            "volatility": float(input("Volatility (decimal, e.g. 0.2): ")),
            "interest_rate": float(input("Interest rate (decimal, e.g. 0.05): ")),
            "dividends": float(input("Dividend yield (decimal, e.g. 0.00): ")),
        }

    print("\nRESULT")
    try:
        result = calculate_option_value(data)
        for k, v in result.items():
            print(f"{k}: {v}")
    except Exception as e:
        print(f"Calculation error: {e}")
