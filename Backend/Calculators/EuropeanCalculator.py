import numpy as np
import scipy.stats as si
from datetime import datetime
from Utils import *

# ---------------------------------------------------------
# Filename: EuropeanCalculator.py
# LastUpdated: 2025-11-15
# Description: Calculate the value of European options based on Black-Scholes
# ---------------------------------------------------------

def calculate_option_value(data):
    """
    Calculates the option value for a European option with fixed maturity based on Black-Scholes. For better performance we use floats instead of precise decimal numbers and accept that this might lead to small deviations.

    Used in:
        - EuropeanCalculator

    Args:
        data: JSON-Object with all necessary data for the calculation.

    Returns:
        data: JSON-Object with the calculated values for the option.
    """

    #Parse Input to useable variables

    strike_price  = float(data["strike"])
    risk_free_rate = normalize_rate(data["interest_rate"])

    #StockPrice and Dividends - reduces stock price by present value of expected discrete dividends
    stock_price = max(float(data["stock_price"]) - calculate_present_value_dividends(data.get("dividends", []),
                                                                                     data["start_date"],
                                                                                     data["expiration_date"],
                                                                                     risk_free_rate), 1e-12)

    sigma = float(data["volatility"])
    time_to_maturity = calculate_time_to_maturity(data["start_date"], data.get("start_time"), data["expiration_date"], data.get("expiration_time"))

    #Continious dividend yield set to zero because NPV of discrete dividends already deducted
    dividend_yield = 0.0

    #Calculate risk_free_rate discounted and dividend_yiel discounted
    discount_factor_risk_free_rate = np.exp(-risk_free_rate * time_to_maturity)
    discount_factor_dividend_yield = np.exp(-dividend_yield * time_to_maturity)


    # Edge Cases, Time to maturity reached, negative volatility, negative stock price or negative strike_price
    # Under consistent and secure input validation not necessary but kept as a fallback option :)
    if time_to_maturity <= 0.0 or sigma <= 0.0 or stock_price <= 0.0 or strike_price <= 0.0:
        if data["type"] == "call":
            #For a call option, the option price is the maximum of the discounted stock price minus the discounted strike price and zero
            price = max(stock_price * discount_factor_dividend_yield - strike_price * discount_factor_risk_free_rate, 0.0)
        elif data["type"] == "put":
            #For a put option, it is the maximum of the discounted strike price minus the discounted stock price and zero
            price = max(strike_price * discount_factor_risk_free_rate - stock_price * discount_factor_dividend_yield, 0.0)
        else:
            raise ValueError("Type must be 'call' or 'put'")
        return {
            "theoretical_price": round(float(price), 3),
            "delta": 0.0,
            "gamma": 0.0,
            "rho": 0.0,
            "theta": 0.0,
            "vega": 0.0
        }

    #Black-scholes calculation
    sqrt_time_to_maturity = np.sqrt(time_to_maturity)
    d1 = (np.log(stock_price / strike_price) + (risk_free_rate - dividend_yield + 0.5 * sigma * sigma) * time_to_maturity) / (sigma * sqrt_time_to_maturity)
    d2 = d1 - sigma * sqrt_time_to_maturity
    Nd1, Nd2 = si.norm.cdf(d1), si.norm.cdf(d2)
    nd1 = si.norm.pdf(d1)


    if data["type"] == "call":
        #Call option price ->  get disocunted stck if exercised and pay discounted strikt price
        price = stock_price * discount_factor_dividend_yield * Nd1 - strike_price * discount_factor_risk_free_rate * Nd2
        delta = discount_factor_dividend_yield * Nd1
        theta = -(stock_price * discount_factor_dividend_yield * nd1 * sigma) / (2 * sqrt_time_to_maturity) - risk_free_rate * strike_price * discount_factor_risk_free_rate * Nd2 + dividend_yield * stock_price * discount_factor_dividend_yield * Nd1
        rho   = strike_price * time_to_maturity * discount_factor_risk_free_rate * Nd2
    elif data["type"] == "put":
        #Put option price -> sell disounted strike price minus the chance you will have to do sp
        Nmd1, Nmd2 = si.norm.cdf(-d1), si.norm.cdf(-d2)
        price = strike_price * discount_factor_risk_free_rate * Nmd2 - stock_price * discount_factor_dividend_yield * Nmd1
        delta = discount_factor_dividend_yield * (Nd1 - 1.0)
        theta = -(stock_price * discount_factor_dividend_yield * nd1 * sigma) / (2 * sqrt_time_to_maturity) + risk_free_rate * strike_price * discount_factor_risk_free_rate * Nmd2 - dividend_yield * stock_price * discount_factor_dividend_yield * Nmd1
        rho   = -strike_price * time_to_maturity * discount_factor_risk_free_rate * Nmd2
    else:
        raise ValueError("Type must be 'call' or 'put'")

    gamma = (discount_factor_dividend_yield * nd1) / (stock_price * sigma * sqrt_time_to_maturity)
    vega  = stock_price * discount_factor_dividend_yield * nd1 * sqrt_time_to_maturity

   # Round result to three decimal places and return it to main.py
    #Delta: How much option price changes when the underlying stock price moves by one unit (calls positve, puts negative)
    #Gamma: Indicates how much delta itself changes when the underlying stock price moves by one unit
    #Theta: The rate at which the option price decreases as time passes ("time decay").
    #Rho: Shows how much the option price changes if the risk-free interest rate increases by one percentage point
    #Vega: Measures how much the option price changes if volatility of the underlying asset increases by one percentage point.


    result = {
        "theoretical_price": round(float(price), 3),
        "delta": round(float(delta), 3),
        "gamma": round(float(gamma), 3),
        "rho": round(float(rho), 3),
        "theta": round(float(theta), 3),
        "vega": round(float(vega), 3),
    }
    return result