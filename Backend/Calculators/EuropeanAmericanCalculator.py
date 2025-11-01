import scipy.stats as si
import numpy as np

# ---------------------------------------------------------
# Filename: EuropeanAmericanCalculator.py
# Author:
# Created: 2025-10-30
# Description:
# ---------------------------------------------------------


def calculateOptionValue(data):
    #Expected Output values
    price = 1
    delta = 0
    gamma = 0
    rho = 0
    theta = 0
    vega = 0
    
    #How to call the values out of the data object:
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
    print(f"dividens: {data['dividends']}" )


    # TODO: Logic to calculate OptionValue here or method to calculate option value

    return {
         "theoretical_price": round(price, 4), "delta": round(delta, 4),
        "gamma": round(gamma, 6), "rho": round(rho, 4), "theta": round(theta, 4), "vega": round(vega, 4)
    }


def BS(S, K, T, r, sigma, option):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    #option: can be 'call' or 'put'
    
    d1 = (np.log(S / K) + (r  + 0.5 * sigma ** 2) * T)\
         / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r  - 0.5 * sigma ** 2) * T)\
         / (sigma * np.sqrt(T))
    
    if option == 'call':
        result = (S  * si.norm.cdf(d1, 0.0, 1.0) \
                     - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    if option == 'put':
        result = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)\
                      - S * si.norm.cdf(-d1, 0.0, 1.0))
        
    return result