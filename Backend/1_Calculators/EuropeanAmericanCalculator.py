### EUROPEAN CALCULATOR:

import scipy.stats as si
import numpy as np


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