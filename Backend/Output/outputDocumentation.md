The output is always provided in the following JSON format:
{
  "theoretical_price": float,
  "delta": float,
  "gamma": float,
  "rho": float,
  "theta": float,
  "vega": float
}

Details:
-All numerical values are rounded to three decimal places for clarity and consistency.

-The fields represent the option’s price (theoretical_price) and its sensitivities (Greeks):
#delta: Rate of change of the option price with respect to the underlying asset price
#gamma: Rate of change of delta with respect to the underlying asset price
#rho: Sensitivity to the risk-free interest rate
#theta: Sensitivity to the passage of time (time decay)
#vega: Sensitivity to volatility

