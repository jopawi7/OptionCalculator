The output is always provided in the following JSON format:
{
  "theoretical_price": float,
  "delta": number,
  "gamma": number,
  "rho": number,
  "theta": number,
  "vega": number
}

Details:
-All numerical values are rounded to three decimal places for clarity and consistency.

-The fields represent the option’s price (theoretical_price) and its sensitivities (Greeks):
#delta: Rate of change of the option price with respect to the underlying asset price
#gamma: Rate of change of delta with respect to the underlying asset price
#rho: Sensitivity to the risk-free interest rate
#theta: Sensitivity to the passage of time (time decay)
#vega: Sensitivity to volatility

-Mathematical Constraints
#theoretical_price is always greater than or equal to zero.
#delta ranges from -1 to 1 (negative for puts, positive for calls).
#gamma and vega are non-negative values (≥ 0).
#theta is typically less than or equal to zero (reflecting time decay).
#rho can be positive or negative, depending on the option type and market conditions.

