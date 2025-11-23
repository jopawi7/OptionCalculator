For the input.json file only the following structure is accepted. NEVER remove keys in input.json

Even if parameters are not used such as average_type, binary_payout, binary_payoff_structure
they NEVER SHOULD BE NULL but have a valid value as DUMMY.

{
  "type": "call" | "put",
  "exercise_style": "american" | "european" | "asian" | "binary",
  "start_date": "YYYY-MM-DD",
  "start_time": "HH:MM:SS" | "AM" | "PM" | "am" | "Am" |...,
  "expiration_date": "YYYY-MM-DD",
  "expiration_time": "HH:MM:SS" | "AM" | "PM" | "am" | "Am" |...,
  "strike": float, (use point for decimal places)
  "stock_price": float, (use point for decimal places)
  "volatility": float,
  "interest_rate": float,
  "average_type": "arithmetic" | "geometric",
  "number_of_steps": int,
  "number_of_simulations": int,
  "binary_payout": float
  "binary_payoff_structure": "cash" | "asset" | "custom"
  "dividends": \[
  { "date": "YYYY-MM-DD", "amount": float },
  { "date": "YYYY-MM-DD", "amount": float },
  ...\]
}

## Mathematical Constraints
- The `stock_price` and `strike` values and `binary_payout` must be strictly positive:
  - \( S > 0 \)
  - \( K > 0 \)

- The `volatility` must be greater than zero:
  - \( \sigma > 0 \)

- The `interest_rate` can be positive or negative but is typically bounded:
  - Usually \( r \geq -1 \) (rarely less than -100%) or \( r \geq 0 \) in most practical cases

- The `number_of_steps` and `number_of_simulations` must be integers greater than or equal to 1:
  - \( n \geq 1 \). Furthe number_of steps <= 10000 and number_of simulations <= 100000

- Dividend amounts (`dividends[].amount`) must be zero or positive:
  - \( D \geq 0 \)

- The `expiration_date` must be chronologically after the `start_date`:
  - \( t_{exp} > t_{start} \)
  - This temporal constraint requires application-level validation, since JSON Schema cannot express cross-field conditions

- All numeric values must be finite (no NaN or infinity)
