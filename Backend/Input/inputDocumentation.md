For the input.json file only the following structure is accepted:
{
  "type": "call" | "put",
  "exercise_style": "american" | "european" | "asian" | "binary",
  "start_date": "YYYY-MM-DD",
  "start_time": "HH:MM:SS" | "AM" | "PM" | "am" | "Am" |...,
  "expiration_date": "YYYY-MM-DD",
  "expiration_time": "HH:MM:SS" | "AM" | "PM" | "am" | "Am" |...,
  "strike": float,
  "stock_price": float,
  "volatility": float,
  "interest_rate": float,
  "average_type": "arithmetic" | "geometric",
  "number_of_steps": int,
  "number_of_simulations": int,
  "dividends": \[
  { "date": "YYYY-MM-DD", "amount": float },
  { "date": "YYYY-MM-DD", "amount": float },
  ...\]
}