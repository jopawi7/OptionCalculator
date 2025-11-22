from __future__ import annotations

from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional

import jsonschema


DATE_FMT = "%Y-%m-%d"


# =========================================================
# Internal helpers
# =========================================================
def _parse_date(value: str) -> date:
    return datetime.strptime(value, DATE_FMT).date()


# =========================================================
# Interactive input functions
# =========================================================

def ask_until_valid_string(prompt: str, valid_values: List[str]) -> str:
    """
    Ask the user for a string until it matches one of the valid_values
    (case-insensitive). Returns the chosen value preserving the original
    casing from valid_values.
    """
    valid_lower = {v.lower(): v for v in valid_values}
    while True:
        s = input(prompt).strip()
        if not s:
            print("Please enter a value.")
            continue
        key = s.lower()
        if key in valid_lower:
            return valid_lower[key]
        print(f"Invalid value. Valid options are: {', '.join(valid_values)}")



def ask_yes_no(prompt: str) -> bool:
    """
    Ask a yes/no question.
    Returns True for 'y'/'yes', False for 'n'/'no'.
    """
    while True:
        s = input(prompt).strip().lower()
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False
        print("Please answer with 'y' or 'n'.")


def ask_until_valid_date(prompt: str) -> str:
    """
    Ask for a date string until a valid YYYY-MM-DD is entered.
    """
    while True:
        s = input(prompt).strip()
        try:
            _ = _parse_date(s)
            return s
        except Exception:
            print(f"Invalid date. Please use format {DATE_FMT}.")


def ask_until_valid_date_in_range(
    prompt: str,
    min_date_str: str,
    max_date_str: str,
) -> str:
    """
    Ask for a date within a given [min_date, max_date] range (inclusive).
    """
    min_d = _parse_date(min_date_str)
    max_d = _parse_date(max_date_str)
    while True:
        s = input(prompt).strip()
        try:
            d = _parse_date(s)
        except Exception:
            print(f"Invalid date. Please use format {DATE_FMT}.")
            continue
        if d < min_d or d > max_d:
            print(
                f"Date must be between {min_date_str} and {max_date_str} "
                f"(inclusive)."
            )
            continue
        return s


def ask_until_valid_time(prompt: str) -> str:
    """
    Ask for a time string.
    Accepts:
      - empty string (returns "")
      - 'am' / 'pm'
      - 'HH:MM:SS' in 24h format
    """
    while True:
        s = input(prompt).strip()
        if not s:
            return ""
        lower = s.lower()
        if lower in {"am", "pm"}:
            return lower
        try:
            datetime.strptime(s, "%H:%M:%S")
            return s
        except ValueError:
            print("Invalid time format. Use 'HH:MM:SS' 24h or 'am'/'pm', or leave empty.")


def ask_until_valid_number(
    prompt: str,
    minimum: Optional[float] = None,
    exclusive_minimum: bool = False,
) -> float:
    """
    Ask for a numeric value until a valid float is provided and optional
    minimum constraints are satisfied.
    """
    while True:
        s = input(prompt).strip()
        try:
            v = float(s)
        except ValueError:
            print("Please enter a valid number.")
            continue

        if minimum is not None:
            if exclusive_minimum and not (v > minimum):
                print(f"Value must be > {minimum}.")
                continue
            if not exclusive_minimum and not (v >= minimum):
                print(f"Value must be >= {minimum}.")
                continue

        return v


def ask_until_valid_integer(
    prompt: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    """
    Ask for an integer until a valid int is provided and optional
    min/max constraints are satisfied.
    """
    while True:
        s = input(prompt).strip()
        try:
            v = int(s)
        except ValueError:
            print("Please enter a valid integer number.")
            continue

        if minimum is not None and v < minimum:
            print(f"Value must be >= {minimum}.")
            continue
        if maximum is not None and v > maximum:
            print(f"Value must be <= {maximum}.")
            continue

        return v


def ask_until_valid_amount(
    prompt: str,
    minimum: float = 0.0,
) -> float:
    """
    Ask for a monetary amount with a given minimum (inclusive).
    """
    return ask_until_valid_number(prompt, minimum=minimum, exclusive_minimum=False)


def input_dividends(start_date_str: str, expiration_date_str: str) -> List[Dict[str, Any]]:
    """
    Interactive input of discrete dividends, one by one:
      - date must be in [start_date, expiration_date]
      - amount >= 0

    Returns a list of {"date": "...", "amount": ...}.
    """
    dividends: List[Dict[str, Any]] = []
    print("\n=== Discrete dividends ===")
    if not ask_yes_no("Do you want to add discrete dividends? (y/n): "):
        return dividends

    while True:
        date_str = ask_until_valid_date_in_range(
            "Dividend date (YYYY-MM-DD): ",
            start_date_str,
            expiration_date_str,
        )
        amount = ask_until_valid_amount("Dividend amount (>= 0): ", minimum=0.0)
        dividends.append(
            {
                "date": date_str,
                "amount": float(amount),
            }
        )
        if not ask_yes_no("Add another discrete dividend? (y/n): "):
            break

    return dividends


# =========================================================
# Dividend stream generation
# =========================================================
def generate_dividends_from_stream(
    stream_info: Dict[str, Any],
    contract_start_date: str,
    contract_expiration_date: str,
    max_payments: int = 12,
) -> List[Dict[str, Any]]:
    """
    Convert a 'dividend_stream' description into a list of discrete payments.

    stream_info expects:
      - 'first_payment_date': 'YYYY-MM-DD'
      - 'amount': float >= 0
      - 'interval_days': int > 0

    Only payments with:
      - date >= contract_start_date
      - date <= contract_expiration_date
      - at most 'max_payments' payments
    are generated.

    'Today' is interpreted as contract_start_date.
    """
    if not stream_info:
        return []

    try:
        first_date_str = stream_info.get("first_payment_date")
        amount = float(stream_info.get("amount", 0.0))
        interval_days = int(stream_info.get("interval_days", 0))
    except (TypeError, ValueError):
        # Malformed data → no payments generated
        return []

    if amount <= 0 or interval_days <= 0 or not first_date_str:
        return []

    try:
        first_date = _parse_date(first_date_str)
        start_date = _parse_date(contract_start_date)
        exp_date = _parse_date(contract_expiration_date)
    except Exception:
        return []

    # Starting point: first payment that occurs on or after start_date
    current = first_date

    if current < start_date:
        delta_days = (start_date - current).days
        # number of intervals to skip (ceil)
        k = (delta_days + interval_days - 1) // interval_days
        current = current + timedelta(days=k * interval_days)

    payments: List[Dict[str, Any]] = []
    count = 0

    while current <= exp_date and count < max_payments:
        payments.append(
            {
                "date": current.strftime(DATE_FMT),
                "amount": amount,
            }
        )
        current = current + timedelta(days=interval_days)
        count += 1

    return payments


# =========================================================
# Input validation (JSON + business logic)
# =========================================================
def validate_input_data(data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> None:
    """
    Validate the input dictionary.
      - If `schema` is provided, validate against jsonschema.
      - Then perform additional business checks:
          * dates and ordering: start_date < expiration_date
          * numeric constraints where required
          * dividends structure
          * dividend_stream structure
          * binary payoff structure (if exercise_style == 'binary')

    Raises ValueError if something is invalid.
    """
    # 1) jsonschema validation (general structure and types)
    if schema is not None:
        try:
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.ValidationError as exc:
            raise ValueError(f"Input does not conform to schema: {exc.message}") from exc

    # 2) Additional checks
    try:
        start_date_str = data["start_date"]
        expiration_date_str = data["expiration_date"]
        start_dt = _parse_date(start_date_str)
        exp_dt = _parse_date(expiration_date_str)
    except Exception as exc:
        raise ValueError("start_date / expiration_date must be valid dates YYYY-MM-DD.") from exc

    if exp_dt <= start_dt:
        raise ValueError("expiration_date must be strictly after start_date.")

    # Numeric constraints similar to those in the schema
    numeric_constraints = {
        "strike": (0.01, False),
        "stock_price": (0.01, False),
        "volatility": (0.0, True),  # exclusiveMinimum: 0
    }
    for key, (min_val, exclusive) in numeric_constraints.items():
        if key in data:
            try:
                v = float(data[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be a number.") from exc
            if exclusive and not (v > min_val):
                raise ValueError(f"{key} must be > {min_val}.")
            if not exclusive and not (v >= min_val):
                raise ValueError(f"{key} must be >= {min_val}.")

    # interest_rate: only check that it is numeric
    try:
        float(data["interest_rate"])
    except (TypeError, ValueError) as exc:
        raise ValueError("interest_rate must be a number.") from exc

    # Dividends: list of objects with 'date' + 'amount' >= 0
    dividends = data.get("dividends", [])
    if not isinstance(dividends, list):
        raise ValueError("dividends must be a list of {date, amount} objects.")
    for div in dividends:
        if not isinstance(div, dict):
            raise ValueError("Each dividend must be an object with keys 'date' and 'amount'.")
        if "date" not in div or "amount" not in div:
            raise ValueError("Each dividend must contain 'date' and 'amount'.")
        try:
            _parse_date(div["date"])
        except Exception as exc:
            raise ValueError(f"Invalid dividend date: {div['date']}") from exc
        try:
            amount = float(div["amount"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Dividend amount must be a number.") from exc
        if amount < 0:
            raise ValueError("Dividend amount must be >= 0.")

    # dividend_stream (if present)
    stream = data.get("dividend_stream")
    if stream is not None:
        if not isinstance(stream, dict):
            raise ValueError("dividend_stream must be an object.")
        amount = stream.get("amount", 0)
        interval_days = stream.get("interval_days", 0)
        first_payment_date = stream.get("first_payment_date")

        # amount
        try:
            amount = float(amount)
        except (TypeError, ValueError) as exc:
            raise ValueError("dividend_stream.amount must be a number.") from exc
        if amount < 0:
            raise ValueError("dividend_stream.amount must be >= 0.")

        # interval
        try:
            interval_days = int(interval_days)
        except (TypeError, ValueError) as exc:
            raise ValueError("dividend_stream.interval_days must be an integer.") from exc
        if interval_days <= 0:
            raise ValueError("dividend_stream.interval_days must be > 0.")

        # date
        if not first_payment_date:
            raise ValueError("dividend_stream.first_payment_date is required.")
        try:
            first_dt = _parse_date(first_payment_date)
            # start_dt, exp_dt already computed above
        except Exception as exc:
            raise ValueError("Invalid date in dividend_stream.first_payment_date.") from exc

        if first_dt > exp_dt:
            raise ValueError(
                "dividend_stream.first_payment_date must be on or before expiration_date."
            )

    # Binary payoff (only relevant if exercise_style == 'binary')
    if str(data.get("exercise_style", "")).lower() == "binary":
        payoff_type = data.get("binary_payoff_type")
        if payoff_type not in ("cash", "asset", "custom"):
            raise ValueError(
                "binary_payoff_type must be one of: 'cash', 'asset', 'custom'."
            )

        if "binary_payout" not in data:
            raise ValueError("binary_payout is required for binary options.")

        try:
            payout_val = float(data["binary_payout"])
        except (TypeError, ValueError) as exc:
            raise ValueError("binary_payout must be a number.") from exc
        if payout_val < 0:
            raise ValueError("binary_payout must be >= 0.")


# =========================================================
# Interactive construction of the input dictionary
# =========================================================
def _ask_expiration_date_after_start(start_date_str: str) -> str:
    """
    Ask for an expiration date that must be strictly after the given start_date.
    """
    while True:
        exp_str = ask_until_valid_date("Expiration date (YYYY-MM-DD): ")
        try:
            start_dt = _parse_date(start_date_str)
            exp_dt = _parse_date(exp_str)
        except Exception:
            print(f"Invalid date format, please use {DATE_FMT}.")
            continue
        if exp_dt <= start_dt:
            print("Expiration date must be after start date.")
        else:
            return exp_str


def build_input_interactively() -> Dict[str, Any]:
    """
    Ask the user for all required input data via console and return
    a dictionary with the same structure as input.json.
    """
    print("=== Interactive input mode ===")

    option_type = ask_until_valid_string(
        "Option type (call / put): ",
        ["call", "put"],
    )

    exercise_style = ask_until_valid_string(
        "Exercise style (american / european / asian / binary): ",
        ["american", "european", "asian", "binary"],
    )

    if exercise_style == "asian":
        print("\n=== Asian option average configuration ===")
        print("Average types:")
        print("  - 'arithmetic': arithmetic mean of the underlying prices")
        print("  - 'geometric' : geometric mean of the underlying prices")
        print("    (geometric often has a closed-form solution under Black-Scholes)")
        average_type = ask_until_valid_string(
            "Average type for Asian options (arithmetic / geometric): ",
            ["arithmetic", "geometric"],
        )
    else:
        average_type = "arithmetic"

    if exercise_style in ("asian", "american"):
        number_of_steps = ask_until_valid_integer(
            "Number of time steps (1 to 1000): ",
            minimum=1,
            maximum=1000,
        )
        number_of_simulations = ask_until_valid_integer(
            "Number of Monte Carlo simulations (1 to 100000): ",
            minimum=1,
            maximum=100_000,
        )
    else:
        number_of_steps = 20
        number_of_simulations = 1000

    if exercise_style == "binary":
        print("\n=== Binary option payoff configuration ===")
        print("Binary payoff types:")
        print("  - 'cash'   : pays a fixed cash amount if the option finishes in the money (1.0)")
        print("  - 'asset'  : pays the underlying asset price S_T if in the money")
        print("  - 'custom' : pays a fixed cash amount you specify if in the money")

        binary_payoff_type = ask_until_valid_string(
            "Binary payoff type (cash / asset / custom): ",
            ["cash", "asset", "custom"],
        )
        if binary_payoff_type == "custom":
            binary_payout = ask_until_valid_amount(
            "Binary payout at expiry (>= 0): ",
            minimum=0.0,)
    else:
        # Default values for non-binary options
        binary_payoff_type = "cash"
        binary_payout = 1.0

    start_date = ask_until_valid_date("Start date (YYYY-MM-DD): ")
    start_time = ask_until_valid_time(
        "Start time (HH:MM:SS 24h or am/pm) [empty allowed]: "
    )

    expiration_date = _ask_expiration_date_after_start(start_date)
    expiration_time = ask_until_valid_time(
        "Expiration time (HH:MM:SS 24h or am/pm) [empty allowed]: "
    )

    strike = ask_until_valid_number(
        "Strike price (>= 0.01): ",
        minimum=0.01,
        exclusive_minimum=False,
    )
    stock_price = ask_until_valid_number(
        "Current stock price (>= 0.01): ",
        minimum=0.01,
        exclusive_minimum=False,
    )
    volatility = ask_until_valid_number(
        "Volatility (e.g. 0.25 or 25%, must be > 0): ",
        minimum=0.0,
        exclusive_minimum=True,
    )
    interest_rate = ask_until_valid_number(
        "Risk-free interest rate (e.g. 0.03 or 3%): "
    )

    # Discrete dividends
    dividends = input_dividends(start_date, expiration_date)

    # Dividend stream
    dividend_stream: Optional[Dict[str, Any]] = None
    if ask_yes_no("Do you want to define a regular dividend stream? (y/n): "):
        first_payment_date = ask_until_valid_date_in_range(
            "First dividend payment date (YYYY-MM-DD): ",
            start_date,
            expiration_date,
        )
        amount_stream = ask_until_valid_amount(
            "Amount per dividend in the stream (>= 0): ",
            minimum=0.0,
        )
        interval_days = ask_until_valid_integer(
            "Interval between dividend payments in days (> 0): ",
            minimum=1,
        )
        dividend_stream = {
            "first_payment_date": first_payment_date,
            "amount": float(amount_stream),
            "interval_days": int(interval_days),
        }

    data: Dict[str, Any] = {
        "type": option_type,
        "exercise_style": exercise_style,
        "start_date": start_date,
        "start_time": start_time,
        "expiration_date": expiration_date,
        "expiration_time": expiration_time,
        "strike": float(strike),
        "stock_price": float(stock_price),
        "volatility": float(volatility),
        "interest_rate": float(interest_rate),
        "average_type": average_type,
        "number_of_steps": int(number_of_steps),
        "number_of_simulations": int(number_of_simulations),
        "dividends": dividends,
        "binary_payoff_type": binary_payoff_type,
        "binary_payout": float(binary_payout),
    }

    if dividend_stream is not None:
        data["dividend_stream"] = dividend_stream

    return data
