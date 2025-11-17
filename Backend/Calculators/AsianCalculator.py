# ---------------------------------------------------------
# Description
#   Pricing and Greeks for an ASIAN option.
#   The function `calculate_option_value` expects a `data`
#   dictionary similar to the CBOE option calculator inputs,
#   plus a few extra fields for Asian options and Monte Carlo.
#
# Notes
#   - This module assumes `data` has already been loaded and
#     validated elsewhere (for example, from a file).
#   - The goal is to keep the pricing logic explicit and
#     readable, not to hide steps behind abstractions.
# ---------------------------------------------------------

from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import math
import numpy as np


def calculate_option_value(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute the theoretical price and Greeks of an Asian option.

    The input `data` should contain at least:
      - type: "CALL" or "PUT"
      - exercise_style: "American" or "European"
      - start_date: "YYYY-MM-DD"
      - start_time: "HH:MM:SS" or "AM"/"PM"
      - expiration_date: "YYYY-MM-DD"
      - expiration_time: "HH:MM:SS" or "AM"/"PM"
      - strike: float
      - stock_price: float
      - volatility: float in percent (e.g. 20.0 → 20%)
      - interest_rate: float in percent (e.g. 1.5 → 1.5%)
      - dividends: list of dividend entries (see helpers below)

    Optional Asian-specific fields:
      - average_type: "arithmetic" or "geometric" (default: "arithmetic")
      - n_fixings: number of averaging points (default: 12)
      - mc_sims: number of Monte Carlo paths (default: 100_000)
      - mc_dt: Monte Carlo time step in years (default: T / n_fixings)
      - seed: RNG seed for reproducibility (default: 42)
      - dividend_yield: continuous dividend yield in percent
                        (used in the geometric closed-form model)

    Supported dividend formats in data["dividends"]:
      1) Single ex-date:
         {"date": "YYYY-MM-DD", "amount": float}
      2) Recurrent schedule:
         {"start_date": "YYYY-MM-DD",
          "day_interval": int,
          "amount": float,
          "end_date": "YYYY-MM-DD" (optional)}
         → expanded into multiple ex-dates.

    Returns:
      Dictionary with:
        - "theoretical_price"
        - "delta", "gamma", "rho", "theta", "vega"
    """

    # 1) Initialize outputs with safe defaults
    option_price = 0.0
    delta_sensitivity = 0.0
    gamma_sensitivity = 0.0
    rho_sensitivity   = 0.0
    theta_sensitivity = 0.0
    vega_sensitivity  = 0.0

    # 2) Basic logging (useful during development or debugging)
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
    print(f"Dividends: {data.get('dividends', [])}")

    # 3) Parse dates and convert percentages to decimals
    start_datetime = _parse_datetime(data["start_date"], data["start_time"])
    expiration_datetime = _parse_datetime(data["expiration_date"], data["expiration_time"])

    if expiration_datetime <= start_datetime:
        raise ValueError("Expiration must be after start date/time.")

    # Time to maturity in years, ACT/365 convention
    time_to_maturity_in_years = (
        expiration_datetime - start_datetime
    ).total_seconds() / (365.0 * 24 * 3600.0)

    # Core numerical inputs
    initial_stock_price = float(data["stock_price"])
    strike_price = float(data["strike"])
    volatility = float(data["volatility"]) / 100.0
    risk_free_interest_rate = float(data["interest_rate"]) / 100.0
    continuous_dividend_yield = float(data.get("dividend_yield", 0.0)) / 100.0

    # 4) Normalize dividend information
    dividend_events_list = _expand_dividends(
        data.get("dividends", []),
        start_datetime=start_datetime,
        end_datetime=expiration_datetime
    )

    dividend_schedule_in_years = _dividends_to_year_times(
        dividend_events_list,
        start_datetime
    )

    # Keep only dividends between 0 and T
    dividend_schedule_in_years = [
        (time, amount)
        for (time, amount) in dividend_schedule_in_years
        if 0.0 < time <= time_to_maturity_in_years
    ]

    # 5) Read Asian option parameters and Monte Carlo settings
    average_price_type = data.get("average_type", "arithmetic").lower()

    number_of_fixings = int(data.get("n_fixings", 12))
    number_of_simulations = int(data.get("mc_sims", 100_000))
    random_seed = int(data.get("seed", 42))
    monte_carlo_time_step_in_years = float(
        data.get(
            "mc_dt",
            time_to_maturity_in_years / max(number_of_fixings, 1)
        )
    )

    is_call_option = (data["type"].upper() == "CALL")

    # 6) Choose pricing method based on averaging type
    if average_price_type == "geometric":

        if dividend_schedule_in_years and continuous_dividend_yield == 0.0:
            print(
                "[NOTICE] Discrete dividends provided but dividend_yield (q) is 0. "
                "The geometric closed-form with pure discrete dividends is only an approximation. "
                "Consider using a non-zero dividend_yield or switching to Monte Carlo."
            )

        option_price = _asian_geometric_closed_form_price(
            initial_stock_price,
            strike_price,
            risk_free_interest_rate,
            continuous_dividend_yield,
            volatility,
            time_to_maturity_in_years,
            number_of_fixings,
            is_call=is_call_option
        )

        delta_sensitivity, gamma_sensitivity, vega_sensitivity, theta_sensitivity, rho_sensitivity = \
            _bump_and_reprice_asian(
                initial_stock_price,
                strike_price,
                risk_free_interest_rate,
                continuous_dividend_yield,
                volatility,
                time_to_maturity_in_years,
                number_of_fixings,
                number_of_simulations,
                monte_carlo_time_step_in_years,
                random_seed,
                average_price_type="geometric",
                is_call=is_call_option,
                dividend_schedule=dividend_schedule_in_years
            )

    else:
        # Default: arithmetic average via Monte Carlo with discrete dividends
        option_price = _asian_arithmetic_monte_carlo_price(
            initial_stock_price,
            strike_price,
            risk_free_interest_rate,
            continuous_dividend_yield,
            volatility,
            time_to_maturity_in_years,
            number_of_fixings,
            number_of_simulations,
            monte_carlo_time_step_in_years,
            random_seed,
            is_call=is_call_option,
            use_antithetic_variates=True,
            use_control_variate_technique=True,
            dividend_schedule=dividend_schedule_in_years
        )

        delta_sensitivity, gamma_sensitivity, vega_sensitivity, theta_sensitivity, rho_sensitivity = \
            _bump_and_reprice_asian(
                initial_stock_price,
                strike_price,
                risk_free_interest_rate,
                continuous_dividend_yield,
                volatility,
                time_to_maturity_in_years,
                number_of_fixings,
                number_of_simulations,
                monte_carlo_time_step_in_years,
                random_seed,
                average_price_type="arithmetic",
                is_call=is_call_option,
                dividend_schedule=dividend_schedule_in_years
            )

    # 7) Package and round the result
    return {
        "theoretical_price": round(option_price, 4),
        "delta": round(delta_sensitivity, 4),
        "gamma": round(gamma_sensitivity, 6),
        "rho": round(rho_sensitivity, 4),
        "theta": round(theta_sensitivity, 4),
        "vega": round(vega_sensitivity, 4),
    }


# =========================================================
# Helpers: dates and dividends
# =========================================================

def _parse_datetime(date_string: str, time_string: str) -> datetime:
    """
    Convert a date string and a time string into a datetime object.

    If time_string is "AM" or "PM", a default market time is used:
      - "AM" → 09:30:00
      - "PM" → 15:30:00

    Otherwise, time_string is expected to be "HH:MM:SS".
    """
    if time_string in ("AM", "PM"):
        full_time_string = "09:30:00" if time_string == "AM" else "15:30:00"
    else:
        full_time_string = time_string

    return datetime.fromisoformat(f"{date_string} {full_time_string}")


def _expand_dividends(dividend_definitions: List[Dict[str, Any]],
                      start_datetime: datetime,
                      end_datetime: datetime) -> List[Tuple[datetime, float]]:
    """
    Normalize raw dividend inputs into a list of (ex_dividend_datetime, amount).

    Supports:
      - Single ex-date:
          {"date": "YYYY-MM-DD", "amount": float}
      - Recurrent schedule:
          {"start_date": "YYYY-MM-DD",
           "day_interval": int,
           "amount": float,
           "end_date": "YYYY-MM-DD" (optional)}

    Rules:
      - Dividends outside [start_datetime, end_datetime] are ignored.
      - If end_date is missing in a recurrent schedule, end_datetime is used.
      - If several dividends fall on the same date, amounts are merged.
    """
    output_list: List[Tuple[datetime, float]] = []

    for dividend_definition in dividend_definitions:

        # Case 1: single fixed date
        if "date" in dividend_definition:
            ex_dividend_datetime = datetime.fromisoformat(
                f"{dividend_definition['date']} 00:00:00"
            )
            amount = float(dividend_definition["amount"])

            if start_datetime <= ex_dividend_datetime <= end_datetime and amount != 0.0:
                output_list.append((ex_dividend_datetime, amount))

            continue

        # Case 2: recurrent schedule by day interval
        if "start_date" in dividend_definition and "day_interval" in dividend_definition:
            current_dividend_date = datetime.fromisoformat(
                f"{dividend_definition['start_date']} 00:00:00"
            )
            day_interval_in_days = int(dividend_definition["day_interval"])
            amount = float(dividend_definition["amount"])
            last_dividend_date = datetime.fromisoformat(
                f"{dividend_definition.get('end_date', end_datetime.date().isoformat())} 00:00:00"
            )

            if day_interval_in_days <= 0:
                continue

            while current_dividend_date <= last_dividend_date and current_dividend_date <= end_datetime:
                if start_datetime <= current_dividend_date <= end_datetime and amount != 0.0:
                    output_list.append((current_dividend_date, amount))

                current_dividend_date += timedelta(days=day_interval_in_days)

    # Sort by date and merge exact duplicates
    output_list.sort(key=lambda x: x[0])

    merged_dividend_list: List[Tuple[datetime, float]] = []
    for ex_dividend_datetime, amount in output_list:
        if merged_dividend_list and merged_dividend_list[-1][0] == ex_dividend_datetime:
            merged_dividend_list[-1] = (
                ex_dividend_datetime,
                merged_dividend_list[-1][1] + amount
            )
        else:
            merged_dividend_list.append((ex_dividend_datetime, amount))

    return merged_dividend_list


def _dividends_to_year_times(dividend_events: List[Tuple[datetime, float]],
                             start_datetime: datetime) -> List[Tuple[float, float]]:
    """
    Convert (ex_dividend_datetime, amount) into (time_in_years, amount)
    using an ACT/365 convention.
    """
    output_list: List[Tuple[float, float]] = []

    for ex_dividend_datetime, amount in dividend_events:
        time_from_start_in_years = (
            ex_dividend_datetime - start_datetime
        ).total_seconds() / (365.0 * 24 * 3600.0)

        output_list.append((time_from_start_in_years, amount))

    return output_list


# =========================================================
# Pricing: geometric closed-form and arithmetic Monte Carlo
# =========================================================

def _asian_geometric_closed_form_price(initial_stock_price, strike_price,
                                       risk_free_interest_rate,
                                       continuous_dividend_yield,
                                       volatility, time_to_maturity_in_years,
                                       number_of_fixings, is_call=True) -> float:
    """
    Closed-form price for a geometric Asian option with equally spaced fixings.

    Assumptions:
      - Black–Scholes dynamics with continuous dividend yield.
      - Fixings are equally spaced in [0, T].
      - Only the geometric average is modeled analytically.

    With purely discrete dividends, there is no exact closed form; in that case,
    this function relies on the continuous dividend yield as an approximation.
    """
    try:
        geometric_mean_drift = (
            (risk_free_interest_rate - continuous_dividend_yield) -
            0.5 * volatility**2
        ) * (number_of_fixings + 1) / (2.0 * number_of_fixings)

        geometric_mean_volatility = volatility * math.sqrt(
            (number_of_fixings + 1) * (2 * number_of_fixings + 1) /
            (6.0 * number_of_fixings**2)
        )

        geometric_mean_spot_price = initial_stock_price * math.exp(
            geometric_mean_drift * time_to_maturity_in_years
        )

        effective_volatility = geometric_mean_volatility * math.sqrt(
            time_to_maturity_in_years
        )

        if effective_volatility <= 0:
            return 0.0

        black_scholes_d1 = (
            math.log(geometric_mean_spot_price / strike_price) +
            (risk_free_interest_rate - continuous_dividend_yield +
             0.5 * effective_volatility**2) * time_to_maturity_in_years
        ) / effective_volatility

        black_scholes_d2 = black_scholes_d1 - effective_volatility

        normal_cumulative_d1 = 0.5 * (1 + math.erf(black_scholes_d1 / math.sqrt(2)))
        normal_cumulative_d2 = 0.5 * (1 + math.erf(black_scholes_d2 / math.sqrt(2)))
        normal_cumulative_minus_d1 = 1 - normal_cumulative_d1
        normal_cumulative_minus_d2 = 1 - normal_cumulative_d2

        discount_factor = math.exp(
            -risk_free_interest_rate * time_to_maturity_in_years
        )
        dividend_carry_factor = math.exp(
            -continuous_dividend_yield * time_to_maturity_in_years
        )

        if is_call:
            return discount_factor * (
                geometric_mean_spot_price * dividend_carry_factor / discount_factor
                * normal_cumulative_d1
                - strike_price * normal_cumulative_d2
            )
        else:
            return discount_factor * (
                strike_price * normal_cumulative_minus_d2
                - geometric_mean_spot_price * dividend_carry_factor / discount_factor
                * normal_cumulative_minus_d1
            )

    except Exception:
        return 0.0


def _asian_arithmetic_monte_carlo_price(
    initial_stock_price,
    strike_price,
    risk_free_interest_rate,
    continuous_dividend_yield,
    volatility,
    time_to_maturity_in_years,
    number_of_fixings,
    number_of_simulations,
    monte_carlo_time_step_in_years,
    random_seed,
    is_call=True,
    use_antithetic_variates=True,
    use_control_variate_technique=True,
    dividend_schedule=None
):
    """
    Monte Carlo pricer for an arithmetic Asian option with discrete dividends.

    Steps:
      1) Build a simulation time grid and a grid of fixing times.
      2) Simulate the log-price under the risk-neutral measure:
           d ln S = (r - q - 0.5 * sigma^2) dt + sigma dW.
      3) At ex-dividend times, apply cash drops to S.
      4) Record S at each fixing time and compute the arithmetic average.
      5) Compute and discount the payoff.
      6) Optionally use antithetic variates and a geometric control variate.
    """
    if dividend_schedule is None:
        dividend_schedule = []

    random_number_generator = np.random.default_rng(random_seed)

    time_step_in_years = float(monte_carlo_time_step_in_years)
    if time_step_in_years <= 0:
        time_step_in_years = time_to_maturity_in_years / max(number_of_fixings, 1)

    number_of_time_steps = max(
        1,
        int(math.ceil(time_to_maturity_in_years / time_step_in_years))
    )
    time_step_in_years = time_to_maturity_in_years / number_of_time_steps

    if number_of_fixings > 0:
        fixing_times_in_years = np.linspace(
            time_step_in_years,
            time_to_maturity_in_years,
            num=number_of_fixings
        )
    else:
        fixing_times_in_years = np.array([time_to_maturity_in_years])

    dividend_times_in_years = np.array(
        [t for (t, _) in dividend_schedule],
        dtype=float
    )
    dividend_amounts = np.array(
        [amount for (_, amount) in dividend_schedule],
        dtype=float
    )
    time_tolerance_in_years = time_step_in_years / 2.0

    number_of_path_batches = (
        number_of_simulations if not use_antithetic_variates
        else number_of_simulations // 2
    )
    if number_of_path_batches <= 0:
        number_of_path_batches = 1

    discount_factor = math.exp(
        -risk_free_interest_rate * time_to_maturity_in_years
    )
    drift_per_time_step = (
        risk_free_interest_rate
        - continuous_dividend_yield
        - 0.5 * volatility**2
    ) * time_step_in_years
    volatility_times_square_root_time_step = (
        volatility * math.sqrt(time_step_in_years)
    )

    option_payoffs = []

    for _ in range(number_of_path_batches):

        standard_normal_random_numbers = random_number_generator.standard_normal(
            (1, number_of_time_steps)
        )
        standard_normal_random_number_sets = [standard_normal_random_numbers]

        if use_antithetic_variates:
            standard_normal_random_number_sets.append(-standard_normal_random_numbers)

        for standard_normal_path_block in standard_normal_random_number_sets:

            stock_price = initial_stock_price
            current_time_in_years = 0.0
            next_fixing_index = 0
            running_sum_of_fixing_prices = 0.0

            for step_index in range(number_of_time_steps):

                stock_price *= math.exp(
                    drift_per_time_step
                    + volatility_times_square_root_time_step
                    * float(standard_normal_path_block[0, step_index])
                )

                current_time_in_years += time_step_in_years

                # Apply dividends if current time is close to an ex-div time
                if dividend_times_in_years.size > 0:
                    dividend_time_mask = (
                        np.abs(dividend_times_in_years - current_time_in_years)
                        <= time_tolerance_in_years
                    )

                    if dividend_time_mask.any():
                        total_dividend_amount = float(
                            dividend_amounts[dividend_time_mask].sum()
                        )
                        stock_price = max(stock_price - total_dividend_amount, 1e-12)

                # Capture fixings that have just been reached
                while (
                    next_fixing_index < len(fixing_times_in_years)
                    and current_time_in_years + 1e-12
                    >= fixing_times_in_years[next_fixing_index]
                ):
                    running_sum_of_fixing_prices += stock_price
                    next_fixing_index += 1

            # If some fixings are still pending (due to rounding), repeat last price
            while next_fixing_index < len(fixing_times_in_years):
                running_sum_of_fixing_prices += stock_price
                next_fixing_index += 1

            average_stock_price = running_sum_of_fixing_prices / max(number_of_fixings, 1)

            if is_call:
                option_payoff = max(average_stock_price - strike_price, 0.0)
            else:
                option_payoff = max(strike_price - average_stock_price, 0.0)

            option_payoffs.append(option_payoff)

    crude_monte_carlo_estimate = discount_factor * np.array(
        option_payoffs, dtype=float
    ).mean()

    if use_control_variate_technique:
        geometric_price = _asian_geometric_closed_form_price(
            initial_stock_price,
            strike_price,
            risk_free_interest_rate,
            continuous_dividend_yield,
            volatility,
            time_to_maturity_in_years,
            number_of_fixings,
            is_call=is_call
        )
        # Hook for a proper control variate implementation.
        # Currently we leave crude_monte_carlo_estimate unchanged if geometric_price > 0.
        if geometric_price > 0:
            pass

    return float(crude_monte_carlo_estimate)


def _bump_and_reprice_asian(
    initial_stock_price,
    strike_price,
    risk_free_interest_rate,
    continuous_dividend_yield,
    volatility,
    time_to_maturity_in_years,
    number_of_fixings,
    number_of_simulations,
    monte_carlo_time_step_in_years,
    random_seed,
    average_price_type: str,
    is_call: bool,
    dividend_schedule: List[Tuple[float, float]] = None
):
    """
    Compute Greeks for an Asian option using bump-and-reprice.

    The same random seed is reused for each bump (common random
    numbers) to reduce Monte Carlo noise.

    Finite-difference scheme:
      - Delta  ≈ [P(S + dS) - P(S - dS)] / (2 dS)
      - Gamma  ≈ [P(S + dS) - 2 P(S) + P(S - dS)] / (dS^2)
      - Vega   ≈ [P(sigma + dv) - P(sigma - dv)] / (2 dv)
      - Rho    ≈ [P(r + dr) - P(r - dr)] / (2 dr)
      - Theta  ≈ [P(T - dT) - P(T + dT)] / (2 dT)

    For geometric averaging with no discrete dividends, the closed-form
    geometric Asian price is used; otherwise, the Monte Carlo pricer is used.
    """
    spot_price_shift = 0.01 * initial_stock_price if initial_stock_price != 0 else 0.01
    volatility_shift = 0.01 * volatility if volatility != 0 else 0.001
    interest_rate_shift = 0.0001
    time_shift_in_years = 1.0 / 365.0  # one day

    def price_function(stock_price_,
                       volatility_,
                       interest_rate_,
                       maturity_):

        if average_price_type == "geometric" and not dividend_schedule:
            return _asian_geometric_closed_form_price(
                stock_price_,
                strike_price,
                interest_rate_,
                continuous_dividend_yield,
                volatility_,
                maturity_,
                number_of_fixings,
                is_call=is_call
            )

        return _asian_arithmetic_monte_carlo_price(
            stock_price_,
            strike_price,
            interest_rate_,
            continuous_dividend_yield,
            volatility_,
            maturity_,
            number_of_fixings,
            number_of_simulations,
            monte_carlo_time_step_in_years,
            random_seed,
            is_call=is_call,
            use_antithetic_variates=True,
            use_control_variate_technique=True,
            dividend_schedule=dividend_schedule or []
        )

    base_option_price = price_function(
        initial_stock_price, volatility, risk_free_interest_rate, time_to_maturity_in_years
    )

    # Delta and Gamma (spot bumps)
    option_price_up_spot = price_function(
        initial_stock_price + spot_price_shift,
        volatility,
        risk_free_interest_rate,
        time_to_maturity_in_years
    )
    option_price_down_spot = price_function(
        initial_stock_price - spot_price_shift,
        volatility,
        risk_free_interest_rate,
        time_to_maturity_in_years
    )
    delta_sensitivity = (
        option_price_up_spot - option_price_down_spot
    ) / (2.0 * spot_price_shift)

    gamma_sensitivity = (
        option_price_up_spot -
        2.0 * base_option_price +
        option_price_down_spot
    ) / (spot_price_shift**2)

    # Vega (volatility bumps)
    option_price_up_volatility = price_function(
        initial_stock_price,
        volatility + volatility_shift,
        risk_free_interest_rate,
        time_to_maturity_in_years
    )
    option_price_down_volatility = price_function(
        initial_stock_price,
        volatility - volatility_shift,
        risk_free_interest_rate,
        time_to_maturity_in_years
    )
    vega_sensitivity = (
        option_price_up_volatility - option_price_down_volatility
    ) / (2.0 * volatility_shift)

    # Rho (interest rate bumps)
    option_price_up_interest_rate = price_function(
        initial_stock_price,
        volatility,
        risk_free_interest_rate + interest_rate_shift,
        time_to_maturity_in_years
    )
    option_price_down_interest_rate = price_function(
        initial_stock_price,
        volatility,
        risk_free_interest_rate - interest_rate_shift,
        time_to_maturity_in_years
    )
    rho_sensitivity = (
        option_price_up_interest_rate - option_price_down_interest_rate
    ) / (2.0 * interest_rate_shift)

    # Theta (time to maturity bumps)
    shorter_time_to_maturity_in_years = max(
        time_to_maturity_in_years - time_shift_in_years,
        1e-8
    )
    longer_time_to_maturity_in_years = (
        time_to_maturity_in_years + time_shift_in_years
    )

    option_price_shorter_time = price_function(
        initial_stock_price,
        volatility,
        risk_free_interest_rate,
        shorter_time_to_maturity_in_years
    )
    option_price_longer_time = price_function(
        initial_stock_price,
        volatility,
        risk_free_interest_rate,
        longer_time_to_maturity_in_years
    )

    theta_sensitivity = (
        option_price_shorter_time - option_price_longer_time
    ) / (2.0 * time_shift_in_years)

    return (
        float(delta_sensitivity),
        float(gamma_sensitivity),
        float(vega_sensitivity),
        float(theta_sensitivity),
        float(rho_sensitivity)
    )
