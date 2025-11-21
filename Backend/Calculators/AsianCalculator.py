# ---------------------------------------------------------
# Filename: AsianCalculator.py
# Description
#   Pricing and Greeks for an ASIAN option with dividends.
#   The function `calculate_option_value` expects a `data`
#   dictionary similar to the other calculators.
#
#   Dividends:
#     - `data["dividends"]` is a list of dicts:
#         {"date": "YYYY-MM-DD", "amount": float}
#     - These discrete dividends are converted into a
#       continuous dividend yield q using the Utils:
#         calculate_present_value_dividends
#         calc_continuous_dividend_yield
# ---------------------------------------------------------

from typing import Dict, Any
import math
import numpy as np
from Utils import *
def calculate_option_value(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute the theoretical price and Greeks of an Asian option with dividends.

    Required keys in `data`:
      - type: "call" or "put"
      - exercise_style: should be "asian"
      - start_date: "YYYY-MM-DD"
      - start_time: "HH:MM:SS" or "AM"/"PM"
      - expiration_date: "YYYY-MM-DD"
      - expiration_time: "HH:MM:SS" or "AM"/"PM"
      - strike: float
      - stock_price: float
      - volatility: float (either %, e.g. 20, or decimal, e.g. 0.20)
      - interest_rate: float (either %, e.g. 1.5, or decimal, e.g. 0.015)
      - average_type: "arithmetic" or "geometric"
      - number_of_steps / n_fixings: int
      - number_of_simulations / mc_sims: int
      - dividends: list of {"date": "YYYY-MM-DD", "amount": float}
    """

    # 1) Initialize outputs with safe defaults
    option_price = 0.0
    delta_sensitivity = 0.0
    gamma_sensitivity = 0.0
    rho_sensitivity   = 0.0
    theta_sensitivity = 0.0
    vega_sensitivity  = 0.0

    # 2) Basic logging (helpful during development)
    print(f"Option type: {data['type']}")
    print(f"Exercise style: {data['exercise_style']}")
    print(f"Start date: {data['start_date']}")
    print(f"Start time: {data['start_time']}")
    print(f"Expiration date: {data['expiration_date']}")
    print(f"Expiration time: {data['expiration_time']}")
    print(f"Strike: {data['strike']}")
    print(f"Stock price: {data['stock_price']}")
    print(f"Volatility (raw): {data['volatility']}")
    print(f"Interest rate (raw): {data['interest_rate']}")
    print(f"Average type: {data.get('average_type', 'arithmetic')}")
    print(f"Dividends: {data.get('dividends', [])}")

    # 3) Time to maturity (ACT/365) via util
    time_to_maturity_in_years = calculate_time_to_maturity(
        data["start_date"],
        data["start_time"],
        data["expiration_date"],
        data["expiration_time"],
    )

    # 4) Core numerical inputs (normalize % vs decimals)
    initial_stock_price = float(data["stock_price"])
    strike_price = float(data["strike"])
    volatility = normalize_interest_rate(data["volatility"])
    risk_free_interest_rate = normalize_interest_rate(data["interest_rate"])

    # 5) Dividends: discrete list -> PV -> continuous yield q
    dividends_list = data.get("dividends", [])
    present_value_dividends = calculate_present_value_dividends(
        dividends_list,
        data["start_date"],
        data["expiration_date"],
        risk_free_interest_rate,
    )
    continuous_dividend_yield = calc_continuous_dividend_yield(
        initial_stock_price,
        present_value_dividends,
        time_to_maturity_in_years,
    )

    print(f"Present value of dividends: {present_value_dividends}")
    print(f"Implied continuous dividend yield q: {continuous_dividend_yield}")

    # 6) Asian parameters & Monte Carlo settings
    average_price_type = data.get("average_type", "arithmetic").lower()

    # allow both naming conventions
    number_of_fixings = int(data.get("n_fixings", data.get("number_of_steps", 12)))
    number_of_simulations = int(
        data.get("mc_sims", data.get("number_of_simulations", 100_000))
    )
    random_seed = int(data.get("seed", 42))
    monte_carlo_time_step_in_years = float(
        data.get(
            "mc_dt",
            time_to_maturity_in_years / max(number_of_fixings, 1),
        )
    )

    is_call_option = (data["type"].lower() == "call")

    # 7) Choose pricing method based on averaging type
    if average_price_type == "geometric":

        option_price = _asian_geometric_closed_form_price(
            initial_stock_price,
            strike_price,
            risk_free_interest_rate,
            continuous_dividend_yield,
            volatility,
            time_to_maturity_in_years,
            number_of_fixings,
            is_call=is_call_option,
        )

        (
            delta_sensitivity,
            gamma_sensitivity,
            vega_sensitivity,
            theta_sensitivity,
            rho_sensitivity,
        ) = _bump_and_reprice_asian(
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
        )

    else:
        # Default: arithmetic average via Monte Carlo with continuous q
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
        )

        (
            delta_sensitivity,
            gamma_sensitivity,
            vega_sensitivity,
            theta_sensitivity,
            rho_sensitivity,
        ) = _bump_and_reprice_asian(
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
        )

    # 8) Package and round the result
    return {
        "theoretical_price": round(option_price, 4),
        "delta": round(delta_sensitivity, 4),
        "gamma": round(gamma_sensitivity, 6),
        "rho": round(rho_sensitivity, 4),
        "theta": round(theta_sensitivity, 4),
        "vega": round(vega_sensitivity, 4),
    }


# =========================================================
# Pricing: geometric closed-form and arithmetic Monte Carlo
# =========================================================

def _asian_geometric_closed_form_price(
    initial_stock_price,
    strike_price,
    risk_free_interest_rate,
    continuous_dividend_yield,
    volatility,
    time_to_maturity_in_years,
    number_of_fixings,
    is_call=True,
) -> float:
    """
    Closed-form price for a geometric Asian option with equally spaced fixings.

    Uses Black–Scholes dynamics with continuous dividend yield q.
    """
    try:
        geometric_mean_drift = (
            (risk_free_interest_rate - continuous_dividend_yield)
            - 0.5 * volatility**2
        ) * (number_of_fixings + 1) / (2.0 * number_of_fixings)

        geometric_mean_volatility = volatility * math.sqrt(
            (number_of_fixings + 1)
            * (2 * number_of_fixings + 1)
            / (6.0 * number_of_fixings**2)
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
            math.log(geometric_mean_spot_price / strike_price)
            + (
                risk_free_interest_rate
                - continuous_dividend_yield
                + 0.5 * effective_volatility**2
            )
            * time_to_maturity_in_years
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
                geometric_mean_spot_price
                * dividend_carry_factor
                / discount_factor
                * normal_cumulative_d1
                - strike_price * normal_cumulative_d2
            )
        else:
            return discount_factor * (
                strike_price * normal_cumulative_minus_d2
                - geometric_mean_spot_price
                * dividend_carry_factor
                / discount_factor
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
):
    """
    Monte Carlo pricer for an arithmetic Asian option
    with continuous dividend yield q (no discrete price jumps).
    """
    random_number_generator = np.random.default_rng(random_seed)

    time_step_in_years = float(monte_carlo_time_step_in_years)
    if time_step_in_years <= 0:
        time_step_in_years = time_to_maturity_in_years / max(number_of_fixings, 1)

    number_of_time_steps = max(
        1,
        int(math.ceil(time_to_maturity_in_years / time_step_in_years)),
    )
    time_step_in_years = time_to_maturity_in_years / number_of_time_steps

    if number_of_fixings > 0:
        fixing_times_in_years = np.linspace(
            time_step_in_years,
            time_to_maturity_in_years,
            num=number_of_fixings,
        )
    else:
        fixing_times_in_years = np.array([time_to_maturity_in_years])

    number_of_path_batches = (
        number_of_simulations
        if not use_antithetic_variates
        else number_of_simulations // 2
    )
    if number_of_path_batches <= 0:
        number_of_path_batches = 1

    discount_factor = math.exp(-risk_free_interest_rate * time_to_maturity_in_years)
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

            average_stock_price = running_sum_of_fixing_prices / max(
                number_of_fixings, 1
            )

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
            is_call=is_call,
        )
        # Hook for a proper control variate implementation.
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
):
    """
    Compute Greeks for an Asian option using bump-and-reprice.

    Uses the same continuous dividend yield q for all bumped prices.
    """
    spot_price_shift = 0.01 * initial_stock_price if initial_stock_price != 0 else 0.01
    volatility_shift = 0.01 * volatility if volatility != 0 else 0.001
    interest_rate_shift = 0.0001
    time_shift_in_years = 1.0 / 365.0  # one day

    def price_function(stock_price_, volatility_, interest_rate_, maturity_):

        if average_price_type == "geometric":
            return _asian_geometric_closed_form_price(
                stock_price_,
                strike_price,
                interest_rate_,
                continuous_dividend_yield,
                volatility_,
                maturity_,
                number_of_fixings,
                is_call=is_call,
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
        )

    base_option_price = price_function(
        initial_stock_price,
        volatility,
        risk_free_interest_rate,
        time_to_maturity_in_years,
    )

    # Delta and Gamma (spot bumps)
    option_price_up_spot = price_function(
        initial_stock_price + spot_price_shift,
        volatility,
        risk_free_interest_rate,
        time_to_maturity_in_years,
    )
    option_price_down_spot = price_function(
        initial_stock_price - spot_price_shift,
        volatility,
        risk_free_interest_rate,
        time_to_maturity_in_years,
    )
    delta_sensitivity = (
        option_price_up_spot - option_price_down_spot
    ) / (2.0 * spot_price_shift)

    gamma_sensitivity = (
        option_price_up_spot - 2.0 * base_option_price + option_price_down_spot
    ) / (spot_price_shift**2)

    # Vega (volatility bumps)
    option_price_up_volatility = price_function(
        initial_stock_price,
        volatility + volatility_shift,
        risk_free_interest_rate,
        time_to_maturity_in_years,
    )
    option_price_down_volatility = price_function(
        initial_stock_price,
        volatility - volatility_shift,
        risk_free_interest_rate,
        time_to_maturity_in_years,
    )
    vega_sensitivity = (
        option_price_up_volatility - option_price_down_volatility
    ) / (2.0 * volatility_shift)

    # Rho (interest rate bumps)
    option_price_up_interest_rate = price_function(
        initial_stock_price,
        volatility,
        risk_free_interest_rate + interest_rate_shift,
        time_to_maturity_in_years,
    )
    option_price_down_interest_rate = price_function(
        initial_stock_price,
        volatility,
        risk_free_interest_rate - interest_rate_shift,
        time_to_maturity_in_years,
    )
    rho_sensitivity = (
        option_price_up_interest_rate - option_price_down_interest_rate
    ) / (2.0 * interest_rate_shift)

    # Theta (time to maturity bumps)
    shorter_time_to_maturity_in_years = max(
        time_to_maturity_in_years - time_shift_in_years,
        1e-8,
    )
    longer_time_to_maturity_in_years = time_to_maturity_in_years + time_shift_in_years

    option_price_shorter_time = price_function(
        initial_stock_price,
        volatility,
        risk_free_interest_rate,
        shorter_time_to_maturity_in_years,
    )
    option_price_longer_time = price_function(
        initial_stock_price,
        volatility,
        risk_free_interest_rate,
        longer_time_to_maturity_in_years,
    )

    theta_sensitivity = (
        option_price_shorter_time - option_price_longer_time
    ) / (2.0 * time_shift_in_years)

    return (
        float(delta_sensitivity),
        float(gamma_sensitivity),
        float(vega_sensitivity),
        float(theta_sensitivity),
        float(rho_sensitivity),
    )
