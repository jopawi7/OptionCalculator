from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import math
import numpy as np


def calculate_option_value(data: Dict[str, Any]) -> Dict[str, float]:

    option_price = 0.0
    delta_sensitivity = 0.0
    gamma_sensitivity = 0.0
    rho_sensitivity   = 0.0
    theta_sensitivity = 0.0
    vega_sensitivity  = 0.0

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

    start_datetime = _parse_datetime(data["start_date"], data["start_time"])
    expiration_datetime = _parse_datetime(data["expiration_date"], data["expiration_time"])

    if expiration_datetime <= start_datetime:
        raise ValueError("Expiration must be after start date/time.")

    time_to_maturity_in_years = (expiration_datetime - start_datetime).total_seconds() / (365.0 * 24 * 3600.0)

    initial_stock_price = float(data["stock_price"])
    strike_price = float(data["strike"])
    volatility = float(data["volatility"]) / 100.0
    risk_free_interest_rate = float(data["interest_rate"]) / 100.0
    continuous_dividend_yield = float(data.get("dividend_yield", 0.0)) / 100.0

    dividend_events_list = _expand_dividends(
        data.get("dividends", []),
        start_datetime=start_datetime,
        end_datetime=expiration_datetime
    )

    dividend_schedule_in_years = _dividends_to_year_times(dividend_events_list, start_datetime)

    dividend_schedule_in_years = [
        (time, amount) for (time, amount) in dividend_schedule_in_years
        if 0.0 < time <= time_to_maturity_in_years
    ]

    average_price_type = data.get("average_type", "arithmetic").lower()

    number_of_fixings = int(data.get("n_fixings", 12))
    number_of_simulations = int(data.get("mc_sims", 100_000))
    random_seed = int(data.get("seed", 42))
    monte_carlo_time_step_in_years = float(data.get(
        "mc_dt",
        time_to_maturity_in_years / max(number_of_fixings, 1)
    ))

    is_call_option = (data["type"].upper() == "CALL")

    if average_price_type == "geometric":

        if dividend_schedule_in_years and continuous_dividend_yield == 0.0:
            print("[NOTICE] Discrete dividends provided but dividend_yield(q) = 0. "
                  "Closed-form geometric formula may be inaccurate. Use yield or Monte Carlo.")

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

    return {
        "theoretical_price": round(option_price, 4),
        "delta": round(delta_sensitivity, 4),
        "gamma": round(gamma_sensitivity, 6),
        "rho": round(rho_sensitivity, 4),
        "theta": round(theta_sensitivity, 4),
        "vega": round(vega_sensitivity, 4),
    }


def _parse_datetime(date_string: str, time_string: str) -> datetime:

    if time_string in ("AM", "PM"):
        full_time_string = "09:30:00" if time_string == "AM" else "15:30:00"
    else:
        full_time_string = time_string

    return datetime.fromisoformat(f"{date_string} {full_time_string}")


def _expand_dividends(dividend_definitions: List[Dict[str, Any]],
                      start_datetime: datetime,
                      end_datetime: datetime) -> List[Tuple[datetime, float]]:

    output_list: List[Tuple[datetime, float]] = []

    for dividend_definition in dividend_definitions:

        if "date" in dividend_definition:
            ex_dividend_datetime = datetime.fromisoformat(f"{dividend_definition['date']} 00:00:00")
            amount = float(dividend_definition["amount"])

            if start_datetime <= ex_dividend_datetime <= end_datetime and amount != 0.0:
                output_list.append((ex_dividend_datetime, amount))

            continue

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

    output_list: List[Tuple[float, float]] = []

    for ex_dividend_datetime, amount in dividend_events:
        time_from_start_in_years = (
            (ex_dividend_datetime - start_datetime).total_seconds() /
            (365.0 * 24 * 3600.0)
        )
        output_list.append((time_from_start_in_years, amount))

    return output_list


def _asian_geometric_closed_form_price(initial_stock_price, strike_price,
                                       risk_free_interest_rate,
                                       continuous_dividend_yield,
                                       volatility, time_to_maturity_in_years,
                                       number_of_fixings, is_call=True) -> float:

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

        discount_factor = math.exp(-risk_free_interest_rate * time_to_maturity_in_years)
        dividend_carry_factor = math.exp(-continuous_dividend_yield * time_to_maturity_in_years)

        if is_call:
            return discount_factor * (
                geometric_mean_spot_price * dividend_carry_factor / discount_factor *
                normal_cumulative_d1 -
                strike_price * normal_cumulative_d2
            )
        else:
            return discount_factor * (
                strike_price * normal_cumulative_minus_d2 -
                geometric_mean_spot_price * dividend_carry_factor / discount_factor *
                normal_cumulative_minus_d1
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

    if dividend_schedule is None:
        dividend_schedule = []

    random_number_generator = np.random.default_rng(random_seed)

    time_step_in_years = float(monte_carlo_time_step_in_years)
    if time_step_in_years <= 0:
        time_step_in_years = time_to_maturity_in_years / max(number_of_fixings, 1)

    number_of_time_steps = max(1, int(math.ceil(time_to_maturity_in_years / time_step_in_years)))
    time_step_in_years = time_to_maturity_in_years / number_of_time_steps

    fixing_times_in_years = (
        np.linspace(time_step_in_years if number_of_fixings > 0 else time_to_maturity_in_years,
                    time_to_maturity_in_years, num=number_of_fixings)
        if number_of_fixings > 0 else np.array([time_to_maturity_in_years])
    )

    dividend_times_in_years = np.array([t for (t, _) in dividend_schedule], dtype=float)
    dividend_amounts = np.array([amt for (_, amt) in dividend_schedule], dtype=float)
    time_tolerance_in_years = time_step_in_years / 2.0

    number_of_path_batches = (
        number_of_simulations if not use_antithetic_variates else number_of_simulations // 2
    )
    if number_of_path_batches <= 0:
        number_of_path_batches = 1

    discount_factor = math.exp(-risk_free_interest_rate * time_to_maturity_in_years)
    drift_per_time_step = (risk_free_interest_rate - continuous_dividend_yield -
                           0.5 * volatility**2) * time_step_in_years
    volatility_times_square_root_time_step = volatility * math.sqrt(time_step_in_years)

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
                    drift_per_time_step +
                    volatility_times_square_root_time_step *
                    float(standard_normal_path_block[0, step_index])
                )

                current_time_in_years += time_step_in_years

                if dividend_times_in_years.size > 0:
                    dividend_time_mask = np.abs(
                        dividend_times_in_years - current_time_in_years
                    ) <= time_tolerance_in_years

                    if dividend_time_mask.any():
                        total_dividend_amount = float(dividend_amounts[dividend_time_mask].sum())
                        stock_price = max(stock_price - total_dividend_amount, 1e-12)

                while (
                    next_fixing_index < len(fixing_times_in_years)
                    and current_time_in_years + 1e-12 >= fixing_times_in_years[next_fixing_index]
                ):
                    running_sum_of_fixing_prices += stock_price
                    next_fixing_index += 1

            while next_fixing_index < len(fixing_times_in_years):
                running_sum_of_fixing_prices += stock_price
                next_fixing_index += 1

            average_stock_price = running_sum_of_fixing_prices / max(number_of_fixings, 1)

            option_payoff = (
                max(average_stock_price - strike_price, 0.0)
                if is_call else
                max(strike_price - average_stock_price, 0.0)
            )

            option_payoffs.append(option_payoff)

    crude_monte_carlo_estimate = discount_factor * np.array(option_payoffs).mean()

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

        if geometric_price > 0:
            pass

    return float(crude_monte_carlo_estimate)


def _bump_and_reprice_asian(
