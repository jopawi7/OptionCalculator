from __future__ import annotations

import json
import os
from typing import Any

from AmericanCalculator import calculate_option_value as calc_american
from AsianCalculator import calculate_option_value as calc_asian
from BinaryCalculator import calculate_option_value as calc_binary
from EuropeanCalculator import calculate_option_value as calc_european

from input_utils import (
    ask_yes_no,
    build_input_interactively,
    validate_input_data,
    generate_dividends_from_stream,
)


def _default_paths() -> tuple[str, str, str]:
    """
    Returns default paths for input.json, input_schema.json, and output.json,
    relative to this file.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "..", "Input", "input.json")
    schema_path = os.path.join(base_dir, "..", "Input", "input_schema.json")
    output_path = os.path.join(base_dir, "..", "Output", "output.json")
    return input_path, schema_path, output_path


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _select_calculator(exercise_style: str):
    style = exercise_style.lower()
    mapping = {
        "european": calc_european,
        "american": calc_american,
        "asian": calc_asian,
        "binary": calc_binary,
    }
    if style not in mapping:
        valid = ", ".join(sorted(mapping))
        raise ValueError(
            f"Unknown exercise_style={exercise_style!r}. "
            f"Valid values are: {valid}."
        )
    return mapping[style]


def calculate_option():
    """
    Main process:
      1) Ask whether to read parameters from input.json or enter them manually.
      2) Validate the inputs (schema + custom validation).
      3) Generate dividends from the stream and combine them with discrete ones.
      4) Call the corresponding calculator.
      5) Print results and save them to output.json.
    """
    input_path, schema_path, output_path = _default_paths()

    print(">>> Option Calculator")
    use_file = ask_yes_no("Do you want to read the parameters from input.json? (y/n): ")

    # 1) Get input_obj (from file or interactively)
    if use_file:
        try:
            input_obj = _load_json(input_path)
        except FileNotFoundError:
            print(f"Input file not found at {input_path}.")
            if ask_yes_no("Do you want to enter the data manually instead? (y/n): "):
                input_obj = build_input_interactively()
            else:
                print("Aborting: no input data provided.")
                return
    else:
        input_obj = build_input_interactively()
        if ask_yes_no("Do you want to save this input to input.json for future runs? (y/n): "):
            os.makedirs(os.path.dirname(input_path), exist_ok=True)
            with open(input_path, "w", encoding="utf-8") as f:
                json.dump(input_obj, f, indent=2)
            print(f"Input saved to {input_path}")

    # 2) Load schema (if exists) and validate
    try:
        schema_obj = _load_json(schema_path)
    except FileNotFoundError:
        schema_obj = None
        print("Warning: input_schema.json not found. Only custom validation will be applied.")

    try:
        validate_input_data(input_obj, schema_obj)
        print("✔ Input validated successfully.")
    except ValueError as exc:
        print(f"Input validation error: {exc}")
        return

    # 3) Generate additional dividends from stream (if configured)
    stream_cfg = input_obj.get("dividend_stream")
    dividends_final = list(input_obj.get("dividends", []))

    if stream_cfg:
        generated = generate_dividends_from_stream(
            stream_cfg,
            input_obj["start_date"],
            input_obj["expiration_date"],
        )
        dividends_final.extend(generated)

        # Sort dividends by date for clarity
        try:
            dividends_final.sort(key=lambda d: d["date"])
        except KeyError:
            pass

        input_obj["dividends"] = dividends_final
        # Optional: remove stream config to avoid clutter in calculators
        input_obj.pop("dividend_stream", None)

    # 4) Select calculator
    try:
        calculator = _select_calculator(input_obj["exercise_style"])
    except KeyError:
        print("Input JSON must contain 'exercise_style'.")
        return
    except ValueError as exc:
        print(exc)
        return

    # 5) Calculate
    try:
        output_obj = calculator(input_obj)
    except Exception as exc:
        print(f"An error occurred while calculating the option value: {exc}")
        return

    # 6) Display results
    print("\n=== RESULTS ===")
    if isinstance(output_obj, dict):
        for key, value in output_obj.items():
            print(f"{key:20s}: {value}")
    else:
        print(output_obj)

    # 7) Save results
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_obj, f, indent=2)
        print(f"\n✔ Results successfully written to: {output_path}")
    except Exception as exc:
        print(f"An error occurred while saving the output: {exc}")

    return output_obj


if __name__ == "__main__":
    calculate_option()
