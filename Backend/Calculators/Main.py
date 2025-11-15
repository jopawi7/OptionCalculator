import json
import os
from AmericanCalculator import calculate_option_value as calcOptionAmerican
from AsianCalculator import calculate_option_value as calcOptionAsian
from BinaryCalculator import calculate_option_value as calcOptionBinary
from EuropeanCalculator import calculate_option_value as calcOptionEuropean

# ---------------------------------------------------------
# Filename: Main.py
# Author:
# Created: 2025-10-30
# Description: Reads input.json, Calculates the Corresponding results, Writes it into output.json
# ---------------------------------------------------------


def calculate_option():
    print("Hi, welcome to our Option Calculator! ")
    print("")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_path = os.path.join(dir_path, '..', 'Input', 'input.json')
    output_path = os.path.join(dir_path, '..', 'Output', 'output.json')

    # Path to Input in input.json... :)
    with open(input_path, 'r') as f:
        input_obj = json.load(f)


    # Select the corresponding calculator and calculate results and put into output.json
    output_obj = None

    #TODO - value Validation for all variables
    #TODO - eventuell input output abfrage
    #TODO - Robustness Checks

    match input_obj['exercise_style'].lower():
        case "american":
            output_obj = calcOptionAmerican(input_obj)
        case "asian":
            output_obj = calcOptionAsian(input_obj)
        case "binary":
            output_obj = calcOptionBinary(input_obj)
        case "european":
            output_obj = calcOptionEuropean(input_obj)
        case _:
            raise ValueError("Invalid exercise style")

    # Print Results from Calculation
    if output_obj is None:
        print("Something went wrong!")
    else:
        print("")
        print("----- Result Summary -----")
        print(f"Theoretical Price: {output_obj.get('theoretical_price', 'N/A')}")
        print(f"Delta: {output_obj.get('delta', 'N/A')}")
        print(f"Gamma: {output_obj.get('gamma', 'N/A')}")
        print(f"Rho: {output_obj.get('rho', 'N/A')}")
        print(f"Theta: {output_obj.get('theta', 'N/A')}")
        print(f"Vega: {output_obj.get('vega', 'N/A')}")

    # Safe the output
    try:
        with open(output_path, "w") as f:
            json.dump(output_obj, f, indent=2)
        print(f"Results successfully written to: output.json")
    except Exception as e:
        print(f"An error occured while calculating or saving the output: {e}")
    return output_obj



if __name__ == '__main__':
   calculate_option()

