import json
import jsonschema
import os
from AmericanCalculator import calculate_option_value as calcOptionAmerican
from AsianCalculator import calculate_option_value as calcOptionAsian
from BinaryCalculator import calculate_option_value as calcOptionBinary
from EuropeanCalculator import calculate_option_value as calcOptionEuropean
from ValidateInput import *

# ---------------------------------------------------------
# Filename: Main.py
# Created: 2025-11-17
# Description: Reads input.json, Calculates the Corresponding results, Writes it into output.json
# ---------------------------------------------------------


def calculate_option():
    print("Hi, welcome to our Option Calculator! ")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_path = os.path.join(dir_path, '..', 'Input', 'input.json')
    input_scheme = os.path.join(dir_path, '..', 'Input', 'input_scheme.json')
    output_path = os.path.join(dir_path, '..', 'Output', 'output.json')

    # Path to Input in input.json... :)
    with open(input_path, 'r') as f:
        input_obj = json.load(f)

    with open(input_scheme, 'r') as f:
        input_scheme = json.load(f)




    choice = input("Use JSON file input? (y/n): ").strip().lower()

    if choice == "n":
        #small and capital letters do not matter at this point
        create_or_update_input_json()


    #Transform all non-monthly Stings to lowercase
    input_obj['type'] = input_obj['type'].lower()
    input_obj['exercise_style'] = input_obj['exercise_style'].lower()
    input_obj['start_time'] = input_obj['start_time'].lower()
    input_obj['expiration_time'] = input_obj['expiration_time'].lower()
    input_obj['average_type'] = input_obj['average_type'].lower()

    try:
        jsonschema.validate(instance=input_obj, schema=input_scheme)
        print("The input is valid! You are only seconds away from the option price.")
    except jsonschema.ValidationError as e:
        print("Something in the JSON is invalid:", e.message)


    # Select the corresponding calculator and calculate results and put into output.json
    output_obj = None
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
        print("Something went wrong! No result to display.")
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
        print(f"An error occurred while calculating or saving the output: {e}")
    return output_obj



if __name__ == '__main__':
   calculate_option()

