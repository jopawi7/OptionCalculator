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

    #Create paths to input and output file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_path = os.path.join(dir_path, '..', 'Input', 'input.json')
    input_schema = os.path.join(dir_path, '..', 'Input', 'input_schema.json')
    output_path = os.path.join(dir_path, '..', 'Output', 'output.json')
    output_schema = os.path.join(dir_path, '..', 'Output', 'output_schema.json')

    # Path to Input in input.json... :)
    with open(input_path, 'r') as f:
        input_obj = json.load(f)
    with open(input_schema, 'r') as f:
        input_schema = json.load(f)



    #TODO – Code input dialog and write Json Schema
    #choice = input("Use JSON file input? (y/n): ").strip().lower()

    #if choice == "n":
        #small and capital letters do not matter at this point
    #    create_or_update_input_json()








    #Transform all Stings to lowercase
    input_obj['type'] = input_obj['type'].lower()
    input_obj['exercise_style'] = input_obj['exercise_style'].lower()
    input_obj['start_time'] = input_obj['start_time'].lower()
    input_obj['expiration_time'] = input_obj['expiration_time'].lower()
    input_obj['average_type'] = input_obj['average_type'].lower()

    #TODO - valid date
    #Validate that Object fits to our input_schema.json
    try:
        jsonschema.validate(instance=input_obj, schema=input_schema)
        print("The input is valid! You are only seconds away from the option price.")
    except jsonschema.ValidationError as e:
        raise

    # Select the corresponding calculator and calculate results
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

    #Validate that output_obj fits to our defined output_schema.json
    with open(output_schema, 'r') as f:
        output_schema = json.load(f)

    #Usually this should not be possible but is an additional robustness proof
    #that only allows mathematical valid values.
    try:
        jsonschema.validate(instance=output_obj, schema=output_schema)
    except jsonschema.ValidationError as e:
        raise

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
        print(f"An error occurred while saving the output: {e}")
    return output_obj



if __name__ == '__main__':
   calculate_option()

