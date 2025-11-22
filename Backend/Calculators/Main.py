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
    print("Hi, welcome to our Option Calculator!\n")

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
    print(
        "Do you want to enter new data (type 'new') or use the existing data in input.json (type 'json')?\n"
        "If you put data directly into the JSON file, please adhere strictly to the input_schema; otherwise, the calculation will not run.\n"
    )

    while True:
        choice = input("Your choice (new/json): ").strip().lower()

        if choice in ("new", "json"):
            break
        else:
            print("Invalid input. Please type 'new' or 'json' to proceed.")


    #If new input dialog make dialog... else skip:
    if choice == "new":
        input_obj['type'] = ask_until_valid_string("Which option type do you want to calculate? (call|put): ", {"call", "put"})
        input_obj['exercise_style'] = ask_until_valid_string("Which exercise style do you want (american|european|asian|binary): ", {"american", "european", "asian", "binary"} )


        while True:
            input_obj['start_date'] = ask_until_valid_date("When does the option start? (YYYY-MM-DD): ")
            input_obj['start_time'] = ask_until_valid_time("At what time does the option start? (HH:MM:SS or AM/PM): ")
            input_obj['expiration_date'] = ask_until_valid_date("When does the option expire? (YYYY-MM-DD): ")
            input_obj['expiration_time'] = ask_until_valid_time("At what time does the option expire? (HH:MM:SS or AM/PM): ")
            if validate_start_expiration(input_obj['start_date'], input_obj['start_time'], input_obj['expiration_date'], input_obj['expiration_time']):
                break
            print("The start date must be before the expiration date.")

        input_obj['stock_price'] = ask_until_valid_number("Enter stock price (>= 0.01): ", minimum=0.01, exclusive_minimum=False)
        input_obj['strike'] = ask_until_valid_number("Enter strike price (>= 0.01): ", minimum=0.01, exclusive_minimum=False)

        while True:
            vol_input = input("Enter volatility (> 0), e.g. 0.20 for 20%: ")
            try:
                input_obj['volatility'] = validate_volatility(vol_input)
                break
            except ValueError as e:
                print(f"Error: {e}")

        while True:
            ir_input = input("Enter interest rate (percent), e.g. 1.5 for 1.5%: ")
            try:
                input_obj['interest_rate'] = validate_interest_rate(ir_input)
                break
            except ValueError as e:
                print(f"Error: {e}")

        if input_obj['exercise_style'] == 'binary':
            input_obj['binary_payoff_structure'] = ask_until_valid_string("Binary option type (cash | asset | custom): ", {"cash", "asset", "custom"})
            if input_obj['binary_payoff_structure'] == "custom":
                input_obj['binary_payout'] = ask_until_valid_number("Enter binary payout (>= 0.01): ", minimum=0.01, exclusive_minimum=False)

        if input_obj['exercise_style'] == 'asian':
            input_obj['average_type'] = ask_until_valid_string("Enter average type (arithmetic|geometric): ", {"arithmetic", "geometric"})

        if input_obj['exercise_style'] == 'asian' or input_obj['exercise_style'] == 'american':
            input_obj['number_of_steps'] = ask_until_valid_integer("Enter number of steps (>= 1) for MC-Simulation: ", minimum=1,
                                                                   maximum=1000)
            input_obj['number_of_simulations'] = ask_until_valid_integer("Enter number of simulations (>= 1): ",
                                                                         minimum=1, maximum=1000000)

        input_obj['dividends'] = input_dividends(input_obj['start_date'], input_obj['expiration_date'])



    else:
        # Transform all Stings to lowercase if person wrote to json file. Otherwise this step happens directly when user inserts new value
        input_obj['type'] = input_obj['type'].lower()
        input_obj['exercise_style'] = input_obj['exercise_style'].lower()
        input_obj['start_time'] = input_obj['start_time'].lower()
        input_obj['expiration_time'] = input_obj['expiration_time'].lower()
        input_obj['average_type'] = input_obj['average_type'].lower()
        input_obj['binary_payoff_structure'] = input_obj['binary_payoff_structure'].lower()


    #Validate that Object fits to our input_schema.json, safe the updated object
    try:
        with open(input_path, 'w', encoding='utf-8') as f:
            json.dump(input_obj, f, ensure_ascii=False, indent=4)
        jsonschema.validate(instance=input_obj, schema=input_schema)
        print("The input is valid! You are only seconds away from the option price.")
    except jsonschema.ValidationError as e:
        raise

    # Select the corresponding calculator and calculate results
    output_obj = None
    match input_obj['exercise_style']:
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

