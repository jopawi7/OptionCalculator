import json
import  EuropeanAmericanCalculator
import  AsianCalculator
import  BinaryCalculator
import  BarrierCalculator

# ---------------------------------------------------------
# Filename: main.py
# Author: Jonas Patrick Witzel
# Created: 2025-10-30
# Description: Reads input.json, Calculates the Corresponding results, Writes it into output.json
# ---------------------------------------------------------


def calculate_option():
    print("Option calculation starts...")
    print("")

    # Path to Input in input.json... :)
    with open('../Input/input.json', 'r') as f:
        input_obj = json.load(f)

    # How to call the Variables
    # print("Input data loaded successfully!")
    # print(f"Option type: {input_obj['type']}")
    # print(f"Exercise style: {input_obj['exercise_style']}")
    # print(f"Start date: {input_obj['start_date']}")
    # print(f"Start time: {input_obj['start_time']}")
    # print(f"Expiration date: {input_obj['expiration_date']}")
    # print(f"Expiration time: {input_obj['expiration_time']}")
    # print(f"Strike: {input_obj['strike']}")
    # print(f"Stock price: {input_obj['stock_price']}")
    # print(f"Volatility: {input_obj['volatility']}")
    # print(f"Interest rate: {input_obj['interest_rate']}")
    # print(f"Dividend yield: {input_obj['dividend_yield']}")

    # Select the corresponding calculator and calculate results and put into output.json
    output_obj = None

    match input_obj['exercise_style'].lower():
        case "european" | "american":
            output_obj = EuropeanAmericanCalculator.calculateOptionValue(input_obj)
        case "asian":
            output_obj = AsianCalculator.calculateOptionValue(input_obj)
        case "binary":
            output_obj = BinaryCalculator.calculateOptionValue(input_obj)
        case "barrier":
            output_obj = BarrierCalculator.calculateOptionValue(input_obj)
        case _:
            raise ValueError("Invalid exercise style")

    # Print Results from Calulation
    if output_obj is None:
        print("Something went wrong!")
    else:
        print("")
        print("----- Result Summary -----")
        print(f"Symbol: {output_obj.get('symbol', 'N/A')}")
        print(f"Theoretical Price: {output_obj.get('theoretical_price', 'N/A')}")
        print(f"Delta: {output_obj.get('delta', 'N/A')}")
        print(f"Gamma: {output_obj.get('gamma', 'N/A')}")
        print(f"Rho: {output_obj.get('rho', 'N/A')}")
        print(f"Theta: {output_obj.get('theta', 'N/A')}")
        print(f"Vega: {output_obj.get('vega', 'N/A')}")

    # Safe the output
    try:
        with open('../Output/output.json', "w") as f:
            json.dump(output_obj, f, indent=2)
        print(f"Results successfully written to: output.json")
    except Exception as e:
        print(f"An error occured while calculating or saving the output: {e}")
    return output_obj



if __name__ == '__main__':
   calculate_option()

