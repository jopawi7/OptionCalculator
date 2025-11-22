import re
from datetime import datetime, timedelta

# ---------------------------------------------------------
# Filename: UtilsInput.py
# Created: 2025-11-22
# Description: Summarize Input Object, Validate Inputs, Handling of dividend input
# ---------------------------------------------------------


#TODO - Summarizes input_object
#Print input.json
def print_input(input_obj, *bool_flags):
    keys = list(input_obj.keys())
    for i, flag in enumerate(bool_flags):
        if flag and i < len(keys):
            key = keys[i]
            value = input_obj[key]
            print(f"{key}: {value}")


#TODO - Input validation
#Asks until the string is valid
def ask_until_valid_string(prompt, valid_options):
    valid_options = {opt.lower() for opt in valid_options}
    while True:
        val = input(prompt).strip().lower()
        if val in valid_options:
            return val
        print(f"Invalid input. Please enter one of: {', '.join(valid_options)}.")

#Asks until the date is in a valid date formal
def ask_until_valid_date(prompt):
    date_pattern = r"^\d{4}-\d{2}-\d{2}$"
    regex = re.compile(date_pattern)
    while True:
        val = input(prompt).strip()
        if not regex.fullmatch(val):
            print("Invalid date format. Please use YYYY-MM-DD.")
            continue
        try:
            # Hier erfolgt die echte Validierung des Datums
            datetime.strptime(val, "%Y-%m-%d")
            return val
        except ValueError:
            print("Invalid date. Please enter a real calendar date.")


#Asls until the time is in a valid format
def ask_until_valid_time(prompt):
    time_pattern = r"^(?:([01]\d|2[0-3]):[0-5]\d:[0-5]\d|[aA][mM]|[pP][mM])$"
    regex = re.compile(time_pattern)
    while True:
        val = input(prompt).strip()
        if regex.fullmatch(val):
            return val
        print("Invalid time format. Please use HH:MM:SS (24h) or am/pm.")

#Asks until number is valid
def ask_until_valid_number(prompt, minimum=None, exclusive_minimum=False):
    while True:
        val_str = input(prompt).strip()
        # Also replaces komma wiht point to be able to calcuate
        val_str = val_str.replace(',', '.')
        try:
            val = float(val_str)
            if minimum is not None:
                if exclusive_minimum and not (val > minimum):
                    print(f"Please enter a number greater than {minimum}.")
                    continue
                elif not exclusive_minimum and not (val >= minimum):
                    print(f"Please enter a number greater than or equal to {minimum}.")
                    continue
            return val
        except ValueError:
            print("Invalid input. Please enter a valid number.")

#Asks until is an integer and valid
def ask_until_valid_integer(prompt, minimum=None, maximum=None):
    while True:
        val_str = input(prompt).strip()
        try:
            val = int(val_str)
            if minimum is not None and val < minimum:
                print(f"Please enter an integer greater than or equal to {minimum}.")
                continue
            if maximum is not None and val > maximum:
                print(f"Please enter an integer less than or equal to {maximum}.")
                continue
            return val
        except ValueError:
            print("Invalid input. Please enter a valid integer.")


#Validate start_date before expiration date
def validate_start_expiration(start_date, start_time, expiration_date, expiration_time):
    def parse_dt(date_str, time_str):
        if time_str.lower() == 'am':
            time_str = '09:30:00'
        elif time_str.lower() == 'pm':
            time_str = '16:00:00'
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")

    start_dt = parse_dt(start_date, start_time)
    exp_dt = parse_dt(expiration_date, expiration_time)

    return start_dt <= exp_dt

#Validates prompt for validate volatility and validate interest_rate
def prompt_and_validate(prompt_text, validate_func):
    while True:
        user_input = input(prompt_text)
        try:
            return validate_func(user_input)
        except ValueError as e:
            print(f"Error: {e}")

#Validates volatility
def validate_volatility(value):
    if value is None:
        raise ValueError("Volatility is missing.")
    try:
        vol = float(str(value).replace(',', '.'))
    except Exception:
        raise ValueError("Volatility must be a number.")
    if vol <= 0:
        raise ValueError("Volatility must be greater than 0.")
    return vol

#Validates interest rate
def validate_interest_rate(value):
    if value is None:
        raise ValueError("Interest rate is missing.")
    try:
        ir = float(str(value).replace(',', '.'))
    except Exception:
        raise ValueError("Interest rate must be a number.")
    return ir


#Ask for yes or no
def ask_yes_no(prompt):
    while True:
        ans = input(prompt + " (yes/no): ").strip().lower()
        if ans in ('yes', 'no'):
            return ans == 'yes'
        print("Please answer 'yes' or 'no'.")

#Aks until valid date in range
def ask_until_valid_date_in_range(prompt, start_date, end_date):
    date_pattern = r"^\d{4}-\d{2}-\d{2}$"
    regex = re.compile(date_pattern)
    while True:
        date_str = input(prompt).strip()
        if not regex.fullmatch(date_str):
            print("Invalid date format. Please use YYYY-MM-DD.")
            continue
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        if date_obj < datetime.strptime(start_date, "%Y-%m-%d") or date_obj > datetime.strptime(end_date, "%Y-%m-%d"):
            print(f"Date must be between {start_date} and {end_date}.")
            continue
        return date_str

#Asks until the amount is valid (float)
def ask_until_valid_amount(prompt):
    while True:
        val_str = input(prompt).strip().replace(',', '.')
        try:
            val = float(val_str)
            if val < 0:
                print("Amount must be 0 or greater.")
                continue
            return val
        except ValueError:
            print("Invalid number. Please try again.")


#TODO - Input of dividend prompting
#Ask for discrete dividend payments
def input_dividends(start_date, expiration_date):
    dividends = []
    if not ask_yes_no("Do you want to enter dividends?"):
        return []
    print("Please enter dividends between the start and expiration dates:")
    while True:
        date = ask_until_valid_date_in_range("Enter dividend date (YYYY-MM-DD): ", start_date, expiration_date)
        amount = ask_until_valid_amount("Enter dividend amount (>= 0): ")
        dividends.append({'date': date, 'amount': amount})
        if not ask_yes_no("Add another dividend?"):
            break
    return dividends


#Function to create dividen streams
def generate_dividend_stream( start_date: str, expiration_date: str,dividend_amount: float,day_interval: int) -> list[dict]:
    dividends = []

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(expiration_date, "%Y-%m-%d")

    current_date = start
    while current_date <= end:
        dividends.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "amount": dividend_amount
        })
        current_date += timedelta(days=day_interval)

    return dividends

