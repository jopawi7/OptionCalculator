import os
import json
import re
from datetime import datetime


#Function that ask for a valid answer until is valid
def ask_until_valid_string(prompt, valid_options):
    valid_options = {opt.lower() for opt in valid_options}
    while True:
        val = input(prompt).strip().lower()
        if val in valid_options:
            return val
        print(f"Invalid input. Please enter one of: {', '.join(valid_options)}.")


def ask_until_valid_date(prompt):
    date_pattern = r"^\d{4}-\d{2}-\d{2}$"
    regex = re.compile(date_pattern)
    while True:
        val = input(prompt).strip()
        if regex.fullmatch(val):
            return val
        print("Invalid date format. Please use YYYY-MM-DD.")


def ask_until_valid_time(prompt):
    time_pattern = r"^(?:([01]\d|2[0-3]):[0-5]\d:[0-5]\d|[aA][mM]|[pP][mM])$"
    regex = re.compile(time_pattern)
    while True:
        val = input(prompt).strip()
        if regex.fullmatch(val):
            return val
        print("Invalid time format. Please use HH:MM:SS (24h) or am/pm.")


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



def validate_start_expiration(start_date, start_time, expiration_date, expiration_time):
    def parse_dt(date_str, time_str):
        # Zeitstrings wie „am“/„pm“ wandeln wir auf feste Zeiten um
        if time_str.lower() == 'am':
            time_str = '09:30:00'
        elif time_str.lower() == 'pm':
            time_str = '16:00:00'
        # Ansonsten nehmen wir die genaue Zeit an
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")

    start_dt = parse_dt(start_date, start_time)
    exp_dt = parse_dt(expiration_date, expiration_time)

    return start_dt <= exp_dt

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

def validate_interest_rate(value):
    if value is None:
        raise ValueError("Interest rate is missing.")
    try:
        ir = float(str(value).replace(',', '.'))
    except Exception:
        raise ValueError("Interest rate must be a number.")
    if ir > 1:
        ir = ir / 100.0
    return ir




