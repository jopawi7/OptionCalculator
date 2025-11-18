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











def validate_date(date_text):
    try:
        datetime.strptime(date_text, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def validate_time(time_text):
    if time_text.upper() in ['AM', 'PM']:
        return True
    try:
        datetime.strptime(time_text, '%H:%M:%S')
        return True
    except ValueError:
        try:
            datetime.strptime(time_text, '%H:%M')
            return True
        except ValueError:
            return False

def get_valid_input(prompt, valid_options=None, is_float=False, is_int=False, min_val=None, max_val=None, validate_func=None):
    while True:
        value = input(prompt)
        if valid_options and value.lower() not in [opt.lower() for opt in valid_options]:
            print(f'Ungültige Eingabe, bitte eine der Optionen: {valid_options}')
            continue
        if is_float:
            try:
                fvalue = float(value)
                if (min_val is not None and fvalue < min_val) or (max_val is not None and fvalue > max_val):
                    print(f"Wert muss zwischen {min_val} und {max_val} liegen.")
                    continue
                return fvalue
            except ValueError:
                print("Bitte eine gültige Kommazahl eingeben.")
                continue
        if is_int:
            try:
                ivalue = int(value)
                if (min_val is not None and ivalue < min_val) or (max_val is not None and ivalue > max_val):
                    print(f"Wert muss zwischen {min_val} und {max_val} liegen.")
                    continue
                return ivalue
            except ValueError:
                print("Bitte eine gültige ganze Zahl eingeben.")
                continue
        if validate_func:
            if not validate_func(value):
                print("Ungültiges Format.")
                continue
        return value

#Creates input.json if nonexistent. other
def create_or_update_input_json():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_path = os.path.join(dir_path, '..', 'Input', 'input.json')

    # Falls schon existiert: Laden und update, sonst neues Dict
    if os.path.isfile(input_path):
        with open(input_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    print("Hi, welcome to our Option Calculator! Leave fields empty to keep current value.")

    # Hilfsfunktion, die alten Wert anzeigt und neuen Wert fragt, ggf. übernommen wird
    def prompt(field_name, current_value, **kwargs):
        prompt_str = f"{field_name} [{current_value}]: "
        value = get_valid_input(prompt_str, **kwargs)
        return value if value != '' else current_value

    # Update Felder mit Abfrage, leer lassen behält alten Wert
    data["type"] = prompt("Optionstyp (call/put)", data.get("type", "call"), valid_options=["call", "put"])
    data["exercise_style"] = prompt("Ausübungsart (american/european/asian/binary)",
                                   data.get("exercise_style", "american"),
                                   valid_options=["american", "european", "asian", "binary"])
    data["start_date"] = prompt("Startdatum (YYYY-MM-DD)", data.get("start_date", "2025-11-16"), validate_func=validate_date)
    data["start_time"] = prompt("Startzeit (HH:MM:SS oder AM/PM)", data.get("start_time", "09:30:00"), validate_func=validate_time)
    data["expiration_date"] = prompt("Verfallsdatum (YYYY-MM-DD)", data.get("expiration_date", "2026-11-16"), validate_func=validate_date)
    data["expiration_time"] = prompt("Verfallszeit (HH:MM:SS oder AM/PM)", data.get("expiration_time", "16:00:00"), validate_func=validate_time)
    data["strike"] = prompt("Strike Preis", data.get("strike", 100.0), is_float=True, min_val=0.000001)
    data["stock_price"] = prompt("Aktueller Aktienkurs", data.get("stock_price", 100.0), is_float=True, min_val=0.000001)
    data["volatility"] = prompt("Volatilität (z.B. 0.20)", data.get("volatility", 0.20), is_float=True, min_val=0.0)
    data["interest_rate"] = prompt("Zinssatz (z.B. 1.5)", data.get("interest_rate", 1.5), is_float=True)

    # Average_type, Steps und Simulations nur bei Bedarf abfragen, sonst bleiben aktuelle Werte
    if data["exercise_style"].lower() == "asian":
        data["average_type"] = prompt("Average Type (arithmetic/geometric)", data.get("average_type", "arithmetic"), valid_options=["arithmetic", "geometric"])
    else:
        # behalten, aber nicht abfragen
        data["average_type"] = data.get("average_type", "arithmetic")

    data["number_of_steps"] = prompt("Anzahl Steps", data.get("number_of_steps", 100), is_int=True, min_val=1)
    data["number_of_simulations"] = prompt("Anzahl Simulationen", data.get("number_of_simulations", 10000), is_int=True, min_val=1)

    # Dividenden bleiben immer Liste; neue Dividenden nur hinzufügen
    print("Aktuelle Dividenden:")
    for d in data.get("dividends", []):
        print(f'  Datum: {d["date"]}, Betrag: {d["amount"]}')

    add_div = get_valid_input("Neue Dividenden hinzufügen? (ja/nein): ", valid_options=["ja", "nein"])
    while add_div.lower() == "ja":
        div_date = get_valid_input("Dividenden Datum (YYYY-MM-DD): ", validate_func=validate_date)
        div_amount = get_valid_input("Dividenden Betrag (positive Zahl): ", is_float=True, min_val=0.0)
        data.setdefault("dividends", []).append({"date": div_date, "amount": div_amount})
        add_div = get_valid_input("Weitere Dividenden hinzufügen? (ja/nein): ", valid_options=["ja", "nein"])

    # Speichern
    with open(input_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Eingabedaten wurden erfolgreich nach {input_path} gespeichert.")

if __name__ == "__main__":
    create_or_update_input_json()
