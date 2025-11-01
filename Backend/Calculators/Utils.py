from datetime import datetime
from math import exp, sqrt, log, erf, pi

# ---------------------------------------------------------
# Filename: Utils.py
# Author: Jonas Patrick Witzel
# Created: 2025-10-30
# Description: Some util functions that are needed for all calculations
# ---------------------------------------------------------

#Converts string from JSON-Object into Datetime format
def parse_datetime(date_str: str, time_str: str = None):
    """
    :param date_str: Format: "2025-10-30"
    :param time_str: Format: "09:30:18" or "AM"/"PM"
    :return: same object as daytime
    """
    if time_str:
        time_str = time_str.strip().upper()

        #Case 1: time_str is given as time
        if ":" in time_str:
            fmt = "%Y-%m-%d %H:%M:%S" if time_str.count(":") == 2 else "%Y-%m-%d %H:%M"
            return datetime.strptime(f"{date_str} {time_str}", fmt)

        #Case 2: time_str is given as AM or PM <- take 9AM for time AM and 15 for time PM
        elif time_str in ["AM", "PM"]:
            hour = 9 if time_str == "AM" else 15
            return datetime.strptime(f"{date_str} {hour}:00:00", "%Y-%m-%d %H:%M:%S")

        else:
            raise ValueError(f"Unknown Timeformat: {time_str}")

        #Case 3: no time_str is given
    return datetime.strptime(date_str, "%Y-%m-%d")

