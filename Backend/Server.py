from fastapi import FastAPI

from starlette.middleware.cors import CORSMiddleware
from Calculators.AmericanCalculator import calculate_option_value as calcOptionAmerican
from Calculators.AsianCalculator import calculate_option_value as calcOptionAsian
from Calculators.BinaryCalculator import calculate_option_value as calcOptionBinary
from Calculators.EuropeanCalculator import calculate_option_value as calcOptionEuropean


# ---------------------------------------------------------
# Filename: Server.py
# Author: Jonas Patrick Witzel
# Created: 2025-10-31
# Description: Use FastAPI() to connect Frontend with Backend
# ---------------------------------------------------------

#TODO - define FastAPI Interface for Frontend
# 1) pip install uvicorn
# 2) change in Folder Backend
# 3) uvicorn server:app --reload
# 4) die API läuft unter dem angezeigten link

app = FastAPI()

#Add CORSE Middleware so that Angular has access to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


#Post Endpoint for the calculatoon
@app.post("/calculate/")
async def calculate(input_data: dict):
    # Speichere Input temporär oder übergebe direkt
    output_obj = calculateOptionWithData(input_data)
    return output_obj


def calculateOptionWithData(input_obj):
    #calculteOptionWithData without input output logic
    match input_obj['exercise_style'].lower():
        case  "american":
            return calcOptionAmerican(input_obj)
        case "asian":
            return calcOptionAsian(input_obj)
        case "binary":
            return calcOptionBinary(input_obj)
        case "european":
            return calcOptionEuropean(input_obj)
        case _:
            raise ValueError("Invalid exercise style")