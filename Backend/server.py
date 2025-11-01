from fastapi import FastAPI
from Calculators.main import calculate_option
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware


# ---------------------------------------------------------
# Filename: server.py
# Author: Jonas Patrick Witzel
# Created: 2025-10-31
# Description: Use FastAPI() to connect Frontend with Backend
# ---------------------------------------------------------

#TODO - define FastAPI Interface for Frontend
# 1) pip install uvicorn
# 2) change in Folder Backend
# 3) uvicorn server:app --reload
# 4) die API läuft unter dem angezeigten link


#Denke beim Drücken auf Caluculate Option daran, das an die input.json weiterzuleiten

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
        case "european" | "american":
            from Calculators.EuropeanAmericanCalculator import calculateOptionValue as calcOptionEA
            return calcOptionEA(input_obj)
        case "asian":
            from Calculators.AsianCalculator import calculateOptionValue as calcOptionAsian
            return calcOptionAsian(input_obj)
        case "binary":
            from Calculators.BinaryCalculator import calculateOptionValue as calcOptionBinary
            return calcOptionBinary(input_obj)
        case "barrier":
            from Calculators.BarrierCalculator import calculateOptionValue as calcOptionBarrier
            return calcOptionBarrier(input_obj)
        case _:
            raise ValueError("Invalid exercise style")