from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

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
class Dividend(BaseModel):
    date: str
    amount: float

class OptionInput(BaseModel):
    type: str
    exercise_style: str = Field(..., alias='style')
    start_date: str
    start_time: str
    expiration_date: str
    expiration_time: str
    strike: float
    stock_price: float
    volatility: float
    interest_rate: float
    dividends: Optional[List[Dividend]] = []

app = FastAPI()

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

@app.post("/calculate/")
async def calculate(input_data: OptionInput):
    try:
        output_obj = calculateOptionWithData(input_data.dict())
        return output_obj
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

def calculateOptionWithData(input_obj):
    exercise_style = input_obj.get('exercise_style', '').lower()
    match exercise_style:
        case "american":
            return calcOptionAmerican(input_obj)
        case "asian":
            return calcOptionAsian(input_obj)
        case "binary":
            return calcOptionBinary(input_obj)
        case "european":
            return calcOptionEuropean(input_obj)
        case _:
            raise ValueError(f"Invalid exercise style: {input_obj.get('exercise_style')}")