from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from Calculators.AmericanCalculator import calculate_option_value as calcOptionAmerican
from Calculators.AsianCalculator import calculate_option_value as calcOptionAsian
from Calculators.BinaryCalculator import calculate_option_value as calcOptionBinary
from Calculators.EuropeanCalculator import calculate_option_value as calcOptionEuropean

# ---------------------------------------------------------
# Filename: server.py
# Description: Use FastAPI() to connect Frontend with Backend,
# now with detailed validation error handler!
# ---------------------------------------------------------

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Exception Handler for Validation Errors ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"VALIDATION ERROR: {exc.errors()}")
    print(f"Request body: {exc.body}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": exc.body,
        },
    )

# Models
class Dividend(BaseModel):
    date: str
    amount: float

class OptionInput(BaseModel):
    type: str
    exercise_style: str
    start_date: str
    start_time: str
    expiration_date: str
    expiration_time: str
    strike: float
    stock_price: float
    volatility: float
    interest_rate: float
    average_type: str
    number_of_steps: int
    number_of_simulations: int
    dividends: List[Dividend] = []


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/api/price")
async def calculate(input_data: OptionInput):
    try:
        output_obj = calculateOptionWithData(input_data.dict())
        return output_obj
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

def calculateOptionWithData(input_obj):
    exercise_style = input_obj.get('exercise_style', '').lower()
    print(input_obj)
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
