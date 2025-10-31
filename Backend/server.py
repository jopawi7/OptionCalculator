from fastapi import FastAPI
from Calculators.main import calculate_option

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

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.put("/calculate/")
async def update_item():
    #Schreibe in Input
    output_obj = calculate_option()
    #Nehme aus Outpu
    return output_obj