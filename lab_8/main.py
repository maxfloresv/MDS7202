import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
  title="API Laboratorio 8",
  description="Clasificación binaria de potabilidad del agua con XGBoost"
)

class Payload(BaseModel):
  ph: float
  Hardness: float
  Solids: float
  Chloramines: float
  Sulfate: float
  Conductivity: float
  Organic_carbon: float
  Trihalomethanes: float
  Turbidity: float

@app.get("/")
def home():
  return {"message": "Bienvenido a mi API con FastAPI"}

@app.post("/potabilidad/")
def predict(payload: Payload):
  try:
    with open("models/best_model.pkl", "rb") as f:
      model = pickle.load(f)
    data = payload.model_dump()

    X = pd.DataFrame([data], columns=[
      'ph',
      'Hardness',
      'Solids',
      'Chloramines',
      'Sulfate',
      'Conductivity',
      'Organic_carbon',
      'Trihalomethanes',
      'Turbidity'
    ])

    prediction = model.predict(X)[0]
    return {"potabilidad": int(prediction)}
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
