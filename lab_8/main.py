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
  description = {
    "model": "Clasificador de potabilidad de agua con XGBoost.",
    "problem": "Determinar si una muestra de agua es potable (1) o no potable (0).",
    "input": {
      "ph": "Nivel de acidez del agua",
      "Hardness": "Capacidad de precipitar jabón (mg/L)",
      "Solids": "Cantidad de sólidos disueltos (ppm)",
      "Chloramines": "Cantidad de cloraminas (ppm)",
      "Sulfate": "Cantidad de sulfatos disueltos (ppm)",
      "Conductivity": "Conductividad eléctrica del agua (µS/cm)",
      "Organic_carbon": "Cantidad de carbono orgánico (ppm)",
      "Trihalomethanes": "Cantidad de trihalometanos (µg/L)",
      "Turbidity": "Turbidez (NTU)"
    },
    "output": {
      "potabilidad": "1 si es agua potable, 0 si es no potable"
    }
  }
  return description

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
