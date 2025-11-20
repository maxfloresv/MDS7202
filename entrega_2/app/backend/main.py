import pickle
import os
import pandas as pd
from fastapi import FastAPI, HTTPException,status
from fastapi.middleware.cors import CORSMiddleware
import traceback
from pydantic import BaseModel
import logging
import joblib
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(
  title="API Entrega 2",
  description="Predicción de compra de productos de SodAI con XGBoost"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = f"{BASE_DIR}/models"
MODEL_PATH = f"{MODEL_DIR}/complete_trained_model.joblib"

model = None
model_info = {}

MODEL_COLUMNS = ['customer_id','product_id','week','customer_type','Y','X',
'num_deliver_per_week','brand','sub_category','segment','package','size']

def load_model_from_pickle():
    global model, model_info
    try: 
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No se encontró el archivo: {MODEL_PATH}")     
        with open(MODEL_PATH, 'rb') as f:
            model = joblib.load(f)
        if hasattr(model, 'named_steps'):
            model_info['pipeline_steps'] = list(model.named_steps.keys())
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
            model_info['expected_features'] = expected_features
        return True
    except FileNotFoundError as e:
        model_info = {'error': str(e), 'path': MODEL_PATH, 'loaded': False}
        return False
    except Exception as e:
        logger.error(traceback.format_exc())
        model_info = {'error': str(e), 'loaded': False}
        return False
load_model_from_pickle()


class Payload(BaseModel):
    customer_id: int
    product_id: int
    week: int
    customer_type: str
    Y: float
    X: float
    num_deliver_per_week: int
    brand: str
    sub_category: str
    segment: str
    package: str
    size: float
class BatchPayload(BaseModel):
    data: list[Payload]

@app.get("/")
def home():
  description = {
    "model": "Predictor de probabilidad de compra de productos con XGBoost.",
    "problem": "Determinar la probabilidad de compra de productos semanal por parte de los clientes.",
    "input": {
        'customer_id':'ID del cliente',
        'product_id':'ID del producto',
        'week':'Semana del año',
        'customer_type':'Tipo de cliente',
        'Y':'Coordenada Y del cliente',
        'X':'Coordenada X del cliente',
        'num_deliver_per_week':'Número de entregas por semana',
        'brand':'Marca del producto',
        'sub_category':'Subcategoría del producto',
        'segment':'Segmento del producto',
        'package':'Paquete del producto',
        'size':'Tamaño del producto'
    },
    "output": {
      "probabilidad_compra": "Probabilidad de que el cliente compre el producto esta semana"
    }
  }
  return description

@app.post("/predict/")
def predict(payload: Payload):
    try:
        data = payload.model_dump()
        X = pd.DataFrame([data], columns=['customer_id','product_id','week','customer_type','Y','X',
            'num_deliver_per_week','brand','sub_category','segment','package','size'])   
        prediction = float(model.predict(X)[0])
        probability= float(model.predict_proba(X)[0][1])
        return {
            "customer_id": payload.customer_id,
            "product_id": payload.product_id,
            "prediccion": float(prediction),
            "probabilidad_compra": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__": 
    uvicorn.run(app,host="0.0.0.0",port=8000,log_level="info")