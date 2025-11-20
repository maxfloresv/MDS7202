import gradio as gr
import requests
import json
import pandas as pd
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict/")

def predict_purchase_probability(
    customer_id: int,
    product_id: int,
    week: int,
    customer_type: str,
    Y: float,
    X: float,
    num_deliver_per_week: int,
    brand: str,
    sub_category: str,
    segment: str,
    package: str,
    size: float
) -> str:
    payload = {"customer_id": customer_id,
        "product_id": product_id,
        "week": week,
        "customer_type": customer_type,
        "Y": Y,
        "X": X,
        "num_deliver_per_week": num_deliver_per_week,
        "brand": brand,
        "sub_category": sub_category,
        "segment": segment,
        "package": package,
        "size": size}
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        prob = result.get("probabilidad_compra", "N/A")
        pred = result.get("prediccion", "N/A")
        return (f"¿Va a comprar el producto?: {'Sí' if pred == 1.0 else 'No'}\n"
            f"¿Cuál es la probabilidad de compra?: {prob:.4f}")

    except requests.exceptions.RequestException as e:
        return f"Error en la request: {e}"
    except Exception as e:
        return f"Error: {e}"
inputs = [gr.Number(label="ID de Cliente", value=100, precision=0),
    gr.Number(label="ID de Producto", value=200, precision=0),
    gr.Number(label="Semana del Año", minimum=1, maximum=52, value=40, precision=0),
    gr.Dropdown(label="Tipo de Cliente", 
        choices=['ABARROTES', 'MAYORISTA', 'TIENDA DE CONVENIENCIA', 'CANAL FRIO',
                 'RESTAURANT', 'SUPERMERCADO', 'MINIMARKET'], 
        value='ABARROTES'),
    gr.Number(label="Latitud", value=45.123),
    gr.Number(label="Longitud", value=-75.456),
    gr.Number(label="Entregas por Semana", minimum=0, value=2, precision=0),
    gr.Textbox(label="Marca", value='Brand 31'),
    gr.Dropdown(label="Subcategoría", 
        choices=['GASEOSAS', 'AGUAS SABORIZADAS', 'JUGOS'], 
        value='GASEOSAS'),
    gr.Dropdown(label="Segmento", 
        choices=['HIGH', 'LOW', 'MEDIUM', 'PREMIUM'], 
        value='HIGH'), 
    gr.Dropdown(label="Empaque", 
        choices=['BOTELLA', 'KEG', 'LATA', 'TETRA'], 
        value='BOTELLA'),
    gr.Number(label="Tamaño", minimum=0.05, maximum=50, value=0.5)]

output = gr.Textbox(label="Resultado de la Predicción", lines=3)
description_text="""
## Instrucciones:
1. Ingresa el ID del cliente y sus datos principales, como tipo de cliente, ubicación y frecuencia de entrega.
2. Proporciona el ID del producto, semana del año y características del producto como marca, subcategoría, segmento, paquete y tamaño.
3. Haz clic en **"Submit"** para obtener la predicción de compra y su probabilidad.
4. Si desea limpiar los datos anteriores, use el botón **"Clear"**.
"""
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Roboto&display=swap');
* { font-family: 'Roboto', sans-serif !important; }
"""

iface = gr.Interface(
    fn=predict_purchase_probability,
    inputs=inputs,
    outputs=output,
    title="SodAI Drinks - Predictor de Probabilidad de Compra",
    description=description_text,
    article="Desarrollado por Naomi Cautivo B. y Máximo Flores Valenzuela. (**Borelian**)",
    flagging_mode='never',
    theme=gr.themes.Soft(),
    css=custom_css)

if __name__ == "__main__":
    iface.launch(server_port=7860,server_name="0.0.0.0")