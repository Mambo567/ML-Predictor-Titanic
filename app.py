from flask import Flask, request, jsonify, render_template
from sqlalchemy import create_engine, text
import pickle
import datetime
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from dotenv import load_dotenv
import os
import google.generativeai as genai
from utils import init_db, get_prompt, get_text



app = Flask(__name__)

# Cargar el modelo entrenado
with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

# Aqui definimos variables de entorno
load_dotenv()

# Definicion de variables de conexion
# Cadena de conexión a la base de datos PostgreSQL
churro = os.environ["churro"]
engine = create_engine(churro)
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]



# Llamar a la función
init_db(engine)

# Validación

@app.route('/', methods=['GET'])
def home():
    return render_template("formulario.html")


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint que recibe datos en formato JSON, realiza una predicción y guarda los datos en la base de datos.
    """
    try:
        # 1. Extraer la form. Ahora recibimos los datos del formulario del menu home
        pclass = int(request.form.get("pclass"))
        sex = int(request.form.get("sex"))
        age = int(request.form.get("age"))

    # Completa aquí: extraer valores y formatearlos para el modelo
        input_data = [[pclass, sex, age]]

        # 2. Realizar predicción con el modelo
        # Completa aquí: usa model.predict()
        prediction = model.predict(input_data)[0]
        mapita = {0 : "No Superviviente", 1 : "Superviviente"}
        prediction = mapita[prediction]
        prediction
        # 3. Guardar en la base de datos
        query = '''SELECT * FROM predictions'''
        historial = pd.read_sql(query,con=engine)
        timestamp = datetime.datetime.now().isoformat()
        registro = pd.DataFrame({"pclass": [pclass],"sex": [sex],"age": [age],"prediction": [prediction],"timestamp": [timestamp]})
        historial = pd.concat([historial,registro],axis=0)

        # Completa aquí: inserta los datos (inputs, predicción, timestamp) en la base de datos
        historial.to_sql('predictions', con=engine,if_exists="replace",index=False)

        ### Generamos la gráfica
        read_predictions = pd.read_sql(query, con=engine)
        fig = plt.figure()
        read_predictions.prediction.value_counts().plot(kind="bar",rot=0)
        plt.title("Predicciones totales")

        # Guardar la gráfica en un buffer en memoria
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig)

        # Codificar la imagen para pasarla por JSON a los resultados
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Generar texto de AI
        prompt = get_prompt(input_data,prediction)
        ai_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        text = get_text(prompt=prompt,model=ai_model)

        # Devolver el resultado y la imagen (grafica) como respuesta
        return render_template("resultado.html", prediccion=prediction, grafica=img_base64,text=text)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/records', methods=['GET'])
def records():
    """
    Endpoint que devuelve todos los registros guardados en la base de datos.
    """
    try:
        # Conectar a la base de datos y recuperar los registros
        query = '''SELECT * FROM predictions'''
        historial = pd.read_sql(query,con=engine)
        # Completa aquí: conexión SQLite y lectura de registros
        #records = []  # Sustituir por los datos recuperados de la base de datos

        return json.loads(historial.to_json(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)