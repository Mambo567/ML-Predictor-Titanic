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


app = Flask(__name__)

# Cargar el modelo entrenado
with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

# Definicion de variables de conexion
# Cadena de conexión a la base de datos PostgreSQL
churro = "postgresql://postgres:postgres@104.199.93.74/postgres"
engine = create_engine(churro)

# Inicializar la base de datos
def init_db():
    # Conectar a la base de datos y crear la tabla 'predictions' si no existe
    # Completa aquí: conexión SQLite y creación de tabla con campos (inputs, prediction, timestamp)
    # Crear la tabla 'predictions' con SQL
    create_table_query = """
    CREATE TABLE IF NOT EXISTS predictions (
        pclass INT NOT NULL,
        sex INT NOT NULL,
        age INT NOT NULL,
        prediction INT NOT NULL,
        timestamp TIMESTAMP NOT NULL
    );
    """
    with engine.connect() as connection:
        connection.execute(text(create_table_query))
    print("Tabla 'predictions' creada si no existía.")

# Llamar a la función
init_db()

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
        read_predictions.prediction.value_counts().plot(kind="bar")
        plt.title("Predicciones totales")

        # Guardar la gráfica en un buffer en memoria
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig)

        # Codificar la imagen para pasarla por JSON a los resultados
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Devolver el resultado y la imagen (grafica) como respuesta
        return render_template("resultado.html", prediccion=prediction, grafica=img_base64)
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
    app.run(debug=True)