from sqlalchemy import create_engine, text
import google.generativeai as genai

# Inicializar la base de datos
def init_db(engine):
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

# Función para crear prompt

def get_prompt(inputs,output):
    prompt = f"""Hola Gemini! Que bueno saludarte!. Te comento: Estoy haciendo una API en la cual se predice la supervivencia de una 
    persona en el titanic. Mi modelo utiliza los siguientes 3 inputs para hacer la predicción:
    pclass: Puede ser 1,2 o 3
    sex: es el sexo (0 es hombre, 1 es mujer)
    age: Es la edad en años
    Por medio de las anteriores variables, mi modelo hace una predicción que da el siguiente output: 'Superviviente' o 'No Superviviente'. Te voy a compartir
    los 3 inputs y el output. Quiero que por favor los evalues y cuentes una historia de entre 200 y 300 caracteres donde narres a manera de historia con mucho humor
    el porque considerarias que la persona se salvó en caso de que asi haya sido o por que no se salvó en caso de que no se haya salvado, considerando claro, las variables input.
    QUIERO QUE ME DES SOLO Y EXCLUSIVAMENTE LA HISTORIA, NO ME DIGAS NADA MAS, SOLO LA HISTORIA.
    No temas ser creativo y carismatico. Estaria bien que a veces incluyas referencias a la pelicula Titanic con Leonardo Di Caprio, con chistes como "No cabia en la balsa con Di Caprio". Te doy ese ejemplo. 
    IMPORTANTE: El formato de salida de tu respuesta debe ser únicamente y exclusivamente el texto narrado ya que en mi producto daré directamente tu respuesta.
    Importante 2: Omite todo tipo de formato enriquecido (markdown, html, etc.) dame solo texto.
    Importante 3: Para que el texto no sea muy horizontal y grande, por favor incluye varios saltos de línea para que sea mas visible para los lectores.
    Las variables input son las siguientes:
    pclass = {inputs[0][0]} sex = {inputs[0][1]} age = {inputs[0][2]}. 
    La predicción es {output}
    """
    return prompt

# Función para generar text

def get_text(prompt, model, temperature=0.7, max_output_tokens=1000, top_p=0.95, top_k=40):
    """Consulta la API de Gemini."""
    response = model.generate_content(
        contents=[{"parts": [{"text": prompt}]}], # El prompt va dentro de un objeto Content
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k
        ))
    return response.candidates[0].content.parts[0].text