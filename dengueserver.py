from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from g4f.client import AsyncClient
import asyncio

app = Flask(__name__, template_folder='.')
socketio = SocketIO(app)
client = AsyncClient()

# Diccionario para almacenar los codificadores de etiquetas
label_encoders = {}

# Función para categorizar la edad
def categorize_age(edad):
    if edad < 2:
        return 'PRIMERA INFANCIA'
    elif edad < 12:
        return 'INFANCIA'
    elif edad < 18:
        return 'ADOLESCENCIA'
    elif edad < 30:
        return 'JOVENES'
    elif edad < 60:
        return 'ADULTEZ'
    else:
        return 'PERSONA MAYOR'

# Preprocesamiento y entrenamiento del modelo (UNA SOLA VEZ)
data = pd.read_csv('dengue2.csv')

# Convertir 'Si'/'No' a 1/0 y 'sexo' a 0/1
yes_no_columns = ['fiebre', 'cefalea', 'dolrretroo', 'malgias', 'artralgia', 'erupcionr', 'dolor_abdo', 'vomito', 'diarrea', 'hipotensio', 'hepatomeg']
for col in yes_no_columns:
    data[col] = data[col].map({1: 1, 0: 0})
data['sexo'] = data['sexo'].map({'M': 0, 'F': 1})

# Codificar otras variables categóricas
label_columns = ['clasfinal', 'conducta', 'def_clas_edad']  # Incluir def_clas_edad
for col in label_columns:
    le = LabelEncoder()
    if col == 'def_clas_edad':
        # Codificar def_clas_edad con todas las categorías posibles
        all_possible_categories = np.array(['PRIMERA INFANCIA', 'INFANCIA', 'ADOLESCENCIA', 'JOVENES', 'ADULTEZ', 'PERSONA MAYOR'])
        le.fit(all_possible_categories)
    else:
        le.fit(data[col])
    data[col] = le.transform(data[col])
    label_encoders[col] = le

# Definir características
features = ['def_clas_edad', 'sexo', 'fiebre', 'cefalea', 'dolrretroo', 'malgias', 'artralgia', 'erupcionr', 'dolor_abdo', 'vomito', 'diarrea', 'hipotensio', 'hepatomeg']

# Dividir datos
X = data[features].values
y_clasfinal = data['clasfinal']
y_conducta = data['conducta']
X_train, X_test, y_train, y_test = train_test_split(X, np.column_stack((y_clasfinal, y_conducta)), test_size=0.2, random_state=42)
y_clasfinal_train, y_conducta_train = y_train[:, 0], y_train[:, 1]

# Entrenar modelos
model_clasfinal = RandomForestClassifier(n_estimators=100, random_state=42)
model_conducta = RandomForestClassifier(n_estimators=100, random_state=42)
model_clasfinal.fit(X_train, y_clasfinal_train)
model_conducta.fit(X_train, y_conducta_train)

# Manejar conexión del cliente
@socketio.on('connect')
def handle_connect():
    print('Client connected')

# Ruta principal
@app.route('/', methods=['GET', 'POST'])
async def index():
    if request.method == 'POST':
        # Obtener datos del formulario
        edad = int(request.form.get('edad'))
        sexo = request.form.get('sexo')
        sexo = 'Masculino' if sexo == 'M' else 'Femenino'
        fiebre = 'Sí' if request.form.get('fiebre', 0) == '1' else 'No'
        cefalea = 'Sí' if request.form.get('cefalea', 0) == '1' else 'No'
        dolrretroo = 'Sí' if request.form.get('dolrretroo', 0) == '1' else 'No'
        malgias = 'Sí' if request.form.get('malgias', 0) == '1' else 'No'
        artralgia = 'Sí' if request.form.get('artralgia', 0) == '1' else 'No'
        erupcionr = 'Sí' if request.form.get('erupcionr', 0) == '1' else 'No'
        dolor_abdo = 'Sí' if request.form.get('dolor_abdo', 0) == '1' else 'No'
        vomito = 'Sí' if request.form.get('vomito', 0) == '1' else 'No'
        diarrea = 'Sí' if request.form.get('diarrea', 0) == '1' else 'No'
        hipotensio = 'Sí' if request.form.get('hipotensio', 0) == '1' else 'No'
        hepatomeg = 'Sí' if request.form.get('hepatomeg', 0) == '1' else 'No'

        # Preparar datos del usuario
        user_data = [label_encoders['def_clas_edad'].transform([categorize_age(edad)])[0]] + [1 if sexo == 'Masculino' else 0, 1 if fiebre == 'Sí' else 0, 1 if cefalea == 'Sí' else 0, 1 if dolrretroo == 'Sí' else 0, 1 if malgias == 'Sí' else 0, 1 if artralgia == 'Sí' else 0, 1 if erupcionr == 'Sí' else 0, 1 if dolor_abdo == 'Sí' else 0, 1 if vomito == 'Sí' else 0, 1 if diarrea == 'Sí' else 0, 1 if hipotensio == 'Sí' else 0, 1 if hepatomeg == 'Sí' else 0]

        # Predicciones
        predicted_clasfinal = model_clasfinal.predict([user_data])[0]
        predicted_conducta = model_conducta.predict([user_data])[0]

        # Decodificar y emitir predicciones
        predicted_clasfinal = label_encoders['clasfinal'].inverse_transform([predicted_clasfinal])[0]
        predicted_conducta = label_encoders['conducta'].inverse_transform([predicted_conducta])[0]
        socketio.emit('classified_answer', {'predicted_clasfinal': predicted_clasfinal, 'predicted_conducta': predicted_conducta})

        # Solicitud a LLM
        user_symptoms = f"Edad: {categorize_age(edad)}, Sexo: {sexo}, Fiebre: {fiebre}, Dolor de cabeza: {cefalea}, Dolor detrás de los ojos: {dolrretroo}, Dolores musculares: {malgias}, Dolor en las articulaciones: {artralgia}, Erupción cutánea: {erupcionr}, Dolor abdominal: {dolor_abdo}, Vómito: {vomito}, Diarrea: {diarrea}, Hipotensión: {hipotensio}, Hepatomegalia: {hepatomeg}"
        stream = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente virtual médico especializado en el tratamiento del dengue. Tu función es brindar recomendaciones de atención médica (hospitalización o manejo ambulatorio) a los pacientes en función de sus síntomas y la gravedad de la enfermedad. Los datos que recibirás del paciente son: - Edad - Sexo - Síntomas (fiebre, dolor de cabeza, dolor detrás de los ojos, dolores musculares, dolor en las articulaciones, erupción cutánea, dolor abdominal, vómito, diarrea, hipotensión, hepatomegalia) - Clasificación final predicha del dengue (sin señal de alarma, con señal de alarma, dengue grave) - Conducta predicha (ambulatorio u hospitalización). Debes proporcionar una respuesta breve pero concisa, recomendando el curso de acción adecuado (hospitalización o manejo ambulatorio) según la gravedad de los síntomas y la clasificación final del dengue. Tu recomendación debe ser clara y fácil de entender para el paciente."},
                {"role": "user", "content": f"Dado los siguientes síntomas: {user_symptoms}, y la clasificación final predicha es: {predicted_clasfinal}, cuya conducta puede ser {predicted_conducta}, dame una respuesta entre 60 y 80 palabras"}
            ],
            stream=True
        )

        gpt_response = ""
        async for chunk in stream:
            partial_response = chunk.choices[0].delta.content or ""
            gpt_response += partial_response
            socketio.emit('ai_response', {'data': partial_response})

    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)
