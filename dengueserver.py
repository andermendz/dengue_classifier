from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import os
from dotenv import load_dotenv
import google.generativeai as genai
import eventlet
import joblib

# Load environment variables from .env file
load_dotenv()

# Initialize Flask and Socket.IO
app = Flask(__name__, template_folder='.')
socketio = SocketIO(app, async_mode='eventlet')

# Configure the Gemini client
try:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise KeyError
    genai.configure(api_key=gemini_api_key)
    gemini_enabled = True
    print("Gemini AI client configured successfully.")
except KeyError:
    print("WARNING: GEMINI_API_KEY not found. AI assistant will be disabled.")
    gemini_enabled = False

# --- NUEVO: Definir las rutas de los archivos de los modelos guardados ---
MODEL_CLASFINAL_PATH = 'modelo_clasfinal.joblib'
MODEL_CONDUCTA_PATH = 'modelo_conducta.joblib'

# --- DATA PREPROCESSING (SIEMPRE SE EJECUTA PORQUE ES RÁPIDO) ---

label_encoders = {}

def categorize_age(edad):
    if edad < 2: return 'PRIMERA INFANCIA'
    elif edad < 12: return 'INFANCIA'
    elif edad < 18: return 'ADOLESCENCIA'
    elif edad < 30: return 'JOVENES'
    elif edad < 60: return 'ADULTEZ'
    else: return 'PERSONA MAYOR'

try:
    data = pd.read_csv('dengue2.csv')
except FileNotFoundError:
    print("FATAL ERROR: 'dengue2.csv' not found. The application cannot start.")
    exit()

symptom_columns = ['fiebre', 'cefalea', 'dolrretroo', 'malgias', 'artralgia', 'erupcionr', 'dolor_abdo', 'vomito', 'diarrea', 'hipotensio', 'hepatomeg']
for col in symptom_columns:
    if data[col].dtype == 'object':
        data[col] = data[col].str.strip().str.lower().map({'si': 1, 'no': 0, '1': 1, '0': 0})
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)

data['symptom_count'] = data[symptom_columns].sum(axis=1)

if data['sexo'].dtype == 'object':
    data['sexo'] = data['sexo'].str.strip().str.upper().map({'M': 0, 'F': 1})

label_columns = ['clasfinal', 'conducta', 'def_clas_edad']
for col in label_columns:
    le = LabelEncoder()
    if col == 'def_clas_edad':
        all_categories = ['PRIMERA INFANCIA', 'INFANCIA', 'ADOLESCENCIA', 'JOVENES', 'ADULTEZ', 'PERSONA MAYOR']
        le.fit(all_categories)
    else:
        le.fit(data[col].astype(str))
    data[col] = le.transform(data[col].astype(str))
    label_encoders[col] = le



# Comprobar si los modelos ya existen
if os.path.exists(MODEL_CLASFINAL_PATH) and os.path.exists(MODEL_CONDUCTA_PATH):
    print("\nCargando modelos pre-entrenados desde el disco...")
    model_clasfinal = joblib.load(MODEL_CLASFINAL_PATH)
    model_conducta = joblib.load(MODEL_CONDUCTA_PATH)
    print("Modelos cargados exitosamente.")
else:
    # Si los modelos no existen, ejecutar el proceso de entrenamiento
    print("\nModelos no encontrados. Iniciando proceso de entrenamiento único...")
    
    features = ['def_clas_edad', 'sexo', 'symptom_count'] + symptom_columns
    X = data[features].values
    y_clasfinal = data['clasfinal']
    y_conducta = data['conducta']

    X_train, _, y_train, _ = train_test_split(X, np.column_stack((y_clasfinal, y_conducta)), test_size=0.2, random_state=42)
    y_clasfinal_train, y_conducta_train = y_train[:, 0], y_train[:, 1]

    def find_best_model(X, y, model_name):
        print(f"\n--- Optimizando modelo para: {model_name} ---")
        param_grid = {
            'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1],
            'subsample': [0.7, 1.0], 'colsample_bytree': [0.7, 1.0]
        }
        xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
        grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X, y)
        print(f"Mejores parámetros para {model_name}: {grid_search.best_params_}")
        print(f"Mejor precisión (cross-validation) para {model_name}: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_

    model_clasfinal = find_best_model(X_train, y_clasfinal_train, 'clasfinal')
    model_conducta = find_best_model(X_train, y_conducta_train, 'conducta')

    # --- NUEVO: Guardar los modelos entrenados en el disco ---
    print("\nGuardando modelos entrenados en el disco para uso futuro...")
    joblib.dump(model_clasfinal, MODEL_CLASFINAL_PATH)
    joblib.dump(model_conducta, MODEL_CONDUCTA_PATH)
    print("Modelos guardados exitosamente.")

# --- FLASK AND SOCKET.IO ROUTES (SIN CAMBIOS) ---

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')

@socketio.on('submit_prediction')
def handle_prediction(form_data):
    try:
        edad = int(form_data.get('edad', 0))
        sexo = form_data.get('sexo', 'M')
        
        symptoms_values = [int(form_data.get(symptom, 0)) for symptom in symptom_columns]
        symptom_count = sum(symptoms_values)
        
        user_data = [
            label_encoders['def_clas_edad'].transform([categorize_age(edad)])[0],
            1 if sexo == 'F' else 0,
            symptom_count,
            *symptoms_values
        ]

        predicted_clasfinal_enc = model_clasfinal.predict([user_data])[0]
        predicted_conducta_enc = model_conducta.predict([user_data])[0]

        predicted_clasfinal = label_encoders['clasfinal'].inverse_transform([predicted_clasfinal_enc])[0]
        predicted_conducta = label_encoders['conducta'].inverse_transform([predicted_conducta_enc])[0]

        emit('classified_answer', {
            'predicted_clasfinal': str(predicted_clasfinal),
            'predicted_conducta': str(predicted_conducta)
        })
        
        if not gemini_enabled:
            emit('ai_response', {'data': "El asistente de IA no está disponible en este momento."})
            return

        prompt = (
            "Eres un asistente virtual médico especializado en dengue. Tu tono es profesional, tranquilizador y claro. "
            "Un modelo predictivo ha analizado los síntomas de un paciente y ha determinado lo siguiente:\n"
            f"- Clasificación de Dengue: **{predicted_clasfinal}**\n"
            f"- Conducta Sugerida: **{predicted_conducta}**\n\n"
            "Basado en esta información, proporciona una recomendación concisa (entre 60 y 80 palabras) para el paciente. "
            "La recomendación debe explicar brevemente lo que significa la clasificación, qué debe hacer a continuación (por ejemplo, 'manejo en casa con hidratación' o 'acudir a un centro médico para evaluación'), "
            "y enfatizar la importancia de no automedicarse y de buscar atención médica si los síntomas empeoran."
        )
        
        model = genai.GenerativeModel('gemini-1.5-flash') # Actualizado a un modelo más reciente
        response_stream = model.generate_content(prompt, stream=True)
        
        for chunk in response_stream:
            if chunk.text:
                emit('ai_response', {'data': chunk.text})
                socketio.sleep(0.05)

    except Exception as e:
        print(f"An error occurred during prediction for SID {request.sid}: {e}")
        emit('ai_response', {'data': f"\n\n[Ha ocurrido un error en el servidor: {e}]"})

if __name__ == '__main__':
    print("Starting Flask-SocketIO server on http://127.0.0.1:5000")
    socketio.run(app, host='0.0.0.0', port=5000)