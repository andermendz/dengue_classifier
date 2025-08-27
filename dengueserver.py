from flask import Flask, render_template, request # <-- THE FIX IS HERE
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
from dotenv import load_dotenv
import google.generativeai as genai
import eventlet # Required for production-ready Socket.IO server

# Load environment variables from .env file
load_dotenv()

# Initialize Flask and Socket.IO
app = Flask(__name__, template_folder='.')
socketio = SocketIO(app, async_mode='eventlet')

# Configure the Gemini client using the API key from the environment
try:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise KeyError
    genai.configure(api_key=gemini_api_key)
    gemini_enabled = True
    print("Gemini AI client configured successfully.")
except KeyError:
    print("WARNING: GEMINI_API_KEY not found in .env file. The AI assistant will be disabled.")
    gemini_enabled = False

# --- DATA PREPROCESSING AND MODEL TRAINING (RUNS ONCE AT STARTUP) ---

# Dictionary to store label encoders for later use
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
    print("FATAL ERROR: 'dengue2.csv' not found. The application cannot start without the dataset.")
    exit()

# Preprocess binary (Yes/No) columns
yes_no_columns = ['fiebre', 'cefalea', 'dolrretroo', 'malgias', 'artralgia', 'erupcionr', 'dolor_abdo', 'vomito', 'diarrea', 'hipotensio', 'hepatomeg']
for col in yes_no_columns:
    if data[col].dtype == 'object':
        data[col] = data[col].str.strip().str.lower().map({'si': 1, 'no': 0, '1': 1, '0': 0})
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)

# Preprocess 'sexo' column
if data['sexo'].dtype == 'object':
    data['sexo'] = data['sexo'].str.strip().str.upper().map({'M': 0, 'F': 1})

# Encode other categorical columns using LabelEncoder
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

# Define features and split data for training
features = ['def_clas_edad', 'sexo', 'fiebre', 'cefalea', 'dolrretroo', 'malgias', 'artralgia', 'erupcionr', 'dolor_abdo', 'vomito', 'diarrea', 'hipotensio', 'hepatomeg']
X = data[features].values
y_clasfinal = data['clasfinal']
y_conducta = data['conducta']

X_train, _, y_train, _ = train_test_split(X, np.column_stack((y_clasfinal, y_conducta)), test_size=0.2, random_state=42)
y_clasfinal_train, y_conducta_train = y_train[:, 0], y_train[:, 1]

# Train the classification models
model_clasfinal = RandomForestClassifier(n_estimators=100, random_state=42)
model_conducta = RandomForestClassifier(n_estimators=100, random_state=42)
model_clasfinal.fit(X_train, y_clasfinal_train)
model_conducta.fit(X_train, y_conducta_train)

print("Models trained successfully.")

# --- FLASK AND SOCKET.IO ROUTES ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handles a new client connection."""
    print(f'Client connected: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    """Handles a client disconnection."""
    print(f'Client disconnected: {request.sid}')

@socketio.on('submit_prediction')
def handle_prediction(form_data):
    """Receives form data via Socket.IO, runs predictions, and emits results."""
    try:
        # --- 1. Process incoming data from the form ---
        edad = int(form_data.get('edad', 0))
        sexo = form_data.get('sexo', 'M')
        
        # Prepare the feature vector for the local model
        user_data = [
            label_encoders['def_clas_edad'].transform([categorize_age(edad)])[0],
            1 if sexo == 'F' else 0,
            int(form_data.get('fiebre', 0)), int(form_data.get('cefalea', 0)), int(form_data.get('dolrretroo', 0)),
            int(form_data.get('malgias', 0)), int(form_data.get('artralgia', 0)), int(form_data.get('erupcionr', 0)),
            int(form_data.get('dolor_abdo', 0)), int(form_data.get('vomito', 0)), int(form_data.get('diarrea', 0)),
            int(form_data.get('hipotensio', 0)), int(form_data.get('hepatomeg', 0))
        ]

        # --- 2. Run local ML predictions ---
        predicted_clasfinal_enc = model_clasfinal.predict([user_data])[0]
        predicted_conducta_enc = model_conducta.predict([user_data])[0]

        predicted_clasfinal = label_encoders['clasfinal'].inverse_transform([predicted_clasfinal_enc])[0]
        predicted_conducta = label_encoders['conducta'].inverse_transform([predicted_conducta_enc])[0]

        # --- 3. Emit the local prediction back to the client immediately ---
        emit('classified_answer', {
            'predicted_clasfinal': str(predicted_clasfinal),
            'predicted_conducta': str(predicted_conducta)
        })
        
        # --- 4. Call the Gemini LLM for a descriptive recommendation ---
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
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response_stream = model.generate_content(prompt, stream=True)
        
        for chunk in response_stream:
            if chunk.text:
                emit('ai_response', {'data': chunk.text})
                socketio.sleep(0.05) # A small sleep allows the server to send the message

    except Exception as e:
        print(f"An error occurred during prediction for SID {request.sid}: {e}")
        emit('ai_response', {'data': f"\n\n[Ha ocurrido un error en el servidor: {e}]"})

if __name__ == '__main__':
    print("Starting Flask-SocketIO server on http://127.0.0.1:5000")
    # Use eventlet as the web server, which is ideal for Socket.IO
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)