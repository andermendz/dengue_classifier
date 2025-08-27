from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier # <-- IMPORT THE NEW MODEL
import os
from dotenv import load_dotenv
import google.generativeai as genai
import eventlet

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

# --- DATA PREPROCESSING AND MODEL TRAINING (RUNS ONCE AT STARTUP) ---

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

# Preprocess binary (Yes/No) columns
symptom_columns = ['fiebre', 'cefalea', 'dolrretroo', 'malgias', 'artralgia', 'erupcionr', 'dolor_abdo', 'vomito', 'diarrea', 'hipotensio', 'hepatomeg']
for col in symptom_columns:
    if data[col].dtype == 'object':
        data[col] = data[col].str.strip().str.lower().map({'si': 1, 'no': 0, '1': 1, '0': 0})
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)

# --- NEW: FEATURE ENGINEERING ---
# Create a new feature for the total count of symptoms
data['symptom_count'] = data[symptom_columns].sum(axis=1)
print("New 'symptom_count' feature created.")

# Preprocess 'sexo' column
if data['sexo'].dtype == 'object':
    data['sexo'] = data['sexo'].str.strip().str.upper().map({'M': 0, 'F': 1})

# Encode other categorical columns
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
features = ['def_clas_edad', 'sexo', 'symptom_count'] + symptom_columns # <-- ADDED symptom_count
X = data[features].values
y_clasfinal = data['clasfinal']
y_conducta = data['conducta']

X_train, _, y_train, _ = train_test_split(X, np.column_stack((y_clasfinal, y_conducta)), test_size=0.2, random_state=42)
y_clasfinal_train, y_conducta_train = y_train[:, 0], y_train[:, 1]


# --- NEW: HYPERPARAMETER TUNING WITH GridSearchCV ---
def find_best_model(X, y, model_name):
    print(f"\n--- Tuning model for: {model_name} ---")
    
    # Define a smaller, more efficient parameter grid for faster startup
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0]
    }
    
    xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
    
    # Use GridSearchCV to find the best parameters
    # cv=3 (3-fold cross-validation) is a good balance for speed and robustness
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy for {model_name}: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# Find the best models for both targets
model_clasfinal = find_best_model(X_train, y_clasfinal_train, 'clasfinal')
model_conducta = find_best_model(X_train, y_conducta_train, 'conducta')

print("\nOptimized models trained successfully.")


# --- FLASK AND SOCKET.IO ROUTES ---

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
        
        # --- UPDATE: Prepare feature vector with the new 'symptom_count' ---
        symptoms_values = [int(form_data.get(symptom, 0)) for symptom in symptom_columns]
        symptom_count = sum(symptoms_values)
        
        user_data = [
            label_encoders['def_clas_edad'].transform([categorize_age(edad)])[0],
            1 if sexo == 'F' else 0,
            symptom_count, # Add the new feature
            *symptoms_values # Unpack the rest of the symptom values
        ]

        # Run local ML predictions
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
        
        model = genai.GenerativeModel('gemini-2.5-flash')
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
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)