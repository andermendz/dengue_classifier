import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Paso 1: Cargar el dataset y preprocesamiento de datos
data = pd.read_csv('dengue.csv')

# Convertir variables categóricas 'Si'/'No' a 1/0 y 'sexo' a 0/1
yes_no_columns = ['fiebre', 'cefalea', 'dolrretroo', 'malgias', 'artralgia', 'erupcionr', 'dolor_abdo', 'vomito', 'diarrea', 
                  'hipotensio', 'hepatomeg', 'pac_hos']
for col in yes_no_columns:
    data[col] = data[col].map({'Si': 1, 'No': 0})

# Mapping 'sexo' to 0/1
data['sexo'] = data['sexo'].map({'M': 0, 'F': 1})

# Encoding other categorical variables using LabelEncoder
label_columns = ['tip_cas', 'clasfinal', 'conducta']
label_encoders = {}
for col in label_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Paso 2: División de datos
features = ['edad', 'sexo', 'fiebre', 'diarrea']  # Adjust based on relevant features
X = data[features]
y_clasfinal = data['clasfinal']
y_tip_cas = data['tip_cas']
X_train, X_test, y_clasfinal_train, y_clasfinal_test, y_tip_cas_train, y_tip_cas_test = train_test_split(X, y_clasfinal, y_tip_cas, test_size=0.2, random_state=42)

# Paso 3 y 4: Selección y entrenamiento del modelo
model_clasfinal = RandomForestClassifier(n_estimators=100, random_state=42)
model_tip_cas = RandomForestClassifier(n_estimators=100, random_state=42)
model_clasfinal.fit(X_train, y_clasfinal_train)
model_tip_cas.fit(X_train, y_tip_cas_train)

# Paso 7: Predicción
new_data = pd.DataFrame([[4, 0, 0, 0]], columns=features)  # Adjusted to use numeric values directly
predicted_clasfinal = model_clasfinal.predict(new_data)
predicted_tip_cas = model_tip_cas.predict(new_data)
print(f'Predicted clasfinal: {label_encoders["clasfinal"].inverse_transform(predicted_clasfinal)[0]}')
print(f'Predicted tip_cas: {label_encoders["tip_cas"].inverse_transform(predicted_tip_cas)[0]}')
