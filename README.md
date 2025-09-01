# Dengue Classifier

Small Flask + Socket.IO app that predicts dengue classification and suggested conduct using XGBoost models.

## Project structure

- `dengueserver.py` - Main Flask + Socket.IO server. Loads data, trains models (if missing), then serves predictions and streams an AI recommendation.
- `dengue2.csv` - Training data used to build models (required to train models).
- `modelo_clasfinal.joblib`, `modelo_conducta.joblib` - Pretrained models (saved by the server after training).
- `templates/` and `static/` - Frontend assets and HTML templates.
- `index.html`, `dengue.html` - Example UI files.

## Requirements

This project targets Python 3.8+ and depends on the following packages:

- flask
- flask_socketio
- pandas
- numpy
- scikit-learn
- xgboost
- python-dotenv
- google-generativeai (optional — only if you want the AI assistant)
- eventlet
- joblib

Install them (recommended in a virtualenv):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install flask flask_socketio pandas numpy scikit-learn xgboost python-dotenv google-generativeai eventlet joblib
```

## Environment

Create a `.env` file in the project root for optional configuration:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

- If `GEMINI_API_KEY` is not provided, the server will still run but the AI assistant responses will be disabled.

## Running the app

From the project root (PowerShell on Windows):

```powershell
# optional: activate your virtual environment first
python dengueserver.py
```

The server listens on port 5000 by default and prints a startup message:

```
Starting Flask-SocketIO server on http://127.0.0.1:5000
```

Open `http://127.0.0.1:5000` in a browser to use the UI.

## How the server behaves

- On startup, `dengueserver.py` attempts to load `modelo_clasfinal.joblib` and `modelo_conducta.joblib`.
  - If both exist, they are loaded and used for predictions.
  - If either is missing, the server reads `dengue2.csv`, pre-processes data, trains two XGBoost classifiers with a GridSearchCV hyperparameter search, and saves the resulting models to disk.
- Socket.IO event handlers:
  - `submit_prediction` — accepts form data (age, sex, symptom flags), runs the models, emits `classified_answer` with predicted labels, and streams an AI recommendation via `ai_response` if Gemini is enabled.

## Retraining models

To retrain models from scratch:

1. Ensure `dengue2.csv` is present and formatted as expected by `dengueserver.py`.
2. Remove or rename the existing `modelo_clasfinal.joblib` and `modelo_conducta.joblib` files.
3. Start the server; it will perform the training run and save new `.joblib` files.

Note: training uses GridSearchCV and may take time depending on your hardware.

## Data expectations / labels

The server expects symptom columns with names used in the code (for example: `fiebre`, `cefalea`, `dolrretroo`, `malgias`, etc.). Non-numeric entries like `si`/`no` are handled by the preprocessing step.

Labels encoded by the server include `clasfinal`, `conducta`, and `def_clas_edad` (age buckets). Label encoders are fitted from the training CSV.

## Troubleshooting

- If `dengue2.csv` is missing the server will exit with a message: `FATAL ERROR: 'dengue2.csv' not found.`
- If the AI assistant does not respond, confirm `GEMINI_API_KEY` is set and valid.
- For issues with XGBoost installation on Windows, consult XGBoost docs for proper wheel or build instructions.

## Notes & next steps

- Consider adding a `requirements.txt` or `pyproject.toml` for reproducible installs.
- Add unit tests for preprocessing and prediction handlers.
- If you want CI to run training or smoke tests, include a small sample CSV with sanitized data.

---

Created for the `dengue_classifier` repository.
