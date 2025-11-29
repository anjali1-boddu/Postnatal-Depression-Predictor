from flask import Flask, request, render_template
import pickle, json
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
with open('feature_columns.json', 'r') as f:
    FEATURE_COLS = json.load(f)

# Encoding maps
YN_MAP = {
    'Yes': 1.0, 'No': 0.0, 'Maybe': 0.5, 'Sometimes': 0.5,
    'Two or more days a week': 1.0, 'nan': 0.0, 'None': 0.0, '': 0.0
}

AGE_MAP = {
    '20-25': 22.5, '25-30': 27.5, '30-35': 32.5,
    '35-40': 37.5, '40-45': 42.5, '45-50': 47.5
}

def preprocess_one(payload: dict) -> pd.DataFrame:
    """Clean and transform form data into model-ready format"""
    df = pd.DataFrame([payload])
    df = df.apply(lambda s: s.astype(str).str.strip())
    df = df.replace(YN_MAP)
    if 'Age' in df.columns:
        df['Age'] = df['Age'].replace(AGE_MAP)
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(0.0)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df = df[FEATURE_COLS]
    return df

@app.route('/')
def home():
    return render_template('index.html', prediction=None,color="#f0f0f0")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form.to_dict()
        X = preprocess_one(form_data)
        X_scaled = scaler.transform(X)
        pred = int(model.predict(X_scaled)[0])
        prob = model.predict_proba(X_scaled)[0, 1] if hasattr(model, "predict_proba") else None

        message = "High Risk 😔" if pred == 1 else "Low Risk 😊"
        color = "#ff4c4c" if pred == 1 else "#4caf50"

        return render_template(
            'index.html',
            prediction=message,
            color=color,
            prob=round(prob * 100, 2) if prob else None
        )
    except Exception as e:
        return render_template(
            'index.html',
            prediction=f"Error: {str(e)}",
            color="#ff9800"
        )

if __name__ == "__main__":
    app.run(debug=True)
