import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle, json

# Load and clean
data = pd.read_csv("post natal data.csv")
data = data.drop(columns=['Timestamp'], errors='ignore')
data = data.apply(lambda s: s.astype(str).str.strip())

# Convert categorical answers → numeric
yn_map = {
    'Yes': 1.0,
    'No': 0.0,
    'Maybe': 0.5,
    'Sometimes': 0.5,
    'Two or more days a week': 1.0,
    'nan': 0.0,
    'None': 0.0,
    '': 0.0
}
data = data.replace(yn_map)

# Handle Age mapping
if 'Age' in data.columns:
    age_map = {
        '20-25': 22.5, '25-30': 27.5, '30-35': 32.5,
        '35-40': 37.5, '40-45': 42.5, '45-50': 47.5
    }
    data['Age'] = data['Age'].replace(age_map)
    data['Age'] = pd.to_numeric(data['Age'], errors='coerce').fillna(0.0)

# ✅ Use only the same fields as in your HTML form
used_cols = [
    'Age',
    'Feeling sad or Tearful',
    'Irritable towards baby & partner',
    'Trouble sleeping at night',
    'Feeling anxious'
]

# Keep only available columns
data = data[[c for c in used_cols if c in data.columns]]

# 🔒 Ensure all columns are numeric before summing
data = data.apply(pd.to_numeric, errors='coerce').fillna(0.0)

# ✅ Create realistic target: high risk if ≥3 “Yes” symptoms
data['DepressionStatus'] = (data.drop('Age', axis=1).sum(axis=1) >= 3).astype(int)

# Split data
X = data.drop('DepressionStatus', axis=1)
y = data['DepressionStatus']

# Save training columns for Flask app
feature_cols = list(X.columns)
with open('feature_columns.json', 'w') as f:
    json.dump(feature_cols, f)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("✅ Model retrained successfully!")
print(f"Training Accuracy: {model.score(X_train, y_train):.2f}")
print(f"Testing Accuracy: {model.score(X_test, y_test):.2f}")
