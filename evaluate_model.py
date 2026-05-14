import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("heatstrokedataset.csv")

print("Columns:", df.columns)

# Features (correct mapping)
X = df[['ambient_temp', 'humidity', 'body_temp', 'duration', 'hour', 'heat_index']]

# Target
y = df['risk_label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load model
model = joblib.load("model.pkl")

# Predict
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)

print("\n MODEL PERFORMANCE\n")
print("Accuracy:", round(accuracy * 100, 2), "%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))