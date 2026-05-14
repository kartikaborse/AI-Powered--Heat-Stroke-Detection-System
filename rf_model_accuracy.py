import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("heatstrokedataset.csv")

# MUST MATCH training features EXACTLY
df["temp_humidity_stress"] = df["ambient_temp"] * df["humidity"] / 100
df["body_temp_deviation"] = df["body_temp"] - 37.0

def time_category(hour):
    if 5 <= hour < 12:
        return 0
    elif 12 <= hour < 17:
        return 1
    elif 17 <= hour < 21:
        return 2
    else:
        return 3

df["time_of_day"] = df["hour"].apply(time_category)

feature_cols = [
    "ambient_temp",
    "humidity",
    "body_temp",
    "duration",
    "hour",
    "heat_index",
    "temp_humidity_stress",
    "body_temp_deviation",
    "time_of_day"
]

X = df[feature_cols]
y = df["risk_label"]

data = joblib.load("rf_model.pkl")
model = data["model"] if isinstance(data, dict) else data

y_pred = model.predict(X)

print("\n===== MODEL ACCURACY =====")
print("Accuracy:", accuracy_score(y, y_pred) * 100, "%")

print("\nClassification Report:\n")
print(classification_report(y, y_pred))